// Copyright (C) 2026 David Reveman.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::cupti_profiler::{self as profiler, *};
use crate::state::{KernelActivity, KernelLaunch, MemcpyActivity, MemsetActivity, GLOBAL_STATE};
use libc::c_void;
use perfetto_gpu_compute_injection::tracing::{
    get_counter_config, get_track_event_data_source, is_cuda_driver_debug_enabled_for,
    is_cuda_driver_enabled_for, is_cuda_events_enabled, is_cuda_runtime_debug_enabled_for,
    is_cuda_runtime_enabled_for, is_instrumented_enabled, is_tx_events_enabled,
    is_tx_events_enabled_for, process_track_uuid, thread_track_uuid, trace_time_ns,
    TrackEventIncrState,
};
use perfetto_gpu_compute_injection::{injection_fatal, injection_log};
use perfetto_sdk::data_source::TraceContext;
use perfetto_sdk::protos::trace::interned_data::interned_data::InternedData;
use perfetto_sdk::protos::trace::trace_packet::{
    TracePacket, TracePacketDefaults, TracePacketSequenceFlags,
};
use perfetto_sdk::protos::trace::track_event::track_event::EventCategory;
use perfetto_sdk::protos::trace::track_event::track_event::EventName;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEvent as TrackEventProto;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEventDefaults;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEventType;
use perfetto_sdk_protos_gpu::protos::trace::gpu::gpu_track_event::{
    GpuApi, GpuCorrelation, TrackEventExt as GpuTrackEventExt,
};
use std::cell::RefCell;
use std::{ffi::CStr, panic, ptr};

/// Buffer size for activity records (8 MB).
const BUF_SIZE: usize = 8 * 1024 * 1024;
/// Alignment for activity buffers (8 bytes).
const ALIGN_SIZE: usize = 8;

thread_local! {
    /// Per-thread NVTX range name stack, maintained by push/pop callbacks.
    static NVTX_RANGE_STACK: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// Nesting depth for our own callback handler. When > 0, API calls
    /// are from the injection library itself (e.g. cuDeviceGetAttribute
    /// during kernel launch processing) and should not emit track events.
    static CALLBACK_DEPTH: RefCell<u32> = const { RefCell::new(0) };
    /// Nesting depth for runtime API calls. When > 0, driver API callbacks
    /// are nested inside a runtime API call and their track events should be
    /// suppressed to avoid duplicate slices.
    static RUNTIME_API_DEPTH: RefCell<u32> = const { RefCell::new(0) };
    /// Deferred runtime API track event. Populated on RUNTIME_API ENTER,
    /// emitted on RUNTIME_API EXIT with the driver API's correlation ID.
    static DEFERRED_EVENT: RefCell<Option<DeferredTrackEvent>> = const { RefCell::new(None) };
}

struct DeferredTrackEvent {
    ts: u64,
    track_uuid: u64,
    name: String,
    correlation: Option<u64>,
    /// Category iid resolved at ENTER time from the runtime cbid base name.
    /// Held on the deferred record so the EXIT path can emit BEGIN+END with
    /// the same category without recomputing.
    category_iid: u64,
}

// Category iids. Names mirror kineto's ActivityType strings (cuda_runtime,
// cuda_driver). The ".debug" variants carry the events kineto would drop
// (runtime blocklist / non-allowlisted driver) and let consumers opt in to
// the verbose set without changing the producer.
const CUDA_RUNTIME_CATEGORY_IID: u64 = 1;
const CUDA_RUNTIME_DEBUG_CATEGORY_IID: u64 = 2;
const CUDA_DRIVER_CATEGORY_IID: u64 = 3;
const CUDA_DRIVER_DEBUG_CATEGORY_IID: u64 = 4;
const TX_CATEGORY_IID: u64 = 5;

fn is_cuda_category(iid: u64) -> bool {
    matches!(
        iid,
        CUDA_RUNTIME_CATEGORY_IID
            | CUDA_RUNTIME_DEBUG_CATEGORY_IID
            | CUDA_DRIVER_CATEGORY_IID
            | CUDA_DRIVER_DEBUG_CATEGORY_IID
    )
}

/// Maps a runtime API base name to the matching category iid. Mirrors the
/// blocklist in kineto's CuptiCbidRegistry — anything in the blocklist goes
/// to `cuda_runtime.debug`, everything else to `cuda_runtime`.
fn runtime_cbid_category(base_name: &str) -> u64 {
    match base_name {
        "cudaGetDevice"
        | "cudaSetDevice"
        | "cudaGetLastError"
        | "cudaEventCreate"
        | "cudaEventCreateWithFlags"
        | "cudaEventDestroy" => CUDA_RUNTIME_DEBUG_CATEGORY_IID,
        _ => CUDA_RUNTIME_CATEGORY_IID,
    }
}

/// Maps a driver API base name to the matching category iid. Mirrors the
/// allowlist in kineto's CuptiCbidRegistry — only the listed launch and
/// virtual-memory operations go to `cuda_driver`, everything else (memcpy,
/// memset, sync, module load, etc.) goes to `cuda_driver.debug`.
fn driver_cbid_category(base_name: &str) -> u64 {
    match base_name {
        "cuLaunchKernel"
        | "cuLaunchKernelEx"
        | "cuMemCreate"
        | "cuMemMap"
        | "cuMemUnmap"
        | "cuMemRelease"
        | "cuMemExportToShareableHandle"
        | "cuMemImportFromShareableHandle" => CUDA_DRIVER_CATEGORY_IID,
        _ => CUDA_DRIVER_DEBUG_CATEGORY_IID,
    }
}

fn is_gpu_work_cbid(cbid: CUpti_CallbackId) -> bool {
    let name = profiler::get_callback_name(CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API, cbid);
    name.starts_with("cuLaunchKernel")
        || name.starts_with("cuMemcpy")
        || name.starts_with("cuMemset")
}

#[allow(clippy::too_many_arguments)]
fn emit_track_event_with_interning(
    ctx: &mut TraceContext<TrackEventIncrState>,
    state: &mut TrackEventIncrState,
    ts: u64,
    track_uuid: u64,
    category_iid: u64,
    _category_name: &str,
    event_type: TrackEventType,
    name: Option<&str>,
    gpu_correlation: Option<u64>,
) {
    let inst_id = ctx.instance_index();
    let enabled = match category_iid {
        CUDA_RUNTIME_CATEGORY_IID => is_cuda_runtime_enabled_for(inst_id),
        CUDA_RUNTIME_DEBUG_CATEGORY_IID => is_cuda_runtime_debug_enabled_for(inst_id),
        CUDA_DRIVER_CATEGORY_IID => is_cuda_driver_enabled_for(inst_id),
        CUDA_DRIVER_DEBUG_CATEGORY_IID => is_cuda_driver_debug_enabled_for(inst_id),
        TX_CATEGORY_IID => is_tx_events_enabled_for(inst_id),
        _ => false,
    };
    if !enabled {
        return;
    }
    // Sequential per-sequence iids: 1-2 byte varints vs the 9-10 byte
    // hash we used previously. interned_names doubles as the "have we
    // emitted this name?" check — a fresh insert means we still need to
    // emit the EventName entry below.
    let (name_iid, new_name) = match name {
        Some(n) => match state.interned_names.get(n) {
            Some(&iid) => (Some(iid), false),
            None => {
                let iid = state.next_name_iid;
                state.next_name_iid += 1;
                state.interned_names.insert(n.to_string(), iid);
                (Some(iid), true)
            }
        },
        None => (None, false),
    };
    let need_state_clear = std::mem::replace(&mut state.was_cleared, false);

    if need_state_clear || new_name {
        ctx.add_packet(|packet: &mut TracePacket| {
            if need_state_clear {
                packet.set_sequence_flags(
                    TracePacketSequenceFlags::SeqIncrementalStateCleared as u32,
                );
                // Per-sequence defaults — pin this thread's track as the
                // default so subsequent event packets can omit set_track_uuid
                // (saves ~10 bytes per packet for the typical case where the
                // calling thread issues all CUDA work).
                packet.set_trace_packet_defaults(|defaults: &mut TracePacketDefaults| {
                    defaults.set_track_event_defaults(|te_defaults: &mut TrackEventDefaults| {
                        te_defaults.set_track_uuid(track_uuid);
                    });
                });
            }
            packet.set_interned_data(|interned_data: &mut InternedData| {
                // Categories are sequence-stable — only emit on state clear,
                // not on every new-name packet (otherwise every interned_data
                // re-emits the set and burns ~16 B per category per repeat).
                if need_state_clear {
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(CUDA_RUNTIME_CATEGORY_IID);
                        ec.set_name("cuda_runtime");
                    });
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(CUDA_RUNTIME_DEBUG_CATEGORY_IID);
                        ec.set_name("cuda_runtime.debug");
                    });
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(CUDA_DRIVER_CATEGORY_IID);
                        ec.set_name("cuda_driver");
                    });
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(CUDA_DRIVER_DEBUG_CATEGORY_IID);
                        ec.set_name("cuda_driver.debug");
                    });
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(TX_CATEGORY_IID);
                        ec.set_name("tx");
                    });
                }
                if new_name {
                    if let (Some(n), Some(iid)) = (name, name_iid) {
                        interned_data.set_event_names(|en: &mut EventName| {
                            en.set_iid(iid);
                            en.set_name(n);
                        });
                    }
                }
            });
        });
        if need_state_clear {
            state.default_track_uuid = Some(track_uuid);
        }
    }

    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_timestamp(ts);
        packet.set_sequence_flags(TracePacketSequenceFlags::SeqNeedsIncrementalState as u32);
        packet.set_track_event(|te: &mut TrackEventProto| {
            te.set_type(event_type);
            if state.default_track_uuid != Some(track_uuid) {
                te.set_track_uuid(track_uuid);
            }
            te.set_category_iids(category_iid);
            if let Some(iid) = name_iid {
                te.set_name_iid(iid);
            }
            if is_cuda_category(category_iid) {
                GpuTrackEventExt::set_gpu_api(te, GpuApi::GpuApiCuda);
            }
            if let Some(corr_id) = gpu_correlation {
                GpuTrackEventExt::set_gpu_correlation(te, |gc: &mut GpuCorrelation| {
                    gc.set_render_stage_submission_event_ids(corr_id);
                });
            }
        });
    });
}

fn ensure_track_descriptor(
    ctx: &mut TraceContext<TrackEventIncrState>,
    state: &mut TrackEventIncrState,
    track_uuid: u64,
) {
    use perfetto_gpu_compute_injection::tracing::emit_track_descriptor;

    if state.track_descriptor_emitted {
        return;
    }
    state.track_descriptor_emitted = true;

    let pid = std::process::id() as i32;
    let tid = perfetto_gpu_compute_injection::tracing::current_tid() as i32;
    let proc_uuid = process_track_uuid();

    ctx.add_packet(|packet: &mut TracePacket| {
        emit_track_descriptor(packet, proc_uuid, None, Some(pid), None);
    });
    ctx.add_packet(|packet: &mut TracePacket| {
        emit_track_descriptor(packet, track_uuid, Some(proc_uuid), None, Some((pid, tid)));
    });
}

/// Returns a snapshot of the current thread's NVTX range stack.
pub fn nvtx_range_stack() -> Vec<String> {
    NVTX_RANGE_STACK.with(|stack| stack.borrow().clone())
}

/// Callback for CUPTI to request a buffer for storing activity records.
/// # Safety
///
/// This function is intended to be called by CUPTI. Pointers must be valid.
pub unsafe extern "C" fn buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_num_records: *mut usize,
) {
    let _ = panic::catch_unwind(|| {
        // Use posix_memalign for aligned allocation
        let mut aligned_buffer: *mut libc::c_void = ptr::null_mut();
        let result = libc::posix_memalign(&mut aligned_buffer, ALIGN_SIZE, BUF_SIZE);
        if result != 0 || aligned_buffer.is_null() {
            *buffer = ptr::null_mut();
            *size = 0;
            return;
        }
        *buffer = aligned_buffer as *mut u8;
        *size = BUF_SIZE;
        *max_num_records = 0; // Let CUPTI determine the max records
    });
}

/// Callback for CUPTI to notify that a buffer is full or completed.
///
/// Processes the activity records in the buffer, extracting kernel launch details.
/// Always processes kernel activities (needed for renderstages extra data).
/// # Safety
///
/// This function is intended to be called by CUPTI. Pointers must be valid.
pub unsafe extern "C" fn buffer_completed(
    _ctx: CUcontext,
    _stream_id: u32,
    buffer: *mut u8,
    _size: usize,
    valid_size: usize,
) {
    let _ = panic::catch_unwind(|| {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            let mut record: *mut CUpti_Activity = ptr::null_mut();
            while unsafe { profiler::activity_get_next_record(buffer, valid_size, &mut record) }
                .is_ok()
            {
                let r = &*record;
                if r.kind == CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL {
                    let k = &*(record as *const CUpti_ActivityKernel9);
                    if let Some(data) = state.context_data.get_mut(&k.contextId) {
                        let kernel_name = CStr::from_ptr(k.name).to_string_lossy().to_string();
                        injection_log!("kernel activity: {}", kernel_name);
                        data.kernel_activities.push(KernelActivity {
                            kernel_name,
                            grid_size: (k.gridX, k.gridY, k.gridZ),
                            block_size: (k.blockX, k.blockY, k.blockZ),
                            registers_per_thread: k.registersPerThread,
                            dynamic_shared_memory: k.dynamicSharedMemory,
                            static_shared_memory: k.staticSharedMemory,
                            start: k.start,
                            end: k.end,
                            device_id: k.deviceId,
                            context_id: k.contextId,
                            stream_id: k.streamId,
                            channel_id: k.channelID,
                            channel_type: k.channelType,
                            correlation_id: k.correlationId,
                        });
                    }
                } else if r.kind == CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMCPY {
                    let m = &*(record as *const CUpti_ActivityMemcpy6);
                    if let Some(data) = state.context_data.get_mut(&m.contextId) {
                        injection_log!("memcpy activity: {} bytes", m.bytes);
                        data.memcpy_activities.push(MemcpyActivity {
                            copy_kind: m.copyKind,
                            bytes: m.bytes,
                            start: m.start,
                            end: m.end,
                            device_id: m.deviceId,
                            context_id: m.contextId,
                            stream_id: m.streamId,
                            channel_id: m.channelID,
                            channel_type: m.channelType,
                            correlation_id: m.correlationId,
                        });
                    }
                } else if r.kind == CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMSET {
                    let m = &*(record as *const CUpti_ActivityMemset4);
                    if let Some(data) = state.context_data.get_mut(&m.contextId) {
                        injection_log!("memset activity: {} bytes", m.bytes);
                        data.memset_activities.push(MemsetActivity {
                            value: m.value,
                            bytes: m.bytes,
                            memory_kind: m.memoryKind,
                            start: m.start,
                            end: m.end,
                            device_id: m.deviceId,
                            context_id: m.contextId,
                            stream_id: m.streamId,
                            channel_id: m.channelID,
                            channel_type: m.channelType,
                            correlation_id: m.correlationId,
                        });
                    }
                }
            }
        }
        libc::free(buffer as *mut c_void);
    });
}

/// Main CUPTI callback handler.
///
/// Intercepts CUDA driver API calls (specifically `cuLaunchKernel`) to manage profiling sessions,
/// and handles resource events for context creation/destruction.
///
/// When counters data source is disabled, profiler initialization and range profiling are skipped.
/// Kernel launches are always recorded (needed for renderstages timestamps).
/// # Safety
///
/// This function is intended to be called by CUPTI. Pointers must be valid.
pub unsafe extern "C" fn profiler_callback_handler(
    _userdata: *mut c_void,
    domain: CUpti_CallbackDomain,
    cbid: CUpti_CallbackId,
    cbdata: *const c_void,
) {
    let _ = panic::catch_unwind(|| {
        let res = profiler::get_last_error();
        if res != CUptiResult_CUPTI_SUCCESS {
            return;
        }
        // Emit track events for RUNTIME and DRIVER API calls.
        // When a runtime API call (e.g. cudaLaunchKernel) internally calls
        // a driver API (e.g. cuLaunchKernel), we emit only the outermost
        // (runtime) event and transfer the driver API's correlation ID to it.
        let depth = CALLBACK_DEPTH.with(|d| *d.borrow());
        if depth == 0
            && is_cuda_events_enabled()
            && (domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API
                || domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API)
        {
            let cb_data = &*(cbdata as *const CUpti_CallbackData);
            let is_runtime = domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API;
            let is_driver = domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API;
            let runtime_depth = RUNTIME_API_DEPTH.with(|d| *d.borrow());

            if is_runtime {
                let ts = trace_time_ns();
                let track_uuid = thread_track_uuid();

                let full_name = profiler::get_callback_name(
                    CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
                    cbid,
                );
                let base_name = match full_name.rfind("_v") {
                    Some(pos) if full_name[pos + 2..].chars().all(|c| c.is_ascii_digit()) => {
                        &full_name[..pos]
                    }
                    _ => full_name.as_str(),
                };

                if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER {
                    RUNTIME_API_DEPTH.with(|d| *d.borrow_mut() += 1);
                    let category_iid = runtime_cbid_category(base_name);
                    DEFERRED_EVENT.with(|d| {
                        *d.borrow_mut() = Some(DeferredTrackEvent {
                            ts,
                            track_uuid,
                            name: base_name.to_string(),
                            correlation: None,
                            category_iid,
                        });
                    });
                } else if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_EXIT {
                    let current_depth = RUNTIME_API_DEPTH.with(|d| *d.borrow());
                    if current_depth > 0 {
                        RUNTIME_API_DEPTH.with(|d| *d.borrow_mut() -= 1);
                    }
                    let deferred = DEFERRED_EVENT.with(|d| d.borrow_mut().take());
                    if let Some(ev) = deferred {
                        get_track_event_data_source().trace(|ctx| {
                            ctx.with_incremental_state(|ctx, state| {
                                ensure_track_descriptor(ctx, state, ev.track_uuid);
                                emit_track_event_with_interning(
                                    ctx,
                                    state,
                                    ev.ts,
                                    ev.track_uuid,
                                    ev.category_iid,
                                    "",
                                    TrackEventType::TypeSliceBegin,
                                    Some(&ev.name),
                                    ev.correlation,
                                );
                                emit_track_event_with_interning(
                                    ctx,
                                    state,
                                    ts,
                                    ev.track_uuid,
                                    ev.category_iid,
                                    "",
                                    TrackEventType::TypeSliceEnd,
                                    None,
                                    None,
                                );
                            });
                        });
                    }
                }
            } else if is_driver && runtime_depth == 0 {
                // Direct driver API call (not nested inside runtime API).
                let produces_render_stage = is_gpu_work_cbid(cbid);
                let ts = trace_time_ns();
                let track_uuid = thread_track_uuid();

                let full_name = profiler::get_callback_name(
                    CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API,
                    cbid,
                );
                let base_name = match full_name.rfind("_v") {
                    Some(pos) if full_name[pos + 2..].chars().all(|c| c.is_ascii_digit()) => {
                        &full_name[..pos]
                    }
                    _ => full_name.as_str(),
                };
                let correlation = if produces_render_stage {
                    Some(cb_data.correlationId as u64)
                } else {
                    None
                };

                let category_iid = driver_cbid_category(base_name);
                if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER {
                    get_track_event_data_source().trace(|ctx| {
                        ctx.with_incremental_state(|ctx, state| {
                            ensure_track_descriptor(ctx, state, track_uuid);
                            emit_track_event_with_interning(
                                ctx,
                                state,
                                ts,
                                track_uuid,
                                category_iid,
                                "",
                                TrackEventType::TypeSliceBegin,
                                Some(base_name),
                                correlation,
                            );
                        });
                    });
                } else if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_EXIT {
                    get_track_event_data_source().trace(|ctx| {
                        ctx.with_incremental_state(|ctx, state| {
                            ensure_track_descriptor(ctx, state, track_uuid);
                            emit_track_event_with_interning(
                                ctx,
                                state,
                                ts,
                                track_uuid,
                                category_iid,
                                "",
                                TrackEventType::TypeSliceEnd,
                                None,
                                None,
                            );
                        });
                    });
                }
            } else if is_driver && runtime_depth > 0 {
                // Nested driver API call — capture correlation for the
                // deferred runtime API event, but only for calls that
                // produce GPU render stage events (kernel launches, memcpy,
                // memset). Other driver API calls (e.g. cuCtxSynchronize)
                // don't have matching render stage events.
                let produces_render_stage = is_gpu_work_cbid(cbid);
                if produces_render_stage
                    && cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER
                {
                    DEFERRED_EVENT.with(|d| {
                        if let Some(ev) = d.borrow_mut().as_mut() {
                            ev.correlation = Some(cb_data.correlationId as u64);
                        }
                    });
                }
            }
        }
        // Suppress internal CUPTI calls from the injection library.
        CALLBACK_DEPTH.with(|d| *d.borrow_mut() += 1);
        let instrumented = is_instrumented_enabled();
        if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API
            && cbid == CUpti_driver_api_trace_cbid_enum_CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
        {
            let cb_data = &*(cbdata as *const CUpti_CallbackData);
            let ctx = cb_data.context;
            let params = &*(cb_data.functionParams as *const cuLaunchKernel_params);
            if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER {
                // Compute bitmask of instance IDs that want counters for this kernel.
                // Profile if any instance wants it (non-zero bitmask).
                let profiled_instances: u8 = if instrumented {
                    let mangled = if !cb_data.symbolName.is_null() {
                        CStr::from_ptr(cb_data.symbolName)
                            .to_string_lossy()
                            .to_string()
                    } else {
                        String::new()
                    };
                    let configs: Vec<_> =
                        (0..8u32).map(|id| (id, get_counter_config(id))).collect();
                    // Phase 1: kernel name filtering.
                    let mut mask: u8 = 0;
                    let mut need_demangled = false;
                    for &(id, ref cfg) in &configs {
                        if let Some(cfg) = cfg {
                            let isc = &cfg.instrumented_sampling_config;
                            if isc.activity_name_filters.is_empty() {
                                mask |= 1 << id;
                            } else {
                                need_demangled = true;
                            }
                        }
                    }
                    if need_demangled {
                        let demangled =
                            perfetto_gpu_compute_injection::kernel::demangle_name(&mangled);
                        let function_name =
                            perfetto_gpu_compute_injection::kernel::simplify_name(&demangled);
                        for &(id, ref cfg) in &configs {
                            if mask & (1 << id) != 0 {
                                continue;
                            }
                            if let Some(cfg) = cfg {
                                if cfg.instrumented_sampling_config.should_profile_kernel(
                                    &mangled,
                                    &demangled,
                                    function_name,
                                ) {
                                    mask |= 1 << id;
                                }
                            }
                        }
                    }
                    // Phase 2: NVTX range filtering. Clear bits for instances
                    // whose NVTX include/exclude globs reject the current context.
                    if mask != 0 {
                        let mut need_nvtx = false;
                        for &(id, ref cfg) in &configs {
                            if mask & (1 << id) != 0 {
                                if let Some(cfg) = cfg {
                                    let isc = &cfg.instrumented_sampling_config;
                                    if !isc.activity_tx_include_globs.is_empty()
                                        || !isc.activity_tx_exclude_globs.is_empty()
                                    {
                                        need_nvtx = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if need_nvtx {
                            let nvtx_stack = nvtx_range_stack();
                            for &(id, ref cfg) in &configs {
                                if mask & (1 << id) == 0 {
                                    continue;
                                }
                                if let Some(cfg) = cfg {
                                    if !cfg
                                        .instrumented_sampling_config
                                        .should_profile_in_nvtx_context(&nvtx_stack)
                                    {
                                        mask &= !(1 << id);
                                    }
                                }
                            }
                        }
                    }
                    mask
                } else {
                    0
                };
                // Phase 3: skip/count filtering (requires GLOBAL_STATE lock
                // for per-instance dispatch counters).
                let mut profiled_instances = profiled_instances;
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    if profiled_instances != 0 {
                        for id in 0..8u32 {
                            if profiled_instances & (1 << id) == 0 {
                                continue;
                            }
                            if let Some(cfg) = get_counter_config(id) {
                                let isc = &cfg.instrumented_sampling_config;
                                if !isc.activity_ranges.is_empty() {
                                    let count = state.dispatch_counters[id as usize];
                                    state.dispatch_counters[id as usize] += 1;
                                    if !isc.should_profile_at_count(count) {
                                        profiled_instances &= !(1 << id);
                                    }
                                }
                            }
                        }
                    }
                    let should_profile = profiled_instances != 0;
                    let active_ctx = state.active_ctx;
                    // Only manage range profiler sessions when instrumented sampling is enabled
                    if should_profile {
                        match active_ctx {
                            Some(active_ctx) if active_ctx != ctx => {
                                let active_ctx_id = unsafe { profiler::get_context_id(active_ctx) };
                                if let Some(old_data) = state.context_data.get_mut(&active_ctx_id) {
                                    old_data.finalize_profiler(true);
                                }
                                state.active_ctx = None;
                            }
                            _ => {}
                        }
                    }
                    let ctx_id = unsafe { profiler::get_context_id(ctx) };
                    let metric_names = state.config.metrics.clone();
                    if state.context_data.contains_key(&ctx_id) {
                        state.active_ctx = Some(ctx);
                        if let Some(data) = state.context_data.get_mut(&ctx_id) {
                            // Initialize range profiler if not yet done
                            if should_profile && data.range_profiler.is_none() {
                                let mut rp = RangeProfiler::new(ctx);
                                if rp.enable().is_ok()
                                    && rp
                                        .set_config(
                                            &metric_names,
                                            &mut data.counter_data_image,
                                            data.max_num_ranges,
                                            CUpti_ProfilerReplayMode_CUPTI_KernelReplay,
                                        )
                                        .is_ok()
                                {
                                    data.range_profiler = Some(rp);
                                    data.is_active = true;
                                }
                            }
                            // Start profiling pass and push range for this kernel launch
                            if should_profile {
                                if let Some(rp) = &data.range_profiler {
                                    let _ = rp.start();
                                    // Push a range so the profiler records counter data.
                                    // Even in AutoRange mode, an explicit range scope is required.
                                    let range_name = if !cb_data.symbolName.is_null() {
                                        CStr::from_ptr(cb_data.symbolName)
                                            .to_string_lossy()
                                            .to_string()
                                    } else {
                                        format!("kernel_{}", data.kernel_launches.len())
                                    };
                                    let c_range_name =
                                        std::ffi::CString::new(range_name.as_str()).unwrap();
                                    let _ = rp.push_range(&c_range_name);
                                }
                            }
                            // Record kernel launch with start timestamp.
                            let start = trace_time_ns();
                            let block_size = params.blockDimX * params.blockDimY * params.blockDimZ;
                            let cache_mode = unsafe {
                                profiler::get_func_attribute(
                                    params.f,
                                    CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                                )
                            }
                            .unwrap_or(0);
                            let max_active_blocks_per_sm = unsafe {
                                profiler::occupancy_max_active_blocks_per_multiprocessor(
                                    params.f,
                                    block_size as i32,
                                    params.sharedMemBytes as usize,
                                )
                            }
                            .unwrap_or(0);
                            data.kernel_launches.push(KernelLaunch {
                                function: params.f,
                                start,
                                end: 0,
                                profiled_instances,
                                cache_mode,
                                max_active_blocks_per_sm,
                                correlation_id: cb_data.correlationId,
                            });
                        }
                    }
                }
            } else if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_EXIT {
                // Handle kernel launch completion - decode profiler data and set end timestamp.
                // Only pop/stop/decode if this kernel was actually profiled (non-zero mask).
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    let ctx_id = unsafe { profiler::get_context_id(ctx) };
                    if let Some(data) = state.context_data.get_mut(&ctx_id) {
                        let was_profiled = data
                            .kernel_launches
                            .last()
                            .map(|l| l.profiled_instances != 0)
                            .unwrap_or(false);
                        if was_profiled {
                            if let Some(rp) = &mut data.range_profiler {
                                // Pop the range and stop profiling
                                let _ = rp.pop_range();
                                let _ = rp.stop();
                                let _ = rp.decode_counter_data();
                                let metric_names = rp.validated_metric_names().to_vec();
                                if let Some(me) = &data.metric_evaluator {
                                    if let Ok(infos) = me.evaluate_all_ranges(
                                        &data.counter_data_image,
                                        &metric_names,
                                    ) {
                                        data.range_info.extend(infos);
                                    }
                                }
                                let _ =
                                    rp.initialize_counter_data_image(&mut data.counter_data_image);
                            }
                        }
                        // Set end timestamp on the current kernel launch
                        if let Some(last_launch) = data.kernel_launches.last_mut() {
                            if last_launch.end == 0 {
                                last_launch.end = trace_time_ns();
                            }
                        }
                    }
                }
            }
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API
            && cbid == CUpti_driver_api_trace_cbid_enum_CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx
        {
            // cuLaunchKernelEx handled by generic track event emission above.
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RESOURCE {
            if cbid == CUpti_CallbackIdResource_CUPTI_CBID_RESOURCE_CONTEXT_CREATED {
                let res_data = &*(cbdata as *const CUpti_ResourceData);
                let ctx = res_data.context;
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    // Only manage active context's range profiler when instrumented sampling is enabled
                    if instrumented {
                        if let Some(active_ctx) = state.active_ctx {
                            let active_ctx_id = unsafe { profiler::get_context_id(active_ctx) };
                            if let Some(data) = state.context_data.get_mut(&active_ctx_id) {
                                data.finalize_profiler(true);
                            }
                            state.active_ctx = None;
                        }
                    }
                    let device_id = unsafe { profiler::get_device(ctx) }.unwrap_or(0);
                    // Eagerly cache the nvidia-smi index for this device so
                    // later lookups (during flush) don't make cu* FFI calls
                    // that would generate spurious DRIVER activity records.
                    let _ = profiler::get_nvidia_smi_index(device_id);
                    let num_sms = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    )
                    .unwrap_or(0);
                    let warp_size = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                    )
                    .unwrap_or(32);
                    let max_threads_per_sm = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                    )
                    .unwrap_or(0);
                    let max_blocks_per_sm = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                    )
                    .unwrap_or(0);
                    let max_regs_per_sm = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                    )
                    .unwrap_or(0);
                    let max_smem_per_sm = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                    )
                    .unwrap_or(0);
                    let cc_major = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    )
                    .unwrap_or(0);
                    let cc_minor = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    )
                    .unwrap_or(0);
                    let mut data = Box::new(crate::state::CtxProfilerData {
                        device_id,
                        num_sms,
                        warp_size,
                        max_threads_per_sm,
                        max_blocks_per_sm,
                        max_regs_per_sm,
                        max_smem_per_sm,
                        compute_capability: (cc_major, cc_minor),
                        max_num_ranges: 10,
                        is_active: false,
                        counter_data_image: Vec::new(),
                        metric_evaluator: None,
                        range_profiler: None,
                        range_info: Vec::new(),
                        kernel_launches: Vec::new(),
                        kernel_activities: Vec::new(),
                        memcpy_activities: Vec::new(),
                        memset_activities: Vec::new(),
                    });
                    // Only initialize profiler and metric evaluator when instrumented sampling is enabled
                    // Otherwise, just track context data for renderstages
                    if instrumented {
                        if Profiler::initialize().is_ok() {
                            if let Ok(me) = unsafe { MetricEvaluator::new(ctx) } {
                                data.metric_evaluator = Some(me);
                            }
                            let metric_names = state.config.metrics.clone();
                            let mut rp = RangeProfiler::new(ctx);
                            if rp.enable().is_ok()
                                && rp
                                    .set_config(
                                        &metric_names,
                                        &mut data.counter_data_image,
                                        data.max_num_ranges,
                                        CUpti_ProfilerReplayMode_CUPTI_KernelReplay,
                                    )
                                    .is_ok()
                            {
                                data.range_profiler = Some(rp);
                                data.is_active = true;
                                state.active_ctx = Some(ctx);
                            }
                        } else {
                            injection_log!("Failed to initialize profiler");
                        }
                    }
                    let ctx_id = unsafe { profiler::get_context_id(ctx) };
                    state.context_data.insert(ctx_id, data);
                }
            } else if cbid == CUpti_CallbackIdResource_CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING
            {
                let res_data = &*(cbdata as *const CUpti_ResourceData);
                let ctx = res_data.context;
                let ctx_id = unsafe { profiler::get_context_id(ctx) };
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    if let Some(data) = state.context_data.get_mut(&ctx_id) {
                        data.finalize_profiler(true);
                    }
                }
            }
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_STATE
            && cbid == CUpti_CallbackIdState_CUPTI_CBID_STATE_FATAL_ERROR
        {
            let state_data = &*(cbdata as *const CUpti_StateData);
            let err_str =
                profiler::get_result_string(state_data.__bindgen_anon_1.notification.result);
            let msg = CStr::from_ptr(state_data.__bindgen_anon_1.notification.message);
            injection_fatal!("CUPTI Fatal Error: {}: {}", err_str, msg.to_string_lossy());
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API
            && cbid
                == CUpti_runtime_api_trace_cbid_enum_CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020
        {
            let cb_data = &*(cbdata as *const CUpti_CallbackData);
            if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER {
                injection_log!("cudaDeviceReset detected, flushing activity buffers");
                let _ = profiler::activity_flush_all(0);
            }
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_NVTX {
            // Track NVTX range push/pop for activity_tx_include/exclude_globs filtering
            // and emit track events when the tx category is enabled.
            if cbid == CUpti_nvtx_api_trace_cbid_CUPTI_CBID_NVTX_nvtxRangePushA {
                let nvtx_data = &*(cbdata as *const CUpti_NvtxData);
                let msg_ptr = *(nvtx_data.functionParams as *const *const std::os::raw::c_char);
                if !msg_ptr.is_null() {
                    let name = CStr::from_ptr(msg_ptr).to_string_lossy().to_string();
                    if is_tx_events_enabled() {
                        let ts = trace_time_ns();
                        let track_uuid = thread_track_uuid();
                        get_track_event_data_source().trace(|ctx| {
                            ctx.with_incremental_state(|ctx, state| {
                                ensure_track_descriptor(ctx, state, track_uuid);
                                emit_track_event_with_interning(
                                    ctx,
                                    state,
                                    ts,
                                    track_uuid,
                                    TX_CATEGORY_IID,
                                    "tx",
                                    TrackEventType::TypeSliceBegin,
                                    Some(&name),
                                    None,
                                );
                            });
                        });
                    }
                    NVTX_RANGE_STACK.with(|stack| stack.borrow_mut().push(name));
                }
            } else if cbid == CUpti_nvtx_api_trace_cbid_CUPTI_CBID_NVTX_nvtxRangePop {
                NVTX_RANGE_STACK.with(|stack| {
                    stack.borrow_mut().pop();
                });
                if is_tx_events_enabled() {
                    let ts = trace_time_ns();
                    let track_uuid = thread_track_uuid();
                    get_track_event_data_source().trace(|ctx| {
                        ctx.with_incremental_state(|ctx, state| {
                            ensure_track_descriptor(ctx, state, track_uuid);
                            emit_track_event_with_interning(
                                ctx,
                                state,
                                ts,
                                track_uuid,
                                TX_CATEGORY_IID,
                                "tx",
                                TrackEventType::TypeSliceEnd,
                                None,
                                None,
                            );
                        });
                    });
                }
            }
        }
        CALLBACK_DEPTH.with(|d| *d.borrow_mut() -= 1);
    });
}
