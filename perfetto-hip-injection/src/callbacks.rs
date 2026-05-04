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

//! rocprofiler-sdk callback handlers for AMD GPU tracing.

use crate::rocprofiler_sys::*;
use crate::state::{
    AgentInfo, CounterResult, KernelDispatch, MemcopyActivity, MemsetActivity, GLOBAL_STATE,
};
use perfetto_gpu_compute_injection::injection_log;
use perfetto_gpu_compute_injection::tracing::{
    get_counter_config, get_track_event_data_source, is_hip_events_enabled,
    is_hip_runtime_debug_enabled_for, is_hip_runtime_enabled_for, is_tx_events_enabled_for,
    process_track_uuid, TrackEventIncrState,
};
// TODO(perfetto-sdk-1.0): drop the trace_packet_defaults shim imports below
// and switch to perfetto_sdk::protos::trace::trace_packet::TracePacketDefaults
// once perfetto-sdk lands the field upstream. See
// `perfetto_gpu_compute_injection::trace_packet_defaults` for context.
use perfetto_gpu_compute_injection::trace_packet_defaults::prelude::*;
use perfetto_gpu_compute_injection::trace_packet_defaults::TracePacketDefaults;
use perfetto_sdk::data_source::TraceContext;
use perfetto_sdk::protos::trace::interned_data::interned_data::InternedData;
use perfetto_sdk::protos::trace::trace_packet::{TracePacket, TracePacketSequenceFlags};
use perfetto_sdk::protos::trace::track_event::track_event::EventCategory;
use perfetto_sdk::protos::trace::track_event::track_event::EventName;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEvent as TrackEventProto;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEventDefaults;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEventType;
use perfetto_sdk_protos_gpu::protos::trace::gpu::gpu_track_event::{
    GpuApi, GpuCorrelation, TrackEventExt as GpuTrackEventExt,
};
use std::ffi::CStr;
use std::panic;

// Category iids. Names mirror kineto's HIP filtering: hip_runtime is the
// kineto-equivalent default-on set (everything *except* the kineto blocklist
// of high-frequency, low-information calls); hip_runtime.debug carries the
// blocklisted calls for consumers that opt in via enabled_tags=["debug"] or
// the explicit category name. HIP unifies its runtime+driver API surface so
// — unlike CUDA — there is no separate hip_driver category.
const HIP_RUNTIME_CATEGORY_IID: u64 = 1;
const HIP_RUNTIME_DEBUG_CATEGORY_IID: u64 = 2;
const TX_CATEGORY_IID: u64 = 3;

fn is_hip_category(iid: u64) -> bool {
    matches!(
        iid,
        HIP_RUNTIME_CATEGORY_IID | HIP_RUNTIME_DEBUG_CATEGORY_IID
    )
}

/// Maps a HIP runtime API name to the matching category iid. Mirrors the
/// blocklist kineto installs on the rocprofiler service via
/// `RocprofApiIdList::setInvertMode(true)` (see kineto's
/// fbcode/kineto/libkineto/src/RocprofLogger.cpp). Names in the blocklist go
/// to `hip_runtime.debug`, everything else to `hip_runtime`.
fn hip_runtime_cbid_category(name: &str) -> u64 {
    match name {
        "hipGetDevice"
        | "hipSetDevice"
        | "hipGetLastError"
        | "__hipPushCallConfiguration"
        | "__hipPopCallConfiguration"
        | "hipCtxSetCurrent"
        | "hipEventRecord"
        | "hipEventQuery"
        | "hipGetDeviceProperties"
        | "hipPeekAtLastError"
        | "hipModuleGetFunction"
        | "hipEventCreateWithFlags" => HIP_RUNTIME_DEBUG_CATEGORY_IID,
        _ => HIP_RUNTIME_CATEGORY_IID,
    }
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
        HIP_RUNTIME_CATEGORY_IID => is_hip_runtime_enabled_for(inst_id),
        HIP_RUNTIME_DEBUG_CATEGORY_IID => is_hip_runtime_debug_enabled_for(inst_id),
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
                // calling thread issues all HIP work).
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
                        ec.set_iid(HIP_RUNTIME_CATEGORY_IID);
                        ec.set_name("hip_runtime");
                    });
                    interned_data.set_event_categories(|ec: &mut EventCategory| {
                        ec.set_iid(HIP_RUNTIME_DEBUG_CATEGORY_IID);
                        ec.set_name("hip_runtime.debug");
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
            if is_hip_category(category_iid) {
                GpuTrackEventExt::set_gpu_api(te, GpuApi::GpuApiHip);
            }
            if let Some(corr_id) = gpu_correlation {
                GpuTrackEventExt::set_gpu_correlation(te, |gc: &mut GpuCorrelation| {
                    gc.set_render_stage_submission_event_ids(corr_id);
                });
            }
        });
    });
}

/// Collected API event data for deferred emission outside of GLOBAL_STATE lock.
struct CollectedApiEvent {
    start_ns: u64,
    end_ns: u64,
    track_uuid: u64,
    tid: u64,
    name: String,
    correlation: Option<u64>,
}

fn ensure_track_descriptor(
    ctx: &mut TraceContext<TrackEventIncrState>,
    state: &mut TrackEventIncrState,
    track_uuid: u64,
    tid: u64,
) {
    use perfetto_gpu_compute_injection::tracing::emit_track_descriptor;

    if state.track_descriptor_emitted {
        return;
    }
    state.track_descriptor_emitted = true;

    let pid = std::process::id() as i32;
    let proc_uuid = process_track_uuid();

    ctx.add_packet(|packet: &mut TracePacket| {
        emit_track_descriptor(packet, proc_uuid, None, Some(pid), None);
    });
    ctx.add_packet(|packet: &mut TracePacket| {
        emit_track_descriptor(
            packet,
            track_uuid,
            Some(proc_uuid),
            None,
            Some((pid, tid as i32)),
        );
    });
}

/// Buffer callback: called by rocprofiler's internal thread when the buffer is
/// flushed (at watermark or explicitly). Processes kernel dispatch and memory
/// copy records.
///
/// # Safety
///
/// Called by the rocprofiler runtime; the caller guarantees that `headers` is
/// either null or a valid array of `num_headers` pointers, each pointing to a
/// valid `rocprofiler_record_header_t` whose `payload` is valid for the
/// corresponding record type.
pub unsafe extern "C" fn buffer_callback(
    _context: rocprofiler_context_id_t,
    _buffer_id: rocprofiler_buffer_id_t,
    headers: *mut *mut rocprofiler_record_header_t,
    num_headers: usize,
    _user_data: *mut std::os::raw::c_void,
    _drop_count: u64,
) {
    let _ = panic::catch_unwind(|| {
        if headers.is_null() || num_headers == 0 {
            return;
        }
        let header_slice = std::slice::from_raw_parts(headers, num_headers);
        let mut state = match GLOBAL_STATE.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut api_events: Vec<CollectedApiEvent> = Vec::new();

        for &header_ptr in header_slice {
            if header_ptr.is_null() {
                continue;
            }
            let hdr = &*header_ptr;
            if hdr.category() != ROCPROFILER_BUFFER_CATEGORY_TRACING {
                continue;
            }

            let kind = hdr.kind();

            if kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH {
                if hdr.payload.is_null() {
                    continue;
                }
                let rec =
                    &*(hdr.payload as *const rocprofiler_buffer_tracing_kernel_dispatch_record_t);
                let info = &rec.dispatch_info;

                // Look up kernel name from code object callbacks.
                let kernel_name = state
                    .kernel_names
                    .get(&info.kernel_id)
                    .cloned()
                    .unwrap_or_else(|| format!("kernel_{}", info.kernel_id));

                // Translate internal ROCm runtime kernels to memcpy/memset events.
                if kernel_name.starts_with("__amd_rocclr_copyBuffer") {
                    let device_index = state
                        .agents
                        .get(&info.agent_id.handle)
                        .map(|a| a.device_index)
                        .unwrap_or(0);
                    state.memcopies.push(MemcopyActivity {
                        bytes: 0,
                        start_ns: rec.start_timestamp,
                        end_ns: rec.end_timestamp,
                        device_index,
                        direction: -1,
                        correlation_id: rec.correlation_id.internal,
                    });
                    continue;
                }
                if kernel_name.starts_with("__amd_rocclr_fillBuffer") {
                    let device_index = state
                        .agents
                        .get(&info.agent_id.handle)
                        .map(|a| a.device_index)
                        .unwrap_or(0);
                    state.memsets.push(MemsetActivity {
                        start_ns: rec.start_timestamp,
                        end_ns: rec.end_timestamp,
                        device_index,
                        correlation_id: rec.correlation_id.internal,
                    });
                    continue;
                }

                // Look up agent info.
                let agent_info = state.agents.get(&info.agent_id.handle);
                let device_index = agent_info.map(|a| a.device_index).unwrap_or(0);
                let arch = agent_info.map(|a| a.arch.clone()).unwrap_or_default();
                let wave_front_size = agent_info.map(|a| a.wave_front_size).unwrap_or(0);
                let cu_count = agent_info.map(|a| a.cu_count).unwrap_or(0);
                let max_engine_clk_fcompute =
                    agent_info.map(|a| a.max_engine_clk_fcompute).unwrap_or(0);

                state.kernel_dispatches.push(KernelDispatch {
                    kernel_name,
                    grid: (info.grid_size.x, info.grid_size.y, info.grid_size.z),
                    workgroup: (
                        info.workgroup_size.x,
                        info.workgroup_size.y,
                        info.workgroup_size.z,
                    ),
                    start_ns: rec.start_timestamp,
                    end_ns: rec.end_timestamp,
                    queue_handle: info.queue_id.handle,
                    device_index,
                    arch,
                    wave_front_size,
                    cu_count,
                    max_engine_clk_fcompute,
                    correlation_id: rec.correlation_id.internal,
                });
            } else if kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY {
                if hdr.payload.is_null() {
                    continue;
                }
                let rec = &*(hdr.payload as *const rocprofiler_buffer_tracing_memory_copy_record_t);

                // Attribute the copy to the destination agent (GPU receiving data).
                let device_index = state
                    .agents
                    .get(&rec.dst_agent_id.handle)
                    .map(|a| a.device_index)
                    .unwrap_or_else(|| {
                        state
                            .agents
                            .get(&rec.src_agent_id.handle)
                            .map(|a| a.device_index)
                            .unwrap_or(0)
                    });

                state.memcopies.push(MemcopyActivity {
                    bytes: rec.bytes,
                    start_ns: rec.start_timestamp,
                    end_ns: rec.end_timestamp,
                    device_index,
                    #[allow(clippy::unnecessary_cast)]
                    direction: rec.operation as i32,
                    correlation_id: rec.correlation_id.internal,
                });
            } else if kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API {
                if hdr.payload.is_null() {
                    continue;
                }
                let rec = &*(hdr.payload as *const rocprofiler_buffer_tracing_hip_api_record_t);

                // Skip records whose size doesn't match our struct (e.g.
                // the extended `hip_api_ext_record_t` variant).
                let expected_size =
                    std::mem::size_of::<rocprofiler_buffer_tracing_hip_api_record_t>() as u64;
                if rec.size != expected_size {
                    continue;
                }

                if !is_hip_events_enabled() {
                    continue;
                }

                let tid = rec.thread_id;
                perfetto_gpu_compute_injection::config::capture_thread_name(
                    &mut state.thread_names,
                    tid,
                );

                // Resolve operation name via rocprofiler.
                let op_name = {
                    let mut name_ptr: *const std::os::raw::c_char = std::ptr::null();
                    let mut name_len: usize = 0;
                    let status = rocprofiler_query_buffer_tracing_kind_operation_name(
                        rec.kind,
                        rec.operation,
                        &mut name_ptr,
                        &mut name_len,
                    );
                    if status == ROCPROFILER_STATUS_SUCCESS && !name_ptr.is_null() {
                        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                    } else {
                        format!("hip_op_{}", rec.operation)
                    }
                };

                // Determine if this API call produces GPU work (for GpuCorrelation).
                let produces_gpu_work = op_name.starts_with("hipLaunchKernel")
                    || op_name.starts_with("hipMemcpy")
                    || op_name.starts_with("hipMemset");
                let correlation = if produces_gpu_work {
                    Some(rec.correlation_id.internal)
                } else {
                    None
                };

                // Compute thread track UUID from tid.
                let track_uuid = {
                    let mut h: u64 = 0xcbf29ce484222325;
                    let pid_bytes = (std::process::id() as u64).to_le_bytes();
                    let tid_bytes = tid.to_le_bytes();
                    for b in pid_bytes.iter().chain(tid_bytes.iter()) {
                        h ^= *b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    h
                };

                // Collect API event for deferred emission after lock release.
                api_events.push(CollectedApiEvent {
                    start_ns: rec.start_timestamp,
                    end_ns: rec.end_timestamp,
                    track_uuid,
                    tid,
                    name: op_name,
                    correlation,
                });
            }
        }
        // Drop the lock before emitting track events.
        drop(state);

        // Emit collected HIP API track events without holding GLOBAL_STATE.
        if !api_events.is_empty() {
            get_track_event_data_source().trace(|ctx| {
                ctx.with_incremental_state(|ctx, state| {
                    for event in &api_events {
                        ensure_track_descriptor(ctx, state, event.track_uuid, event.tid);
                        let category_iid = hip_runtime_cbid_category(&event.name);
                        emit_track_event_with_interning(
                            ctx,
                            state,
                            event.start_ns,
                            event.track_uuid,
                            category_iid,
                            "",
                            TrackEventType::TypeSliceBegin,
                            Some(&event.name),
                            event.correlation,
                        );
                        emit_track_event_with_interning(
                            ctx,
                            state,
                            event.end_ns,
                            event.track_uuid,
                            category_iid,
                            "",
                            TrackEventType::TypeSliceEnd,
                            None,
                            None,
                        );
                    }
                });
            });
            injection_log!("flushed {} API track events", api_events.len());
        }
    });
}

/// Code object callback: called synchronously on the application thread when
/// the ROCm runtime loads/unloads code objects and registers kernel symbols.
/// We capture `kernel_id → kernel_name` mappings here for use in the buffer callback.
///
/// # Safety
///
/// Called by the rocprofiler runtime; `record.payload` is valid for
/// `rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t`
/// when `record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`.
pub unsafe extern "C" fn code_object_callback(
    record: rocprofiler_callback_tracing_record_t,
    _user_data: *mut rocprofiler_user_data_t,
    _callback_data: *mut std::os::raw::c_void,
) {
    let _ = panic::catch_unwind(|| {
        if record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT {
            return;
        }
        if record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER {
            return;
        }
        if record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD {
            return;
        }
        if record.payload.is_null() {
            return;
        }
        let data = &*(record.payload
            as *const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t);
        if data.kernel_name.is_null() {
            return;
        }
        let name = CStr::from_ptr(data.kernel_name)
            .to_string_lossy()
            .into_owned();
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            state.kernel_names.insert(data.kernel_id, name);
        }
    });
}

/// Agent enumeration callback: called once per available agent during
/// `rocprofiler_query_available_agents`. Populates `state.agents` with
/// `agent_id.handle → logical_node_type_id` for GPU agents.
///
/// # Safety
///
/// Called by the rocprofiler runtime; `agents` is either null or a valid array
/// of `num_agents` pointers each pointing to a valid `rocprofiler_agent_v0_t`.
pub unsafe extern "C" fn agents_callback(
    version: rocprofiler_agent_version_t,
    agents: *mut *const std::os::raw::c_void,
    num_agents: usize,
    user_data: *mut std::os::raw::c_void,
) -> rocprofiler_status_t {
    let _ = panic::catch_unwind(|| {
        let _ = (version, user_data);
        if agents.is_null() || num_agents == 0 {
            return;
        }
        let agent_ptrs = std::slice::from_raw_parts(agents, num_agents);
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            for &agent_ptr in agent_ptrs {
                if agent_ptr.is_null() {
                    continue;
                }
                let agent = &*(agent_ptr as *const rocprofiler_agent_v0_t);
                if agent.agent_type() == ROCPROFILER_AGENT_TYPE_GPU {
                    let arch = if agent.name.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(agent.name).to_string_lossy().into_owned()
                    };
                    state.agents.insert(
                        agent.id.handle,
                        AgentInfo {
                            device_index: agent.logical_node_type_id,
                            arch,
                            wave_front_size: agent.wave_front_size,
                            cu_count: agent.cu_count,
                            max_engine_clk_fcompute: agent.max_engine_clk_fcompute,
                        },
                    );
                }
            }
        }
    });
    ROCPROFILER_STATUS_SUCCESS
}

/// Dispatch counting callback: called before each kernel dispatch to select
/// the counter config for this agent.
///
/// # Safety
///
/// Called by the rocprofiler runtime; `config` points to a valid
/// `rocprofiler_counter_config_id_t` that should be set to the desired config.
pub unsafe extern "C" fn dispatch_counting_callback(
    dispatch_data: rocprofiler_dispatch_counting_service_data_t,
    config: *mut rocprofiler_counter_config_id_t,
    _user_data: *mut rocprofiler_user_data_t,
    _callback_data: *mut std::os::raw::c_void,
) {
    let _ = panic::catch_unwind(|| {
        let info = &dispatch_data.dispatch_info;
        let agent_handle = info.agent_id.handle;
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            // Look up kernel name for filtering.
            let kernel_name = state
                .kernel_names
                .get(&info.kernel_id)
                .cloned()
                .unwrap_or_default();

            // Skip internal ROCm runtime kernels (same check as buffer_callback).
            if kernel_name.starts_with("__amd_rocclr_") {
                return;
            }

            // Compute profiled_instances bitmask.
            let mut mask: u8 = 0;
            // Phase 1: kernel name filtering.
            let mut need_demangled = false;
            for id in 0..8u32 {
                if let Some(cfg) = get_counter_config(id) {
                    let isc = &cfg.instrumented_sampling_config;
                    if isc.activity_name_filters.is_empty() {
                        mask |= 1 << id;
                    } else {
                        need_demangled = true;
                    }
                }
            }
            if need_demangled {
                let demangled = perfetto_gpu_compute_injection::kernel::demangle_name(&kernel_name);
                let function_name =
                    perfetto_gpu_compute_injection::kernel::simplify_name(&demangled);
                for id in 0..8u32 {
                    if mask & (1 << id) != 0 {
                        continue;
                    }
                    if let Some(cfg) = get_counter_config(id) {
                        if cfg.instrumented_sampling_config.should_profile_kernel(
                            &kernel_name,
                            &demangled,
                            function_name,
                        ) {
                            mask |= 1 << id;
                        }
                    }
                }
            }
            // Phase 2: NVTX filtering — skipped for HIP (no roctx callbacks).
            // Phase 3: skip/count filtering.
            if mask != 0 {
                for id in 0..8u32 {
                    if mask & (1 << id) == 0 {
                        continue;
                    }
                    if let Some(cfg) = get_counter_config(id) {
                        let isc = &cfg.instrumented_sampling_config;
                        if !isc.activity_ranges.is_empty() {
                            let count = state.dispatch_counters[id as usize];
                            state.dispatch_counters[id as usize] += 1;
                            if !isc.should_profile_at_count(count) {
                                mask &= !(1 << id);
                            }
                        }
                    }
                }
            }

            if mask != 0 {
                if let Some(&config_handle) = state.counter_configs.get(&agent_handle) {
                    *config = rocprofiler_counter_config_id_t {
                        handle: config_handle,
                    };
                }
                // Store bitmask for record_counting_callback to pick up.
                state
                    .dispatch_profiled_instances
                    .insert(dispatch_data.correlation_id.internal, mask);
            }
        }
    });
}

/// Record counting callback: called after a kernel completes with counter values.
///
/// # Safety
///
/// Called by the rocprofiler runtime; `record_data` is a valid array of
/// `record_count` `rocprofiler_counter_record_t` entries.
pub unsafe extern "C" fn record_counting_callback(
    dispatch_data: rocprofiler_dispatch_counting_service_data_t,
    record_data: *mut rocprofiler_counter_record_t,
    record_count: usize,
    _user_data: rocprofiler_user_data_t,
    _callback_data: *mut std::os::raw::c_void,
) {
    let _ = panic::catch_unwind(|| {
        if record_data.is_null() || record_count == 0 {
            return;
        }
        let records = std::slice::from_raw_parts(record_data, record_count);
        let info = &dispatch_data.dispatch_info;

        let mut state = match GLOBAL_STATE.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        let num_counters = state.counter_names.len();
        if num_counters == 0 {
            return;
        }

        let kernel_name = state
            .kernel_names
            .get(&info.kernel_id)
            .cloned()
            .unwrap_or_else(|| format!("kernel_{}", info.kernel_id));

        // Skip internal ROCm runtime kernels.
        if kernel_name.starts_with("__amd_rocclr_") {
            return;
        }

        let device_index = state
            .agents
            .get(&info.agent_id.handle)
            .map(|a| a.device_index)
            .unwrap_or(0);

        // Sum counter values across all instances (per-SE, per-XCD dimensions).
        // rocprofiler returns multiple records per counter — one per hardware
        // instance. Track instance counts for counters that need averaging.
        let mut values = vec![0.0_f64; num_counters];
        let mut instance_counts = vec![0u32; num_counters];
        for record in records {
            let mut counter_id = rocprofiler_counter_id_t { handle: 0 };
            let status = rocprofiler_query_record_counter_id(record.id, &mut counter_id);
            if status != ROCPROFILER_STATUS_SUCCESS {
                continue;
            }
            if let Some(&idx) = state.counter_id_to_index.get(&counter_id.handle) {
                values[idx] += record.counter_value;
                instance_counts[idx] += 1;
            }
        }

        // Apply per-counter aggregation based on the suffix in the metric
        // name. Counters with a "_avg" suffix that was NOT a native
        // rocprofiler name are averaged across instances. Native suffixes
        // (like "TA_BUSY_avr", "TCP_TCC_READ_REQ_sum") are already
        // aggregated by rocprofiler and left as sums here.
        for (idx, name) in state.counter_names.iter().enumerate() {
            if name.ends_with("_avr") && instance_counts[idx] > 1 {
                values[idx] /= instance_counts[idx] as f64;
            }
        }

        // Compute synthetic GRBM_TIME_DUR_max from elapsed cycles and
        // max clock frequency. duration_ns = elapsed_cycles / freq_hz * 1e9.
        let max_clk_mhz = state
            .agents
            .get(&info.agent_id.handle)
            .map(|a| a.max_engine_clk_fcompute)
            .unwrap_or(0);
        let elapsed_idx = state
            .counter_names
            .iter()
            .position(|n| n == "GRBM_GUI_ACTIVE_avr");
        let duration_idx = state
            .counter_names
            .iter()
            .position(|n| n == "GRBM_TIME_DUR_max");
        if let (Some(ei), Some(di)) = (elapsed_idx, duration_idx) {
            if max_clk_mhz > 0 {
                let elapsed_cycles = values[ei];
                let freq_hz = max_clk_mhz as f64 * 1_000_000.0;
                values[di] = elapsed_cycles / freq_hz * 1e9;
            }
        }

        // Retrieve the profiled_instances bitmask stored by dispatch_counting_callback.
        let profiled_instances = state
            .dispatch_profiled_instances
            .remove(&dispatch_data.correlation_id.internal)
            .unwrap_or(0xFF); // default: all instances if no filtering was applied

        state.counter_results.push(CounterResult {
            kernel_name,
            start_ns: dispatch_data.start_timestamp,
            end_ns: dispatch_data.end_timestamp,
            device_index,
            values,
            profiled_instances,
        });
    });
}
