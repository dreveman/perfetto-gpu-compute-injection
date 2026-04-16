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

mod callbacks;
mod metrics;
pub mod rocprofiler_sys;
mod state;

use perfetto_gpu_compute_injection::injection_log;
use perfetto_gpu_compute_injection::tracing::{
    get_counter_config, get_counters_data_source, get_renderstages_data_source, register_backend,
    GpuBackend,
};
use perfetto_sdk::{
    data_source::{StopGuard, TraceContext},
    protos::{
        common::builtin_clock::BuiltinClock,
        trace::{
            interned_data::interned_data::InternedData,
            trace_packet::{TracePacket, TracePacketSequenceFlags},
        },
    },
    track_event::TrackEvent,
    track_event_categories,
};
use perfetto_sdk_protos_gpu::protos::{
    common::gpu_counter_descriptor::{
        GpuCounterDescriptor, GpuCounterDescriptorGpuCounterGroup,
        GpuCounterDescriptorGpuCounterGroupSpec, GpuCounterDescriptorGpuCounterSpec,
    },
    trace::{
        gpu::{
            gpu_counter_event::{
                GpuCounterEvent, GpuCounterEventGpuCounter, InternedGpuCounterDescriptor,
            },
            gpu_render_stage_event::{
                GpuRenderStageEvent, GpuRenderStageEventExtraData,
                InternedGpuRenderStageSpecification,
                InternedGpuRenderStageSpecificationRenderStageCategory, InternedGraphicsContext,
                InternedGraphicsContextApi,
            },
        },
        interned_data::interned_data::prelude::*,
        trace_packet::prelude::*,
    },
};
use rocprofiler_sys::*;
use state::{ConsumerStartOffsets, CounterConsumerStartOffsets, GLOBAL_STATE};
use std::{collections::HashSet, panic};

// ---------------------------------------------------------------------------
// Track event categories for HIP API call tracing
// ---------------------------------------------------------------------------

track_event_categories! {
    pub mod hip_te_ns {
        ( "hip", "HIP Runtime API calls", [ "api" ] ),
    }
}
use hip_te_ns as perfetto_te_ns;

// IID for the HIP Compute stage specification.
const AMD_KERNEL_STAGE_IID: u64 = 1;
const AMD_MEMCPY_STAGE_IID: u64 = 2;
const AMD_MEMSET_STAGE_IID: u64 = 3;
// Queue IID base offset to avoid collision with stage IIDs.
const AMD_HW_QUEUE_IID_OFFSET: u64 = 1000;

// ---------------------------------------------------------------------------
// RocprofilerBackend implementation
// ---------------------------------------------------------------------------

struct RocprofilerBackend;

/// Collected counter event data for emission outside of GLOBAL_STATE lock.
struct CollectedCounterEvent {
    start_ns: u64,
    end_ns: u64,
    device_index: i32,
    values: Vec<f64>,
    /// Bitmask of instance IDs that want counters for this dispatch.
    profiled_instances: u8,
}

impl GpuBackend for RocprofilerBackend {
    fn default_data_source_suffix(&self) -> &'static str {
        "amd"
    }

    fn on_first_consumer_start(&self) {
        // Start the rocprofiler tracing context so buffer records flow.
        let context_handle = GLOBAL_STATE.lock().ok().and_then(|s| s.tracing_context);
        if let Some(handle) = context_handle {
            let ctx = rocprofiler_context_id_t { handle };
            let status = unsafe { rocprofiler_start_context(ctx) };
            if status != ROCPROFILER_STATUS_SUCCESS {
                injection_log!("rocprofiler_start_context failed: {}", status);
            }
        }
    }

    fn on_renderstages_start_no_counters(&self) {
        // No-op for RocProfiler: context is already started by on_first_consumer_start.
    }

    fn register_renderstages_consumer(&self, inst_id: u32) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            let offsets = ConsumerStartOffsets::snapshot(&state);
            state.renderstages_consumers.insert(inst_id, offsets);
        }
    }

    fn run_teardown(&self) {
        // For AMD: stop the tracing context and flush the buffer.
        let (context_handle, buffer_handle) = {
            let state = match GLOBAL_STATE.lock() {
                Ok(s) => s,
                Err(_) => return,
            };
            (state.tracing_context, state.tracing_buffer)
        };

        if let Some(handle) = context_handle {
            let ctx = rocprofiler_context_id_t { handle };
            let _ = unsafe { rocprofiler_stop_context(ctx) };
        }

        if let Some(handle) = buffer_handle {
            let buf = rocprofiler_buffer_id_t { handle };
            let _ = unsafe { rocprofiler_flush_buffer(buf) };
        }
    }

    fn flush_activity_buffers(&self) {
        let buffer_handle = GLOBAL_STATE.lock().ok().and_then(|s| s.tracing_buffer);
        if let Some(handle) = buffer_handle {
            let buf = rocprofiler_buffer_id_t { handle };
            let _ = unsafe { rocprofiler_flush_buffer(buf) };
        }
    }

    fn emit_renderstage_events_for_instance(&self, inst_id: u32, stop_guard: Option<StopGuard>) {
        let _ = panic::catch_unwind(|| {
            let (process_id, process_name) =
                perfetto_gpu_compute_injection::config::get_process_info();

            // Phase 1: Collect all event data under GLOBAL_STATE lock, then release.
            struct PendingRenderStageEvent {
                start_ns: u64,
                end_ns: u64,
                gpu_id: i32,
                hw_queue_iid: u64,
                stage_iid: u64,
                name: String,
                extra_fields: Vec<(String, String)>,
                correlation_id: u64,
            }

            let (events, queues) = {
                let mut state = match GLOBAL_STATE.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let start_offsets = if stop_guard.is_some() {
                    match state.renderstages_consumers.remove(&inst_id) {
                        Some(o) => o,
                        None => return,
                    }
                } else {
                    match state.renderstages_consumers.get(&inst_id).cloned() {
                        Some(o) => o,
                        None => return,
                    }
                };
                let kd_start = start_offsets.kernel_dispatches;
                let mc_start = start_offsets.memcopies;
                let ms_start = start_offsets.memsets;

                // Collect unique queue handles for interned specs.
                let mut queues: std::collections::HashSet<u64> = std::collections::HashSet::new();
                for kd in state.kernel_dispatches[kd_start..].iter() {
                    queues.insert(kd.queue_handle);
                }

                let mut events: Vec<PendingRenderStageEvent> = Vec::new();

                // Kernel dispatch events.
                for kd in state.kernel_dispatches[kd_start..].iter() {
                    let demangled =
                        perfetto_gpu_compute_injection::kernel::demangle_name(&kd.kernel_name);
                    // rocprofiler's grid_size is total work-items per dimension
                    // (gridDim * blockDim), not the number of blocks. Derive
                    // the actual grid (block count) by dividing out the
                    // workgroup size.
                    let grid_x = kd.grid.0.checked_div(kd.workgroup.0).unwrap_or(kd.grid.0);
                    let grid_y = kd.grid.1.checked_div(kd.workgroup.1).unwrap_or(kd.grid.1);
                    let grid_z = kd.grid.2.checked_div(kd.workgroup.2).unwrap_or(kd.grid.2);
                    let grid_size = grid_x * grid_y * grid_z;
                    let workgroup_size = kd.workgroup.0 * kd.workgroup.1 * kd.workgroup.2;
                    let thread_count = kd.grid.0 * kd.grid.1 * kd.grid.2;
                    let total_waves = if kd.wave_front_size > 0 {
                        thread_count.div_ceil(kd.wave_front_size)
                    } else {
                        0
                    };
                    let waves_per_cu = if kd.cu_count > 0 {
                        total_waves as f64 / kd.cu_count as f64
                    } else {
                        0.0
                    };
                    let hw_queue_iid = (kd.queue_handle & 0xFFFF) + AMD_HW_QUEUE_IID_OFFSET;
                    // max_engine_clk_fcompute is in MHz.
                    let clock_freq_hz = kd.max_engine_clk_fcompute as f64 * 1_000_000.0;
                    let extra_fields: Vec<(String, String)> = vec![
                        ("kernel_name".to_string(), kd.kernel_name.clone()),
                        ("kernel_demangled_name".to_string(), demangled.clone()),
                        ("process_id".to_string(), process_id.to_string()),
                        ("process_name".to_string(), process_name.clone()),
                        ("device_id".to_string(), kd.device_index.to_string()),
                        ("arch".to_string(), kd.arch.clone()),
                        ("queue_id".to_string(), kd.queue_handle.to_string()),
                        ("launch__grid_size".to_string(), grid_size.to_string()),
                        ("launch__grid_size_x".to_string(), grid_x.to_string()),
                        ("launch__grid_size_y".to_string(), grid_y.to_string()),
                        ("launch__grid_size_z".to_string(), grid_z.to_string()),
                        ("launch__block_size".to_string(), workgroup_size.to_string()),
                        (
                            "launch__block_size_x".to_string(),
                            kd.workgroup.0.to_string(),
                        ),
                        (
                            "launch__block_size_y".to_string(),
                            kd.workgroup.1.to_string(),
                        ),
                        (
                            "launch__block_size_z".to_string(),
                            kd.workgroup.2.to_string(),
                        ),
                        ("launch__thread_count".to_string(), thread_count.to_string()),
                        (
                            "launch__waves_per_multiprocessor".to_string(),
                            format!("{:.2}", waves_per_cu),
                        ),
                        (
                            "GRBM_GUI_ACTIVE_avr_per_second".to_string(),
                            format!("{:.0}", clock_freq_hz),
                        ),
                    ];
                    events.push(PendingRenderStageEvent {
                        start_ns: kd.start_ns,
                        end_ns: kd.end_ns,
                        gpu_id: kd.device_index,
                        hw_queue_iid,
                        stage_iid: AMD_KERNEL_STAGE_IID,
                        name: perfetto_gpu_compute_injection::kernel::simplify_name(&demangled)
                            .to_string(),
                        extra_fields,
                        correlation_id: kd.correlation_id,
                    });
                }

                // Memory copy events.
                for mc in state.memcopies[mc_start..].iter() {
                    let memcpy_name = match mc.direction {
                        1 => "Memcpy HtoH",
                        2 => "Memcpy HtoD",
                        3 => "Memcpy DtoH",
                        4 => "Memcpy DtoD",
                        _ => "Memcpy",
                    };
                    let extra_fields: Vec<(String, String)> = vec![
                        ("process_id".to_string(), process_id.to_string()),
                        ("process_name".to_string(), process_name.clone()),
                        ("device_id".to_string(), mc.device_index.to_string()),
                        ("size_bytes".to_string(), mc.bytes.to_string()),
                        ("direction".to_string(), mc.direction.to_string()),
                    ];
                    events.push(PendingRenderStageEvent {
                        start_ns: mc.start_ns,
                        end_ns: mc.end_ns,
                        gpu_id: mc.device_index,
                        hw_queue_iid: AMD_HW_QUEUE_IID_OFFSET,
                        stage_iid: AMD_MEMCPY_STAGE_IID,
                        name: memcpy_name.to_string(),
                        extra_fields,
                        correlation_id: mc.correlation_id,
                    });
                }

                // Memory set events.
                for ms in state.memsets[ms_start..].iter() {
                    let extra_fields: Vec<(String, String)> = vec![
                        ("process_id".to_string(), process_id.to_string()),
                        ("process_name".to_string(), process_name.clone()),
                        ("device_id".to_string(), ms.device_index.to_string()),
                    ];
                    events.push(PendingRenderStageEvent {
                        start_ns: ms.start_ns,
                        end_ns: ms.end_ns,
                        gpu_id: ms.device_index,
                        hw_queue_iid: AMD_HW_QUEUE_IID_OFFSET,
                        stage_iid: AMD_MEMSET_STAGE_IID,
                        name: "Memset".to_string(),
                        extra_fields,
                        correlation_id: ms.correlation_id,
                    });
                }

                let emitted = events.len();
                injection_log!(
                    "emitted {} AMD render stage events (instance {})",
                    emitted,
                    inst_id
                );

                (events, queues)
                // state (GLOBAL_STATE lock) dropped here
            };

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_callback which also needs GLOBAL_STATE.
            let mut stop_guard_opt = stop_guard;
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }

                ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                    let was_cleared =
                        std::mem::replace(&mut inc_state.was_cleared, false);

                    if was_cleared {
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet.set_sequence_flags(
                                TracePacketSequenceFlags::SeqIncrementalStateCleared as u32,
                            );
                            packet.set_interned_data(|interned: &mut InternedData| {
                                interned.set_graphics_contexts(
                                    |gctx: &mut InternedGraphicsContext| {
                                        gctx.set_iid(1);
                                        gctx.set_pid(process_id);
                                        gctx.set_api(InternedGraphicsContextApi::Hip);
                                    },
                                );
                                for (idx, &queue_handle) in queues.iter().enumerate() {
                                    let iid =
                                        (queue_handle & 0xFFFF) + AMD_HW_QUEUE_IID_OFFSET;
                                    interned.set_gpu_specifications(
                                        |spec: &mut InternedGpuRenderStageSpecification| {
                                            spec.set_iid(iid);
                                            spec.set_name(format!(
                                                "Queue #{}",
                                                idx + 1
                                            ));
                                            spec.set_category(
                                                InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                                            );
                                        },
                                    );
                                }
                                interned.set_gpu_specifications(
                                    |spec: &mut InternedGpuRenderStageSpecification| {
                                        spec.set_iid(AMD_KERNEL_STAGE_IID);
                                        spec.set_name("Kernel");
                                        spec.set_description("HIP Kernel");
                                        spec.set_category(
                                            InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                                        );
                                    },
                                );
                                interned.set_gpu_specifications(
                                    |spec: &mut InternedGpuRenderStageSpecification| {
                                        spec.set_iid(AMD_MEMCPY_STAGE_IID);
                                        spec.set_name("MemoryTransfer");
                                        spec.set_description("HIP Memory Transfer");
                                        spec.set_category(
                                            InternedGpuRenderStageSpecificationRenderStageCategory::Other,
                                        );
                                    },
                                );
                                interned.set_gpu_specifications(
                                    |spec: &mut InternedGpuRenderStageSpecification| {
                                        spec.set_iid(AMD_MEMSET_STAGE_IID);
                                        spec.set_name("MemorySet");
                                        spec.set_description("HIP Memory Set");
                                        spec.set_category(
                                            InternedGpuRenderStageSpecificationRenderStageCategory::Other,
                                        );
                                    },
                                );
                            });
                        });
                    }

                    for event in &events {
                        let duration_ns = event.end_ns.saturating_sub(event.start_ns);
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(event.start_ns)
                                .set_timestamp_clock_id(
                                    BuiltinClock::BuiltinClockBoottime.into(),
                                )
                                .set_gpu_render_stage_event(
                                    |re: &mut GpuRenderStageEvent| {
                                        re.set_event_id(event.correlation_id)
                                            .set_duration(duration_ns)
                                            .set_gpu_id(event.gpu_id)
                                            .set_hw_queue_iid(event.hw_queue_iid)
                                            .set_stage_iid(event.stage_iid)
                                            .set_context(1)
                                            .set_name(&event.name);
                                        for (name, value) in &event.extra_fields {
                                            re.set_extra_data(
                                                |ed: &mut GpuRenderStageEventExtraData| {
                                                    ed.set_name(name);
                                                    ed.set_value(value);
                                                },
                                            );
                                        }
                                    },
                                );
                        });
                    }
                });

                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);
        });
    }

    fn on_first_counters_start(&self) {
        // Set up per-agent counter configs and configure the dispatch counting
        // service. This is deferred to first consumer start so we don't
        // enumerate counters when only gpu.renderstages is enabled.
        use callbacks::{dispatch_counting_callback, record_counting_callback};

        // Compute the union of counter_names across all active instances.
        // The profiler collects all requested metrics; each instance only
        // sees the subset it asked for during emission.
        let mut union_names: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for id in 0..8u32 {
            if let Some(cfg) = get_counter_config(id) {
                for name in &cfg.counter_names {
                    if seen.insert(name.clone()) {
                        union_names.push(name.clone());
                    }
                }
            }
        }
        if !union_names.is_empty() {
            if let Ok(mut state) = GLOBAL_STATE.lock() {
                injection_log!(
                    "using {} counter names from trace config (union of all instances)",
                    union_names.len()
                );
                state.config.metrics = union_names;
            }
        }

        let requested_metrics: Vec<String> = GLOBAL_STATE
            .lock()
            .ok()
            .map(|s| s.config.metrics.clone())
            .unwrap_or_default();
        let agent_handles: Vec<u64> = GLOBAL_STATE
            .lock()
            .ok()
            .map(|s| s.agents.keys().copied().collect())
            .unwrap_or_default();

        for &agent_handle in &agent_handles {
            let agent_id = rocprofiler_agent_id_t {
                handle: agent_handle,
            };

            // Enumerate available counters for this agent.
            let mut available_counter_ids: Vec<rocprofiler_counter_id_t> = Vec::new();
            unsafe extern "C" fn counters_cb(
                _agent_id: rocprofiler_agent_id_t,
                counters: *mut rocprofiler_counter_id_t,
                num_counters: usize,
                user_data: *mut std::os::raw::c_void,
            ) -> rocprofiler_status_t {
                if !counters.is_null() && num_counters > 0 {
                    let out = &mut *(user_data as *mut Vec<rocprofiler_counter_id_t>);
                    let slice = std::slice::from_raw_parts(counters, num_counters);
                    out.extend_from_slice(slice);
                }
                ROCPROFILER_STATUS_SUCCESS
            }
            let status = unsafe {
                rocprofiler_iterate_agent_supported_counters(
                    agent_id,
                    Some(counters_cb),
                    &mut available_counter_ids as *mut Vec<rocprofiler_counter_id_t>
                        as *mut std::os::raw::c_void,
                )
            };
            if status != ROCPROFILER_STATUS_SUCCESS {
                injection_log!(
                    "agent {:#x}: rocprofiler_iterate_agent_supported_counters failed: {}",
                    agent_handle,
                    status
                );
            }

            // Build name → counter_id map from available counters.
            let mut name_to_id: std::collections::HashMap<String, rocprofiler_counter_id_t> =
                std::collections::HashMap::new();
            for &cid in &available_counter_ids {
                let mut info = std::mem::MaybeUninit::<rocprofiler_counter_info_v0_t>::zeroed();
                let status = unsafe {
                    rocprofiler_query_counter_info(
                        cid,
                        ROCPROFILER_COUNTER_INFO_VERSION_0,
                        info.as_mut_ptr() as *mut std::os::raw::c_void,
                    )
                };
                if status != ROCPROFILER_STATUS_SUCCESS {
                    continue;
                }
                let info = unsafe { info.assume_init() };
                if info.name.is_null() {
                    continue;
                }
                let name = unsafe { std::ffi::CStr::from_ptr(info.name) }
                    .to_string_lossy()
                    .into_owned();
                name_to_id.insert(name, cid);
            }

            injection_log!(
                "agent {:#x}: {} counters available",
                agent_handle,
                name_to_id.len()
            );

            // Match requested metrics against available counters.
            // Some counters have a built-in aggregation suffix that
            // rocprofiler knows natively (e.g. "TCP_TCC_READ_REQ_sum",
            // "TA_BUSY_avr"). Try the full name first. If not found,
            // strip a "_avg", "_sum", or "_max" suffix, look up the base
            // counter, and apply the aggregation in record_counting_callback.
            let mut matched_ids: Vec<rocprofiler_counter_id_t> = Vec::new();
            let mut matched_names: Vec<String> = Vec::new();
            for metric in &requested_metrics {
                if let Some(&cid) = name_to_id.get(metric.as_str()) {
                    // Counter exists as-is in rocprofiler (native suffix).
                    matched_ids.push(cid);
                    matched_names.push(metric.clone());
                } else {
                    let base_name = metric
                        .strip_suffix("_avr")
                        .or_else(|| metric.strip_suffix("_sum"))
                        .or_else(|| metric.strip_suffix("_max"))
                        .or_else(|| metric.strip_suffix("_min"));
                    if let Some((base, cid)) =
                        base_name.and_then(|b| name_to_id.get(b).map(|&c| (b, c)))
                    {
                        matched_ids.push(cid);
                        matched_names.push(metric.clone());
                        injection_log!(
                            "agent {:#x}: '{}' -> base counter '{}'",
                            agent_handle,
                            metric,
                            base
                        );
                    } else {
                        injection_log!(
                            "agent {:#x}: requested counter '{}' not available",
                            agent_handle,
                            metric
                        );
                    }
                }
            }

            if matched_ids.is_empty() {
                injection_log!(
                    "agent {:#x}: no matching counters found, skipping counter config",
                    agent_handle
                );
                continue;
            }

            injection_log!(
                "agent {:#x}: configuring {} counters: {:?}",
                agent_handle,
                matched_names.len(),
                matched_names
            );

            // Create counter config for this agent.
            let mut config_id = rocprofiler_counter_config_id_t { handle: 0 };
            let status = unsafe {
                rocprofiler_create_counter_config(
                    agent_id,
                    matched_ids.as_mut_ptr(),
                    matched_ids.len(),
                    &mut config_id,
                )
            };
            if status != ROCPROFILER_STATUS_SUCCESS {
                injection_log!(
                    "agent {:#x}: rocprofiler_create_counter_config failed: {}",
                    agent_handle,
                    status
                );
                continue;
            }

            if let Ok(mut state) = GLOBAL_STATE.lock() {
                state.counter_configs.insert(agent_handle, config_id.handle);
                // Build counter_id_to_index mapping (only needed once, same for all agents).
                if state.counter_names.is_empty() {
                    state.counter_names = matched_names;
                    for (idx, cid) in matched_ids.iter().enumerate() {
                        state.counter_id_to_index.insert(cid.handle, idx);
                    }
                    // Append synthetic counters (computed from hardware
                    // counters, not collected by rocprofiler).
                    for &name in metrics::SYNTHETIC_COUNTERS {
                        state.counter_names.push(name.to_string());
                    }
                }
            }
        }

        // Configure callback dispatch counting service on the tracing context.
        // The context may already be started (by on_first_consumer_start), so we
        // must stop it first — rocprofiler requires service configuration before
        // context start.
        let has_counter_configs = GLOBAL_STATE
            .lock()
            .ok()
            .map(|s| !s.counter_configs.is_empty())
            .unwrap_or(false);
        if has_counter_configs {
            let context_handle = GLOBAL_STATE.lock().ok().and_then(|s| s.tracing_context);
            if let Some(handle) = context_handle {
                let tracing_ctx = rocprofiler_context_id_t { handle };
                // Stop context so we can add the counting service.
                let _ = unsafe { rocprofiler_stop_context(tracing_ctx) };
                let status = unsafe {
                    rocprofiler_configure_callback_dispatch_counting_service(
                        tracing_ctx,
                        Some(dispatch_counting_callback),
                        std::ptr::null_mut(),
                        Some(record_counting_callback),
                        std::ptr::null_mut(),
                    )
                };
                if status != ROCPROFILER_STATUS_SUCCESS {
                    injection_log!(
                        "rocprofiler_configure_callback_dispatch_counting_service failed: {}",
                        status
                    );
                }
                // Restart context with the counting service now configured.
                let status = unsafe { rocprofiler_start_context(tracing_ctx) };
                if status != ROCPROFILER_STATUS_SUCCESS {
                    injection_log!(
                        "rocprofiler_start_context after counter config failed: {}",
                        status
                    );
                }
            }
        }
    }

    fn register_counters_consumer(&self, inst_id: u32) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            let offsets = CounterConsumerStartOffsets {
                counter_results: state.counter_results.len(),
            };
            state.counters_consumers.insert(inst_id, offsets);
        }
    }

    fn emit_counter_events_for_instance(&self, inst_id: u32, stop_guard: Option<StopGuard>) {
        let _ = panic::catch_unwind(|| {
            // Phase 1: Collect data under GLOBAL_STATE lock, then release.
            let (collected_events, counter_names) = {
                let mut state = match GLOBAL_STATE.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let start_offsets = if stop_guard.is_some() {
                    match state.counters_consumers.remove(&inst_id) {
                        Some(o) => o,
                        None => return,
                    }
                } else {
                    match state.counters_consumers.get(&inst_id).cloned() {
                        Some(o) => o,
                        None => return,
                    }
                };
                let counter_names = state.counter_names.clone();
                if counter_names.is_empty() {
                    return;
                }
                let cr_start = start_offsets.counter_results;
                let events: Vec<CollectedCounterEvent> = state.counter_results[cr_start..]
                    .iter()
                    .map(|r| CollectedCounterEvent {
                        start_ns: r.start_ns,
                        end_ns: r.end_ns,
                        device_index: r.device_index,
                        values: r.values.clone(),
                        profiled_instances: r.profiled_instances,
                    })
                    .collect();
                let emitted = events.len();
                injection_log!("emitted {} counter events (instance {})", emitted, inst_id);
                (events, counter_names)
                // state (GLOBAL_STATE lock) dropped here
            };

            // Per-instance filtering: skip dispatches where this instance's
            // bit is not set in profiled_instances.
            let inst_bit = 1u8 << inst_id;
            let collected_events: Vec<_> = collected_events
                .into_iter()
                .filter(|e| e.profiled_instances & inst_bit != 0)
                .collect();

            // Filter counter_names and values to only those requested by
            // this instance's config. When counter_names is empty in the
            // config (env var fallback), emit all counters.
            let (filtered_names, filter_indices): (Vec<String>, Option<Vec<usize>>) =
                if let Some(cfg) = get_counter_config(inst_id) {
                    if cfg.counter_names.is_empty() {
                        (counter_names, None)
                    } else {
                        let requested: HashSet<&str> =
                            cfg.counter_names.iter().map(|s| s.as_str()).collect();
                        let indices: Vec<usize> = counter_names
                            .iter()
                            .enumerate()
                            .filter(|(_, n)| requested.contains(n.as_str()))
                            .map(|(i, _)| i)
                            .collect();
                        let names: Vec<String> =
                            indices.iter().map(|&i| counter_names[i].clone()).collect();
                        (names, Some(indices))
                    }
                } else {
                    (counter_names, None)
                };

            if filtered_names.is_empty() {
                return;
            }

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_callback which also needs GLOBAL_STATE.
            let gpu_ids: HashSet<i32> = collected_events.iter().map(|e| e.device_index).collect();
            let mut stop_guard_opt = stop_guard;
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for result in &collected_events {
                    let gpu_id = result.device_index;
                    ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                        let was_cleared = std::mem::replace(&mut inc_state.was_cleared, false);
                        if was_cleared {
                            emit_interned_counter_descriptors(ctx, &filtered_names, &gpu_ids);
                        }
                        let desc_iid = gpu_id as u64 + 1;
                        // Emit start sample (zero values).
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(result.start_ns)
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    event.set_counter_descriptor_iid(desc_iid);
                                    for i in 0..filtered_names.len() {
                                        event.set_counters(
                                            |counter: &mut GpuCounterEventGpuCounter| {
                                                counter.set_counter_id(i as u32).set_int_value(0);
                                            },
                                        );
                                    }
                                });
                        });
                        // Emit end sample (actual values).
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(result.end_ns)
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    event.set_counter_descriptor_iid(desc_iid);
                                    let values_iter: Vec<f64> = match &filter_indices {
                                        Some(indices) => indices
                                            .iter()
                                            .filter_map(|&i| result.values.get(i).copied())
                                            .collect(),
                                        None => result.values.clone(),
                                    };
                                    for (i, value) in values_iter.iter().enumerate() {
                                        event.set_counters(
                                            |counter: &mut GpuCounterEventGpuCounter| {
                                                counter
                                                    .set_counter_id(i as u32)
                                                    .set_double_value(*value);
                                            },
                                        );
                                    }
                                });
                        });
                    });
                }
                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);
        });
    }

    fn flush_renderstage_events(&self) {
        // Force rocprofiler to deliver buffered records so that
        // kernel_dispatches / memcopies / memsets vectors are up to date.
        self.flush_activity_buffers();
        let inst_ids: Vec<u32> = GLOBAL_STATE
            .lock()
            .map(|s| s.renderstages_consumers.keys().copied().collect())
            .unwrap_or_default();
        for inst_id in inst_ids {
            self.emit_renderstage_events_for_instance(inst_id, None);
        }
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            state.advance_and_drain_renderstage_events();
        }
    }

    fn flush_counter_events(&self) {
        // Force rocprofiler to deliver buffered records so that
        // kernel_dispatches are up to date for counter emission.
        self.flush_activity_buffers();
        let inst_ids: Vec<u32> = GLOBAL_STATE
            .lock()
            .map(|s| s.counters_consumers.keys().copied().collect())
            .unwrap_or_default();
        for inst_id in inst_ids {
            self.emit_counter_events_for_instance(inst_id, None);
        }
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            state.advance_and_drain_counter_events();
        }
    }
}

// ---------------------------------------------------------------------------
// HIP API call track event emission
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Atexit fallback
// ---------------------------------------------------------------------------

extern "C" fn end_execution() {
    let _ = panic::catch_unwind(|| {
        let amd = RocprofilerBackend;
        amd.run_teardown();
        let (renderstage_ids, counter_ids): (Vec<u32>, Vec<u32>) = GLOBAL_STATE
            .lock()
            .map(|s| {
                (
                    s.renderstages_consumers.keys().copied().collect(),
                    s.counters_consumers.keys().copied().collect(),
                )
            })
            .unwrap_or_default();
        for inst_id in renderstage_ids {
            amd.emit_renderstage_events_for_instance(inst_id, None);
        }
        for inst_id in counter_ids {
            amd.emit_counter_events_for_instance(inst_id, None);
        }
    });
}

// ---------------------------------------------------------------------------
// Counter descriptor interning helpers
// ---------------------------------------------------------------------------

/// Hardware block prefix groups for organizing instrumented counters.
/// Each entry is (metric_name_prefix, display_name).
const COUNTER_GROUPS: &[(&str, &str)] = &[
    ("SQ_", "SQ"),
    ("GRBM_", "GRBM"),
    ("TA_", "TA"),
    ("TCP_", "TCP"),
    ("TCC_", "TCC"),
];

/// Emit interned counter descriptors for all GPUs in the batch.
///
/// Each GPU gets one `InternedGpuCounterDescriptor` with iid = gpu_id + 1,
/// containing all counter specs with simple 0-based counter_ids. The gpu_id
/// on the interned descriptor handles per-GPU track separation.
///
/// Counters are organized into hardware block groups based on their metric
/// name prefix (e.g. `SQ_` → "SQ", `TCC_` → "TCC").
fn emit_interned_counter_descriptors(
    ctx: &mut TraceContext,
    counter_names: &[String],
    gpu_ids: &HashSet<i32>,
) {
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_sequence_flags(TracePacketSequenceFlags::SeqIncrementalStateCleared as u32);
        packet.set_interned_data(|interned: &mut InternedData| {
            for &gpu_id in gpu_ids {
                interned.set_gpu_counter_descriptors(|desc: &mut InternedGpuCounterDescriptor| {
                    desc.set_iid(gpu_id as u64 + 1);
                    desc.set_gpu_id(gpu_id);
                    desc.set_counter_descriptor(|cd: &mut GpuCounterDescriptor| {
                        for (i, name) in counter_names.iter().enumerate() {
                            cd.set_specs(|spec: &mut GpuCounterDescriptorGpuCounterSpec| {
                                spec.set_counter_id(i as u32);
                                spec.set_name(name);
                                spec.set_groups(GpuCounterDescriptorGpuCounterGroup::Compute);
                            });
                        }
                        // Group counters by hardware block prefix.
                        for (group_id, &(prefix, group_name)) in COUNTER_GROUPS.iter().enumerate() {
                            let member_ids: Vec<u32> = counter_names
                                .iter()
                                .enumerate()
                                .filter(|(_, name)| name.starts_with(prefix))
                                .map(|(i, _)| i as u32)
                                .collect();
                            if !member_ids.is_empty() {
                                cd.set_counter_groups(
                                    |g: &mut GpuCounterDescriptorGpuCounterGroupSpec| {
                                        g.set_group_id(group_id as u32);
                                        g.set_name(group_name);
                                        for &id in &member_ids {
                                            g.set_counter_ids(id);
                                        }
                                    },
                                );
                            }
                        }
                        cd.set_supports_instrumented_sampling(true);
                        cd.set_supports_counter_names(true);
                        cd.set_supports_counter_name_globs(true);
                    });
                });
            }
        });
    });
}

// ---------------------------------------------------------------------------
// AMD rocprofiler initialization helpers
// ---------------------------------------------------------------------------

/// Initialize AMD: populate agent map, create contexts and buffers.
/// Called from `tool_initialize` during `rocprofiler_configure`.
fn initialize_rocprofiler() -> rocprofiler_status_t {
    use callbacks::{agents_callback, buffer_callback, code_object_callback};

    // Enumerate GPU agents.
    unsafe {
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            Some(agents_callback),
            std::mem::size_of::<rocprofiler_agent_v0_t>(),
            std::ptr::null_mut(),
        )
    };

    // Create utility context (always-on) for code object callbacks.
    let mut utility_ctx = rocprofiler_context_id_t { handle: 0 };
    let status = unsafe { rocprofiler_create_context(&mut utility_ctx) };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Configure code object kernel symbol callback on utility context.
    let status = unsafe {
        rocprofiler_configure_callback_tracing_service(
            utility_ctx,
            ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
            std::ptr::null_mut(),
            0,
            Some(code_object_callback),
            std::ptr::null_mut(),
        )
    };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Start utility context immediately (it always runs).
    let status = unsafe { rocprofiler_start_context(utility_ctx) };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Create tracing context (started/stopped when Perfetto consumers connect).
    let mut tracing_ctx = rocprofiler_context_id_t { handle: 0 };
    let status = unsafe { rocprofiler_create_context(&mut tracing_ctx) };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Create buffer for kernel dispatch and memory copy records.
    let mut buffer_id = rocprofiler_buffer_id_t { handle: 0 };
    let status = unsafe {
        rocprofiler_create_buffer(
            tracing_ctx,
            4 * 1024 * 1024, // 4 MiB buffer
            0,               // flush on every record
            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
            Some(buffer_callback),
            std::ptr::null_mut(),
            &mut buffer_id,
        )
    };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Configure kernel dispatch buffer tracing.
    let status = unsafe {
        rocprofiler_configure_buffer_tracing_service(
            tracing_ctx,
            ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
            std::ptr::null_mut(),
            0,
            buffer_id,
        )
    };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Configure memory copy buffer tracing.
    let status = unsafe {
        rocprofiler_configure_buffer_tracing_service(
            tracing_ctx,
            ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
            std::ptr::null_mut(),
            0,
            buffer_id,
        )
    };
    if status != ROCPROFILER_STATUS_SUCCESS {
        return status;
    }

    // Configure HIP runtime API buffer tracing. Records are always collected;
    // emission is gated on the "hip" track event category being enabled.
    let status = unsafe {
        rocprofiler_configure_buffer_tracing_service(
            tracing_ctx,
            ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
            std::ptr::null_mut(),
            0,
            buffer_id,
        )
    };
    if status != ROCPROFILER_STATUS_SUCCESS {
        injection_log!(
            "rocprofiler_configure_buffer_tracing_service (HIP_RUNTIME_API) failed: {}",
            status
        );
        // Non-fatal: API call tracing is optional.
    }

    // Store context and buffer handles in global state.
    if let Ok(mut state) = GLOBAL_STATE.lock() {
        state.utility_context = Some(utility_ctx.handle);
        state.tracing_context = Some(tracing_ctx.handle);
        state.tracing_buffer = Some(buffer_id.handle);
    }

    ROCPROFILER_STATUS_SUCCESS
}

// ---------------------------------------------------------------------------
// Public C entry point
// ---------------------------------------------------------------------------

/// AMD registration entry point (called by ROCm runtime at library load).
///
/// Returns a static `rocprofiler_tool_configure_result_t` with `initialize`
/// and `finalize` function pointers.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn rocprofiler_configure(
    _version: u32,
    _runtime_version: *const std::os::raw::c_char,
    _priority: u32,
    client_id: *mut rocprofiler_client_id_t,
) -> *mut rocprofiler_tool_configure_result_t {
    let _ = panic::catch_unwind(|| {
        if !client_id.is_null() {
            unsafe {
                (*client_id).name = c"perfetto-hip-injection".as_ptr();
            }
        }
    });

    unsafe extern "C" fn tool_initialize(
        _finalize_func: rocprofiler_client_finalize_t,
        _tool_data: *mut std::os::raw::c_void,
    ) -> i32 {
        let result = panic::catch_unwind(|| {
            use perfetto_gpu_compute_injection::config::Config;
            use perfetto_sdk::producer::{Backends, Producer, ProducerInitArgsBuilder};

            let config = Config::from_env();

            // Parse metrics from environment.
            let metrics_str = std::env::var("INJECTION_METRICS").unwrap_or_default();
            let metrics = metrics::parse_metrics(&metrics_str);

            if let Ok(mut state) = crate::state::GLOBAL_STATE.lock() {
                if !state.initialized {
                    state.initialized = true;
                    state.config = config;
                    state.config.metrics = metrics;
                }
            }

            // Set up rocprofiler contexts, buffers, and services before
            // registering Perfetto data sources. Data source on_start
            // callbacks need the tracing context to be available.
            let status = initialize_rocprofiler();
            if status != ROCPROFILER_STATUS_SUCCESS {
                injection_log!("rocprofiler initialization failed: {}", status);
                return -1;
            }

            register_backend(RocprofilerBackend);

            // Initialize Perfetto producer and register data sources.
            let producer_args = ProducerInitArgsBuilder::new().backends(Backends::SYSTEM);
            Producer::init(producer_args.build());
            let _ = get_renderstages_data_source();
            let _ = get_counters_data_source();

            // Initialize track event categories for HIP API call tracing.
            // HIP API buffer tracing is configured in initialize_rocprofiler();
            // the category callback only controls whether events are emitted.
            TrackEvent::init();
            let _ = perfetto_te_ns::register();

            unsafe { libc::atexit(end_execution) };

            injection_log!("AMD rocprofiler tool initialized");
            0
        });
        result.unwrap_or(-1)
    }

    unsafe extern "C" fn tool_finalize(_tool_data: *mut std::os::raw::c_void) {
        let _ = panic::catch_unwind(|| {
            injection_log!("AMD rocprofiler tool finalizing");
        });
    }

    static mut CONFIGURE_RESULT: rocprofiler_tool_configure_result_t =
        rocprofiler_tool_configure_result_t {
            size: std::mem::size_of::<rocprofiler_tool_configure_result_t>(),
            initialize: Some(tool_initialize),
            finalize: Some(tool_finalize),
            tool_data: std::ptr::null_mut(),
        };

    std::ptr::addr_of_mut!(CONFIGURE_RESULT)
}
