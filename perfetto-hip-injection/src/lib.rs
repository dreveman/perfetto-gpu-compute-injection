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

use cpp_demangle::Symbol;
use perfetto_gpu_compute_injection::injection_log;
use perfetto_gpu_compute_injection::tracing::{
    get_counters_data_source, get_next_event_id, get_renderstages_data_source, register_backend,
    GpuBackend, GOT_FIRST_COUNTERS, GOT_FIRST_RENDERSTAGES,
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
};
use perfetto_sdk_protos_gpu::protos::{
    common::gpu_counter_descriptor::{
        GpuCounterDescriptor, GpuCounterDescriptorGpuCounterGroup,
        GpuCounterDescriptorGpuCounterSpec,
    },
    trace::{
        gpu::{
            gpu_counter_event::{GpuCounterEvent, GpuCounterEventGpuCounter},
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
use std::{panic, sync::atomic::Ordering};

// IID for the HIP Compute stage specification.
const AMD_KERNEL_STAGE_IID: u64 = 1;
const AMD_MEMCPY_STAGE_IID: u64 = 2;
// Queue IID base offset to avoid collision with stage IIDs.
const AMD_HW_QUEUE_IID_OFFSET: u64 = 1000;

// ---------------------------------------------------------------------------
// RocprofilerBackend implementation
// ---------------------------------------------------------------------------

struct RocprofilerBackend;

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
            let process_id = unsafe { libc::getpid() };
            let process_name = std::fs::read_to_string("/proc/self/comm")
                .unwrap_or_else(|_| "unknown".to_string())
                .trim_end_matches('\n')
                .to_owned();

            let mut state = match GLOBAL_STATE.lock() {
                Ok(s) => s,
                Err(_) => return,
            };
            let start_offsets = match state.renderstages_consumers.remove(&inst_id) {
                Some(o) => o,
                None => return,
            };
            let kd_start = start_offsets.kernel_dispatches;
            let mc_start = start_offsets.memcopies;

            let mut stop_guard_opt = stop_guard;
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }

                // Collect unique queue handles for interned specs.
                let mut queues: std::collections::HashSet<u64> =
                    std::collections::HashSet::new();
                for kd in state.kernel_dispatches[kd_start..].iter() {
                    queues.insert(kd.queue_handle);
                }

                let got_first = GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                let emit_interned_now = got_first & (1 << inst_id) == 0;

                if emit_interned_now {
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet.set_sequence_flags(
                            TracePacketSequenceFlags::SeqIncrementalStateCleared as u32,
                        );
                        packet.set_interned_data(|interned: &mut InternedData| {
                            // GPU context: use process as context (no CUDA ctx equivalent).
                            interned
                                .set_graphics_contexts(|gctx: &mut InternedGraphicsContext| {
                                    gctx.set_iid(1);
                                    gctx.set_pid(process_id);
                                    gctx.set_api(InternedGraphicsContextApi::Undefined);
                                });
                            // HW queue specs per distinct queue.
                            for &queue_handle in &queues {
                                let iid = (queue_handle & 0xFFFF) + AMD_HW_QUEUE_IID_OFFSET;
                                interned.set_gpu_specifications(
                                    |spec: &mut InternedGpuRenderStageSpecification| {
                                        spec.set_iid(iid);
                                        spec.set_name(format!("Queue ({})", queue_handle));
                                        spec.set_category(
                                            InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                                        );
                                    },
                                );
                            }
                            // Kernel stage spec.
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
                            // Memory copy stage spec.
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
                        });
                    });
                }

                // Emit kernel dispatch events.
                for kd in state.kernel_dispatches[kd_start..].iter() {
                    let demangled = if let Ok(sym) = Symbol::new(&kd.kernel_name) {
                        sym.demangle()
                            .map(|d| d.to_string())
                            .unwrap_or(kd.kernel_name.clone())
                    } else {
                        kd.kernel_name.clone()
                    };
                    let grid_size = kd.grid.0 * kd.grid.1 * kd.grid.2;
                    let workgroup_size = kd.workgroup.0 * kd.workgroup.1 * kd.workgroup.2;
                    let thread_count = grid_size * workgroup_size;
                    let hw_queue_iid = (kd.queue_handle & 0xFFFF) + AMD_HW_QUEUE_IID_OFFSET;
                    let duration_ns = kd.end_ns.saturating_sub(kd.start_ns);

                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(kd.start_ns)
                            .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                            .set_gpu_render_stage_event(|event: &mut GpuRenderStageEvent| {
                                event
                                    .set_event_id(get_next_event_id())
                                    .set_duration(duration_ns)
                                    .set_gpu_id(kd.gpu_id as i32)
                                    .set_hw_queue_iid(hw_queue_iid)
                                    .set_stage_iid(AMD_KERNEL_STAGE_IID)
                                    .set_context(1);
                                let extra_fields: &[(&str, String)] = &[
                                    ("kernel_name", kd.kernel_name.clone()),
                                    ("kernel_demangled_name", demangled.clone()),
                                    ("kernel_type", "Compute".to_string()),
                                    ("process_id", process_id.to_string()),
                                    ("process_name", process_name.clone()),
                                    ("device_id", kd.device_index.to_string()),
                                    ("launch__grid_size", grid_size.to_string()),
                                    ("launch__grid_size_x", kd.grid.0.to_string()),
                                    ("launch__grid_size_y", kd.grid.1.to_string()),
                                    ("launch__grid_size_z", kd.grid.2.to_string()),
                                    ("launch__block_size", workgroup_size.to_string()),
                                    ("launch__block_size_x", kd.workgroup.0.to_string()),
                                    ("launch__block_size_y", kd.workgroup.1.to_string()),
                                    ("launch__block_size_z", kd.workgroup.2.to_string()),
                                    ("launch__thread_count", thread_count.to_string()),
                                ];
                                for (name, value) in extra_fields {
                                    event.set_extra_data(
                                        |ed: &mut GpuRenderStageEventExtraData| {
                                            ed.set_name(*name);
                                            ed.set_value(value.as_str());
                                        },
                                    );
                                }
                            });
                    });
                }

                // Emit memory copy events.
                for mc in state.memcopies[mc_start..].iter() {
                    let duration_ns = mc.end_ns.saturating_sub(mc.start_ns);
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(mc.start_ns)
                            .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                            .set_gpu_render_stage_event(|event: &mut GpuRenderStageEvent| {
                                event
                                    .set_event_id(get_next_event_id())
                                    .set_duration(duration_ns)
                                    .set_gpu_id(mc.gpu_id as i32)
                                    .set_hw_queue_iid(AMD_HW_QUEUE_IID_OFFSET)
                                    .set_stage_iid(AMD_MEMCPY_STAGE_IID)
                                    .set_context(1);
                                let extra_fields: &[(&str, String)] = &[
                                    ("process_id", process_id.to_string()),
                                    ("process_name", process_name.clone()),
                                    ("device_id", mc.device_index.to_string()),
                                    ("bytes", mc.bytes.to_string()),
                                    ("direction", mc.direction.to_string()),
                                ];
                                for (name, value) in extra_fields {
                                    event.set_extra_data(
                                        |ed: &mut GpuRenderStageEventExtraData| {
                                            ed.set_name(*name);
                                            ed.set_value(value.as_str());
                                        },
                                    );
                                }
                            });
                    });
                }

                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);

            let emitted = state.kernel_dispatches.len().saturating_sub(kd_start)
                + state.memcopies.len().saturating_sub(mc_start);
            injection_log!(
                "emitted {} AMD render stage events (instance {})",
                emitted,
                inst_id
            );
        });
    }

    fn on_first_counters_start(&self) {
        // Set up per-agent counter configs and configure the dispatch counting
        // service. This is deferred to first consumer start so we don't
        // enumerate counters when only gpu.renderstages is enabled.
        use callbacks::{dispatch_counting_callback, record_counting_callback};

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
            let mut matched_ids: Vec<rocprofiler_counter_id_t> = Vec::new();
            let mut matched_names: Vec<String> = Vec::new();
            for metric in &requested_metrics {
                if let Some(&cid) = name_to_id.get(metric) {
                    matched_ids.push(cid);
                    matched_names.push(metric.clone());
                } else {
                    injection_log!(
                        "agent {:#x}: requested counter '{}' not available",
                        agent_handle,
                        metric
                    );
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
                }
            }
        }

        // Configure callback dispatch counting service on the tracing context.
        let has_counter_configs = GLOBAL_STATE
            .lock()
            .ok()
            .map(|s| !s.counter_configs.is_empty())
            .unwrap_or(false);
        if has_counter_configs {
            let context_handle = GLOBAL_STATE.lock().ok().and_then(|s| s.tracing_context);
            if let Some(handle) = context_handle {
                let tracing_ctx = rocprofiler_context_id_t { handle };
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
            let mut state = match GLOBAL_STATE.lock() {
                Ok(s) => s,
                Err(_) => return,
            };
            let start_offsets = match state.counters_consumers.remove(&inst_id) {
                Some(o) => o,
                None => return,
            };
            let counter_names = state.counter_names.clone();
            if counter_names.is_empty() {
                return;
            }
            let cr_start = start_offsets.counter_results;
            let mut stop_guard_opt = stop_guard;
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for result in state.counter_results[cr_start..].iter() {
                    let gpu_id = result.gpu_id as i32;
                    let got_first_counters =
                        GOT_FIRST_COUNTERS.fetch_or(1 << inst_id, Ordering::SeqCst);
                    if got_first_counters & (1 << inst_id) == 0 {
                        // Descriptor packet (once per instance).
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(result.start_ns)
                                .set_timestamp_clock_id(
                                    BuiltinClock::BuiltinClockBoottime.into(),
                                )
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    event.set_gpu_id(gpu_id).set_counter_descriptor(
                                        |desc: &mut GpuCounterDescriptor| {
                                            for (i, name) in counter_names.iter().enumerate() {
                                                desc.set_specs(|spec: &mut GpuCounterDescriptorGpuCounterSpec| {
                                                    spec.set_counter_id(i as u32);
                                                    spec.set_name(name);
                                                    spec.set_groups(GpuCounterDescriptorGpuCounterGroup::Compute);
                                                });
                                            }
                                        },
                                    );
                                });
                        });
                    }
                    // Zero packet at start_ns.
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(result.start_ns)
                            .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                            .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                event.set_gpu_id(gpu_id);
                                for i in 0..counter_names.len() {
                                    event.set_counters(
                                        |counter: &mut GpuCounterEventGpuCounter| {
                                            counter.set_counter_id(i as u32).set_int_value(0);
                                        },
                                    );
                                }
                            });
                    });
                    // Values packet at end_ns.
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(result.end_ns)
                            .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                            .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                event.set_gpu_id(gpu_id);
                                for (i, &value) in result.values.iter().enumerate() {
                                    event.set_counters(
                                        |counter: &mut GpuCounterEventGpuCounter| {
                                            counter
                                                .set_counter_id(i as u32)
                                                .set_double_value(value);
                                        },
                                    );
                                }
                            });
                    });
                }
                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);
            let emitted = state.counter_results.len().saturating_sub(cr_start);
            injection_log!("emitted {} counter events (instance {})", emitted, inst_id);
        });
    }
}

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
