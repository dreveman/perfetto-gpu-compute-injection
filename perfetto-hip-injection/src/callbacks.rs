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

use crate::perfetto_te_ns;
use crate::rocprofiler_sys::*;
use crate::state::{
    AgentInfo, CounterResult, KernelDispatch, MemcopyActivity, MemsetActivity, GLOBAL_STATE,
};
use perfetto_gpu_compute_injection::injection_log;
use perfetto_gpu_compute_injection::tracing::get_counter_config;
use perfetto_sdk::track_event::{
    EventContext, TrackEventProtoField, TrackEventProtoFields, TrackEventTimestamp,
    TrackEventTrack, TrackEventType,
};
use std::ffi::CStr;
use std::ffi::CString;
use std::panic;
use std::time::Duration;

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

        let mut api_events_emitted: u64 = 0;

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

                let tid = rec.thread_id;
                perfetto_gpu_compute_injection::config::capture_thread_name(
                    &mut state.thread_names,
                    tid,
                );

                let category_index = perfetto_te_ns::category_index("hip");
                if perfetto_te_ns::is_category_enabled(category_index) {
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

                    let c_name = CString::new(op_name.as_str())
                        .unwrap_or_else(|_| CString::new("unknown").unwrap());
                    let name_ptr = c_name.as_ptr();

                    let process_uuid = TrackEventTrack::process_track_uuid();
                    let process_id = libc::getpid() as u64;
                    let thread_name = state.thread_names.get(&tid);
                    perfetto_gpu_compute_injection::build_thread_track!(
                        process_uuid: process_uuid,
                        process_id: process_id,
                        thread_id: tid,
                        thread_name: thread_name.map(|s| s.as_str()),
                        => _thread_fields_named, _thread_fields_unnamed, _track_fields, thread_track
                    );

                    // SliceBegin — attach GpuCorrelation linking this API call
                    // to the corresponding render stage event via correlationId.
                    let correlation_fields = [TrackEventProtoField::VarInt(
                        1, // render_stage_submission_event_ids
                        rec.correlation_id.internal,
                    )];
                    let gpu_correlation_fields = [TrackEventProtoField::Nested(
                        3000, // gpu_correlation
                        &correlation_fields,
                    )];
                    let mut ctx = EventContext::default();
                    ctx.set_timestamp(TrackEventTimestamp::Boot(Duration::from_nanos(
                        rec.start_timestamp,
                    )));
                    ctx.set_proto_track(&thread_track);
                    ctx.set_proto_fields(&TrackEventProtoFields {
                        fields: &gpu_correlation_fields,
                    });
                    ctx.add_debug_arg(
                        "correlation_id",
                        perfetto_sdk::track_event::TrackEventDebugArg::Uint64(
                            rec.correlation_id.internal,
                        ),
                    );
                    perfetto_te_ns::emit(
                        category_index,
                        TrackEventType::SliceBegin(name_ptr),
                        &mut ctx,
                    );

                    // SliceEnd
                    let mut ctx = EventContext::default();
                    ctx.set_timestamp(TrackEventTimestamp::Boot(Duration::from_nanos(
                        rec.end_timestamp,
                    )));
                    ctx.set_proto_track(&thread_track);
                    perfetto_te_ns::emit(category_index, TrackEventType::SliceEnd, &mut ctx);
                    api_events_emitted += 1;
                }
            }
        }
        if api_events_emitted > 0 {
            injection_log!("flushed {} API track events", api_events_emitted);
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
