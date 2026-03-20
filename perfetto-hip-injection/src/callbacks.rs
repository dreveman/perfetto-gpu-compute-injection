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
    AgentInfo, ApiActivity, CounterResult, KernelDispatch, MemcopyActivity, GLOBAL_STATE,
};
use std::ffi::CStr;
use std::panic;

/// Derive a 32-bit pseudo GPU ID from a 64-bit agent handle by XOR-folding.
fn agent_handle_to_gpu_id(handle: u64) -> u32 {
    ((handle >> 32) as u32) ^ (handle as u32)
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

                // Look up agent info.
                let agent_info = state.agents.get(&info.agent_id.handle);
                let device_index = agent_info.map(|a| a.device_index).unwrap_or(0);
                let arch = agent_info.map(|a| a.arch.clone()).unwrap_or_default();
                let wave_front_size = agent_info.map(|a| a.wave_front_size).unwrap_or(0);
                let cu_count = agent_info.map(|a| a.cu_count).unwrap_or(0);

                let gpu_id = agent_handle_to_gpu_id(info.agent_id.handle);

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
                    gpu_id,
                    arch,
                    wave_front_size,
                    cu_count,
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

                let gpu_id = agent_handle_to_gpu_id(rec.dst_agent_id.handle);

                state.memcopies.push(MemcopyActivity {
                    bytes: rec.bytes,
                    start_ns: rec.start_timestamp,
                    end_ns: rec.end_timestamp,
                    device_index,
                    gpu_id,
                    #[allow(clippy::unnecessary_cast)]
                    direction: rec.operation as i32,
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

                state.hip_api_activities.push(ApiActivity {
                    kind: rec.kind,
                    operation: rec.operation,
                    start: rec.start_timestamp,
                    end: rec.end_timestamp,
                    thread_id: tid,
                    correlation_id: rec.correlation_id.internal,
                });
            }
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
        let agent_handle = dispatch_data.dispatch_info.agent_id.handle;
        if let Ok(state) = GLOBAL_STATE.lock() {
            if let Some(&config_handle) = state.counter_configs.get(&agent_handle) {
                *config = rocprofiler_counter_config_id_t {
                    handle: config_handle,
                };
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

        let device_index = state
            .agents
            .get(&info.agent_id.handle)
            .map(|a| a.device_index)
            .unwrap_or(0);
        let gpu_id = agent_handle_to_gpu_id(info.agent_id.handle);

        let mut values = vec![0.0_f64; num_counters];
        for record in records {
            let mut counter_id = rocprofiler_counter_id_t { handle: 0 };
            let status = rocprofiler_query_record_counter_id(record.id, &mut counter_id);
            if status != ROCPROFILER_STATUS_SUCCESS {
                continue;
            }
            if let Some(&idx) = state.counter_id_to_index.get(&counter_id.handle) {
                values[idx] = record.counter_value;
            }
        }

        state.counter_results.push(CounterResult {
            kernel_name,
            start_ns: dispatch_data.start_timestamp,
            end_ns: dispatch_data.end_timestamp,
            device_index,
            gpu_id,
            values,
        });
    });
}
