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

use crate::state::{KernelActivity, KernelLaunch, GLOBAL_STATE};
use crate::tracing::{is_counters_enabled, trace_time_ns};
use cupti_profiler::bindings::*;
use cupti_profiler::{self as profiler, *};
use libc::c_void;
use std::{ffi::CStr, panic, ptr};

/// Callback for CUPTI to request a buffer for storing activity records.
/// # Safety
///
/// This function is intended to be called by CUPTI. Pointers must be valid.
pub unsafe extern "C" fn buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    _max_num_records: *mut usize,
) {
    let _ = panic::catch_unwind(|| {
        *size = 16 * 1024;
        *buffer = libc::malloc(*size) as *mut u8;
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
                    let k = &*(record as *const CUpti_ActivityKernel4);
                    if let Some(data) = state.context_data.get_mut(&k.contextId) {
                        data.kernel_activities.push(KernelActivity {
                            kernel_name: CStr::from_ptr(k.name).to_string_lossy().to_string(),
                            grid_size: (k.gridX, k.gridY, k.gridZ),
                            block_size: (k.blockX, k.blockY, k.blockZ),
                            registers_per_thread: k.registersPerThread,
                            dynamic_shared_memory: k.dynamicSharedMemory,
                            static_shared_memory: k.staticSharedMemory,
                            start: k.start,
                            end: k.end,
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
        let counters_enabled = is_counters_enabled();
        if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API
            && cbid == CUpti_driver_api_trace_cbid_enum_CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
        {
            let cb_data = &*(cbdata as *const CUpti_CallbackData);
            let ctx = cb_data.context;
            let params = &*(cb_data.functionParams as *const cuLaunchKernel_params);
            if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_ENTER {
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    let metric_names = state.config.metrics.clone();
                    let active_ctx = state.active_ctx;
                    // Only manage range profiler sessions when counters are enabled
                    if counters_enabled {
                        match active_ctx {
                            Some(active_ctx) if active_ctx != ctx => {
                                let active_ctx_id = unsafe { profiler::get_context_id(active_ctx) };
                                if let Some(old_data) = state.context_data.get_mut(&active_ctx_id) {
                                    if let Some(rp) = &mut old_data.range_profiler {
                                        let _ = rp.stop();
                                        let _ = rp.decode_counter_data();
                                        if let Some(me) = &old_data.metric_evaluator {
                                            if let Ok(infos) = me.evaluate_all_ranges(
                                                &old_data.counter_data_image,
                                                &metric_names,
                                            ) {
                                                old_data.range_info.extend(infos);
                                            }
                                        }
                                        let _ = rp.disable();
                                    }
                                    old_data.range_profiler = None;
                                    old_data.is_active = false;
                                }
                                state.active_ctx = None;
                            }
                            _ => {}
                        }
                        match active_ctx {
                            Some(active_ctx) if active_ctx != ctx => {
                                let active_ctx_id = unsafe { profiler::get_context_id(active_ctx) };
                                if let Some(old_data) = state.context_data.get_mut(&active_ctx_id) {
                                    if let Some(rp) = &mut old_data.range_profiler {
                                        let _ = rp.stop();
                                        let _ = rp.decode_counter_data();
                                        if let Some(me) = &old_data.metric_evaluator {
                                            if let Ok(infos) = me.evaluate_all_ranges(
                                                &old_data.counter_data_image,
                                                &metric_names,
                                            ) {
                                                old_data.range_info.extend(infos);
                                            }
                                        }
                                        let _ = rp.disable();
                                    }
                                    old_data.range_profiler = None;
                                    old_data.is_active = false;
                                }
                                state.active_ctx = None;
                            }
                            _ => {}
                        }
                    }
                    let ctx_id = unsafe { profiler::get_context_id(ctx) };
                    if state.context_data.contains_key(&ctx_id) {
                        state.active_ctx = Some(ctx);
                        if let Some(data) = state.context_data.get_mut(&ctx_id) {
                            // Only start/stop range profiler sessions if counters are enabled
                            if counters_enabled && data.range_profiler.is_none() {
                                let mut rp = RangeProfiler::new(ctx);
                                let _ = rp.enable();
                                let _ = rp.set_config(
                                    &metric_names,
                                    &mut data.counter_data_image,
                                    data.max_num_ranges,
                                    CUpti_ProfilerReplayMode_CUPTI_KernelReplay,
                                );
                                let _ = rp.start();
                                data.range_profiler = Some(rp);
                                data.is_active = true;
                            }
                            // Record kernel launch with start timestamp
                            // When counters enabled: start is captured now, end will be set in API_EXIT
                            // When counters disabled: start/end are not used (activity timestamps are used instead)
                            let start = trace_time_ns();
                            data.kernel_launches.push(KernelLaunch {
                                function: params.f,
                                start,
                                end: 0, // Will be set in API_EXIT when profiler completes
                            });
                        }
                    }
                }
            } else if cb_data.callbackSite == CUpti_ApiCallbackSite_CUPTI_API_EXIT {
                // Handle kernel launch completion - decode profiler data and set end timestamp
                if counters_enabled {
                    if let Ok(mut state) = GLOBAL_STATE.lock() {
                        let metric_names = state.config.metrics.clone();
                        let ctx_id = unsafe { profiler::get_context_id(ctx) };
                        if let Some(data) = state.context_data.get_mut(&ctx_id) {
                            if let Some(rp) = &mut data.range_profiler {
                                let _ = rp.decode_counter_data();
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
                            // Set end timestamp on the current kernel launch
                            if let Some(last_launch) = data.kernel_launches.last_mut() {
                                if last_launch.end == 0 {
                                    last_launch.end = trace_time_ns();
                                }
                            }
                        }
                    }
                }
            }
        } else if domain == CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RESOURCE {
            if cbid == CUpti_CallbackIdResource_CUPTI_CBID_RESOURCE_CONTEXT_CREATED {
                let res_data = &*(cbdata as *const CUpti_ResourceData);
                let ctx = res_data.context;
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    let metric_names = state.config.metrics.clone();
                    // Only manage active context's range profiler when counters are enabled
                    if counters_enabled {
                        if let Some(active_ctx) = state.active_ctx {
                            let active_ctx_id = unsafe { profiler::get_context_id(active_ctx) };
                            if let Some(data) = state.context_data.get_mut(&active_ctx_id) {
                                if data.is_active {
                                    if let Some(rp) = &mut data.range_profiler {
                                        let _ = rp.stop();
                                        let _ = rp.decode_counter_data();
                                        if let Some(me) = &data.metric_evaluator {
                                            if let Ok(infos) = me.evaluate_all_ranges(
                                                &data.counter_data_image,
                                                &metric_names,
                                            ) {
                                                data.range_info.extend(infos);
                                            }
                                        }
                                        let _ = rp.disable();
                                    }
                                    data.range_profiler = None;
                                    data.is_active = false;
                                }
                            }
                            state.active_ctx = None;
                        }
                    }
                    let device_id = unsafe { profiler::get_device(ctx) }.unwrap_or(0);
                    let num_sms = profiler::get_device_attribute(
                        device_id,
                        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    )
                    .unwrap_or(0);
                    let mut data = Box::new(crate::state::CtxProfilerData {
                        device_id,
                        num_sms,
                        max_num_ranges: 10,
                        is_active: false,
                        counter_data_image: Vec::new(),
                        metric_evaluator: None,
                        range_profiler: None,
                        range_info: Vec::new(),
                        kernel_launches: Vec::new(),
                        kernel_activities: Vec::new(),
                    });
                    // Only initialize profiler and metric evaluator when counters are enabled
                    // Otherwise, just track context data for renderstages
                    if counters_enabled {
                        if Profiler::initialize().is_ok() {
                            if let Ok(me) = unsafe { MetricEvaluator::new(ctx) } {
                                data.metric_evaluator = Some(me);
                            }
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
                                let _ = rp.start();
                                data.range_profiler = Some(rp);
                                data.is_active = true;
                                state.active_ctx = Some(ctx);
                            }
                        } else {
                            eprintln!("Failed to initialize profiler");
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
                    let metric_names = state.config.metrics.clone();
                    if let Some(data) = state.context_data.get_mut(&ctx_id) {
                        if data.is_active {
                            if let Some(rp) = &mut data.range_profiler {
                                let _ = rp.stop();
                                let _ = rp.decode_counter_data();
                                if let Some(me) = &data.metric_evaluator {
                                    if let Ok(infos) = me.evaluate_all_ranges(
                                        &data.counter_data_image,
                                        &metric_names,
                                    ) {
                                        data.range_info.extend(infos);
                                    }
                                }
                                let _ = rp.disable();
                            }
                            data.range_profiler = None;
                            data.is_active = false;
                        }
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
            eprintln!("CUPTI Fatal Error: {}: {}", err_str, msg.to_string_lossy());
            std::process::exit(1);
        }
    });
}
