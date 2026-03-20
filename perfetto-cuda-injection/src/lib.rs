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

pub mod callbacks;
pub mod cupti_profiler;
pub mod cupti_profiler_sys;
pub mod metrics;
pub mod state;

use callbacks::{buffer_completed, buffer_requested, profiler_callback_handler};
use cpp_demangle::Symbol;
use metrics::parse_metrics;
use perfetto_gpu_compute_injection::config::Config;
use perfetto_gpu_compute_injection::injection_log;
use perfetto_gpu_compute_injection::tracing::{
    get_counters_data_source, get_next_event_id, get_renderstages_data_source, register_backend,
    trace_time_ns, GpuBackend, GOT_FIRST_COUNTERS, GOT_FIRST_RENDERSTAGES,
};
use perfetto_sdk::{
    data_source::{StopGuard, TraceContext},
    producer::{Backends, Producer, ProducerInitArgsBuilder},
    protos::{
        common::builtin_clock::BuiltinClock,
        trace::{
            interned_data::interned_data::InternedData,
            trace_packet::{TracePacket, TracePacketSequenceFlags},
        },
    },
    track_event::{
        EventContext, TrackEvent, TrackEventProtoField, TrackEventProtoTrack, TrackEventTimestamp,
        TrackEventTrack, TrackEventType,
    },
    track_event_categories, track_event_set_category_callback,
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
use state::{
    ConsumerStartOffsets, GpuActivity, GLOBAL_STATE, HW_QUEUE_IID_OFFSET, KERNEL_STAGE_IID,
    MEMCPY_STAGE_IID, MEMSET_STAGE_IID,
};
use std::{
    ffi::CString,
    panic, ptr,
    sync::atomic::{AtomicU8, Ordering},
    time::Duration,
};

use cupti_profiler as profiler;
use cupti_profiler::bindings::*;

// ---------------------------------------------------------------------------
// Track event categories for CUDA API call tracing
// ---------------------------------------------------------------------------

track_event_categories! {
    pub mod cuda_te_ns {
        ( "cudart", "CUDA Runtime API calls", [ "api" ] ),
        ( "cuda", "CUDA Driver API calls", [ "api" ] ),
    }
}
use cuda_te_ns as perfetto_te_ns;

// ---------------------------------------------------------------------------
// CuptiBackend implementation
// ---------------------------------------------------------------------------

/// 3-state teardown guard: 0=not started, 1=in progress, 2=done.
static CUPTI_TEARDOWN_STATE: AtomicU8 = AtomicU8::new(0);

struct CuptiBackend;

impl GpuBackend for CuptiBackend {
    fn default_data_source_suffix(&self) -> &'static str {
        "nv"
    }

    fn on_first_consumer_start(&self) {
        CUPTI_TEARDOWN_STATE.store(0, Ordering::SeqCst);
        let _ = profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        let _ = profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMCPY);
        let _ = profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMSET);
        if let Ok(state) = GLOBAL_STATE.lock() {
            let subscriber = state.subscriber_handle;
            if !subscriber.is_null() {
                let _ = unsafe {
                    profiler::enable_callback(
                        1,
                        subscriber,
                        CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUpti_runtime_api_trace_cbid_enum_CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020,
                    )
                };
            }
        }
    }

    fn on_first_counters_start(&self) {}

    fn on_last_counters_stop(&self) {}

    fn on_renderstages_start_no_counters(&self) {}

    fn register_counters_consumer(&self, inst_id: u32) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            let offsets = ConsumerStartOffsets::snapshot(&state.context_data);
            state.counters_consumers.insert(inst_id, offsets);
        }
    }

    fn register_renderstages_consumer(&self, inst_id: u32) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            let offsets = ConsumerStartOffsets::snapshot(&state.context_data);
            state.renderstages_consumers.insert(inst_id, offsets);
        }
    }

    fn flush_activity_buffers(&self) {
        let _ = profiler::activity_flush_all(CUpti_ActivityFlag_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    }

    fn finalize_range_profiler(&self) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            for (_, data) in state.context_data.iter_mut() {
                data.finalize_profiler(false);
                if let Some(last_launch) =
                    data.kernel_launches.iter_mut().rev().find(|l| l.profiled)
                {
                    if last_launch.end == 0 {
                        last_launch.end = trace_time_ns();
                    }
                }
            }
        }
    }

    fn run_teardown(&self) {
        if CUPTI_TEARDOWN_STATE
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let _ = panic::catch_unwind(|| {
                let _ = profiler::get_last_error();

                if let Ok(state) = GLOBAL_STATE.lock() {
                    let subscriber = state.subscriber_handle;
                    if !subscriber.is_null() {
                        let _ = unsafe {
                            profiler::enable_callback(
                                0,
                                subscriber,
                                CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
                                CUpti_runtime_api_trace_cbid_enum_CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020,
                            )
                        };
                    }
                }

                let _ = profiler::activity_disable(
                    CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                );
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMCPY);
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMSET);
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_RUNTIME);
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_DRIVER);
                let _ = profiler::activity_flush_all(
                    CUpti_ActivityFlag_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
                );

                self.finalize_range_profiler();
            });
            CUPTI_TEARDOWN_STATE.store(2, Ordering::SeqCst);
        } else {
            while CUPTI_TEARDOWN_STATE.load(Ordering::SeqCst) != 2 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
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
            let mut stop_guard_opt = stop_guard;
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for (ctx_id, data) in state.context_data.iter() {
                    let range_start = start_offsets.range_info.get(ctx_id).copied().unwrap_or(0);
                    let launch_start =
                        start_offsets.kernel_launches.get(ctx_id).copied().unwrap_or(0);
                    let activity_start =
                        start_offsets.kernel_activities.get(ctx_id).copied().unwrap_or(0);
                    for ((range, launch), activity) in data.range_info[range_start..]
                        .iter()
                        .zip(data.kernel_launches[launch_start..].iter())
                        .zip(data.kernel_activities[activity_start..].iter())
                    {
                        let gpu_id = activity.gpu_id() as i32;
                        let got_first_counters =
                            GOT_FIRST_COUNTERS.fetch_or(1 << inst_id, Ordering::SeqCst);
                        if got_first_counters & (1 << inst_id) == 0 {
                            ctx.add_packet(|packet: &mut TracePacket| {
                                packet
                                    .set_timestamp(launch.start)
                                    .set_timestamp_clock_id(
                                        BuiltinClock::BuiltinClockBoottime.into(),
                                    )
                                    .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                        event.set_gpu_id(gpu_id).set_counter_descriptor(
                                            |desc: &mut GpuCounterDescriptor| {
                                                for (i, metric) in
                                                    range.metric_and_values.iter().enumerate()
                                                {
                                                    desc.set_specs(|desc: &mut GpuCounterDescriptorGpuCounterSpec| {
                                                        desc.set_counter_id(i as u32);
                                                        desc.set_name(&metric.metric_name);
                                                        desc.set_groups(GpuCounterDescriptorGpuCounterGroup::Compute);
                                                    });
                                                }
                                            },
                                        );
                                    });
                            });
                        }
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(launch.start)
                                .set_timestamp_clock_id(
                                    BuiltinClock::BuiltinClockBoottime.into(),
                                )
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    event.set_gpu_id(gpu_id);
                                    for (i, _metric) in range.metric_and_values.iter().enumerate()
                                    {
                                        event.set_counters(
                                            |counter: &mut GpuCounterEventGpuCounter| {
                                                counter.set_counter_id(i as u32).set_int_value(0);
                                            },
                                        );
                                    }
                                });
                        });
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(launch.end)
                                .set_timestamp_clock_id(
                                    BuiltinClock::BuiltinClockBoottime.into(),
                                )
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    event.set_gpu_id(gpu_id);
                                    for (i, metric) in range.metric_and_values.iter().enumerate()
                                    {
                                        event.set_counters(
                                            |counter: &mut GpuCounterEventGpuCounter| {
                                                counter
                                                    .set_counter_id(i as u32)
                                                    .set_double_value(metric.value);
                                            },
                                        );
                                    }
                                });
                        });
                    }
                }
                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);
            let emitted: usize = state
                .context_data
                .iter()
                .map(|(ctx_id, data)| {
                    let start = start_offsets.range_info.get(ctx_id).copied().unwrap_or(0);
                    data.range_info.len().saturating_sub(start)
                })
                .sum();
            injection_log!("emitted {} counter events (instance {})", emitted, inst_id);
        });
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
            let mut stop_guard_opt = stop_guard;
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                let mut channels: std::collections::HashSet<(u32, u32)> =
                    std::collections::HashSet::new();
                let mut contexts: std::collections::HashSet<u32> =
                    std::collections::HashSet::new();
                for (ctx_id, data) in state.context_data.iter() {
                    let ka_start =
                        start_offsets.kernel_activities.get(ctx_id).copied().unwrap_or(0);
                    let mc_start =
                        start_offsets.memcpy_activities.get(ctx_id).copied().unwrap_or(0);
                    let ms_start =
                        start_offsets.memset_activities.get(ctx_id).copied().unwrap_or(0);
                    for activity in data.kernel_activities[ka_start..].iter() {
                        channels.insert((activity.channel_id, activity.channel_type));
                        contexts.insert(activity.context_id);
                    }
                    for activity in data.memcpy_activities[mc_start..].iter() {
                        channels.insert((activity.channel_id, activity.channel_type));
                        contexts.insert(activity.context_id);
                    }
                    for activity in data.memset_activities[ms_start..].iter() {
                        channels.insert((activity.channel_id, activity.channel_type));
                        contexts.insert(activity.context_id);
                    }
                }
                for (ctx_id, data) in state.context_data.iter() {
                    let launch_start =
                        start_offsets.kernel_launches.get(ctx_id).copied().unwrap_or(0);
                    let ka_start =
                        start_offsets.kernel_activities.get(ctx_id).copied().unwrap_or(0);
                    for (launch, activity) in data.kernel_launches[launch_start..]
                        .iter()
                        .zip(data.kernel_activities[ka_start..].iter())
                    {
                        let (timestamp, duration_ns) = if launch.profiled {
                            (launch.start, launch.end.saturating_sub(launch.start))
                        } else {
                            (activity.start, activity.end.saturating_sub(activity.start))
                        };
                        let demangled = if let Ok(sym) = Symbol::new(&activity.kernel_name) {
                            sym.demangle()
                                .map(|d| d.to_string())
                                .unwrap_or(activity.kernel_name.clone())
                        } else {
                            activity.kernel_name.clone()
                        };
                        let grid_size =
                            activity.grid_size.0 * activity.grid_size.1 * activity.grid_size.2;
                        let block_size =
                            activity.block_size.0 * activity.block_size.1 * activity.block_size.2;
                        let thread_count = grid_size * block_size;
                        let mut cache_mode = 0;
                        let _ = unsafe {
                            profiler::get_func_attribute(
                                launch.function,
                                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                            )
                        }
                        .map(|v| cache_mode = v);
                        let max_active_blocks = unsafe {
                            profiler::occupancy_max_active_blocks_per_multiprocessor(
                                launch.function,
                                block_size,
                                activity.dynamic_shared_memory as usize,
                            )
                        }
                        .unwrap_or(0);
                        let waves_per_multiprocessor =
                            if data.num_sms > 0 && max_active_blocks > 0 {
                                grid_size as f64 / (data.num_sms * max_active_blocks) as f64
                            } else {
                                0.0
                            };
                        let regs_per_thread = unsafe {
                            profiler::get_func_attribute(
                                launch.function,
                                CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS,
                            )
                        }
                        .unwrap_or(0);
                        let smem_per_block =
                            activity.dynamic_shared_memory + activity.static_shared_memory;
                        let warp_size = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                        )
                        .unwrap_or(32);
                        let max_threads_sm = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                        )
                        .unwrap_or(0);
                        let max_blocks_sm = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                        )
                        .unwrap_or(0);
                        let regs_per_sm = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                        )
                        .unwrap_or(0);
                        let smem_per_sm = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                        )
                        .unwrap_or(0);
                        let major = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        )
                        .unwrap_or(0);
                        let minor = profiler::get_device_attribute(
                            data.device_id,
                            CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        )
                        .unwrap_or(0);
                        let warps_per_block =
                            if warp_size > 0 { block_size / warp_size } else { 0 };
                        let max_active_warps = max_active_blocks * warps_per_block;
                        let regs_per_block = (regs_per_thread as i32) * block_size;
                        let max_warps_sm =
                            if warp_size > 0 { max_threads_sm / warp_size } else { 0 };
                        let max_active_warps_pct = if max_warps_sm > 0 {
                            100.0 * max_active_warps as f64 / max_warps_sm as f64
                        } else {
                            0.0
                        };
                        let occupancy_limit_shared_mem = if smem_per_block != 0 {
                            smem_per_sm / smem_per_block
                        } else {
                            16
                        };
                        let occupancy_limit_warps =
                            if warps_per_block > 0 { max_warps_sm / warps_per_block } else { 0 };
                        let occupancy_limit_registers =
                            if regs_per_block != 0 { regs_per_sm / regs_per_block } else { 16 };

                        let extra_data = |emit: &mut dyn FnMut(&str, &str)| {
                            emit("kernel_name", &activity.kernel_name);
                            emit("kernel_demangled_name", &demangled);
                            emit("kernel_type", "Compute");
                            emit("process_id", &process_id.to_string());
                            emit("process_name", &process_name);
                            emit("device_id", &activity.device_id.to_string());
                            emit("device_uuid", &activity.device_uuid_string());
                            emit("context_id", &activity.context_id.to_string());
                            emit("stream_id", &activity.stream_id.to_string());
                            emit("channel_id", &activity.channel_id.to_string());
                            emit("channel_type", &activity.channel_type.to_string());
                            emit("arch", &format!("CC_{}{}", major, minor));
                            #[allow(nonstandard_style)]
                            match cache_mode as u32 {
                                CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_NONE => {
                                    emit("launch__func_cache_config", "CachePreferNone")
                                }
                                CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_SHARED => {
                                    emit("launch__func_cache_config", "CachePreferShared")
                                }
                                CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_L1 => {
                                    emit("launch__func_cache_config", "CachePreferL1")
                                }
                                CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_EQUAL => {
                                    emit("launch__func_cache_config", "CachePreferEqual")
                                }
                                _ => emit("launch__func_cache_config", "n/a"),
                            }
                            emit(
                                "launch__waves_per_multiprocessor",
                                &waves_per_multiprocessor.to_string(),
                            );
                            emit("launch__grid_size", &grid_size.to_string());
                            emit("launch__grid_size_x", &activity.grid_size.0.to_string());
                            emit("launch__grid_size_y", &activity.grid_size.1.to_string());
                            emit("launch__grid_size_z", &activity.grid_size.2.to_string());
                            emit("launch__block_size", &block_size.to_string());
                            emit("launch__block_size_x", &activity.block_size.0.to_string());
                            emit("launch__block_size_y", &activity.block_size.1.to_string());
                            emit("launch__block_size_z", &activity.block_size.2.to_string());
                            emit("launch__thread_count", &thread_count.to_string());
                            emit(
                                "launch__registers_per_thread",
                                &activity.registers_per_thread.to_string(),
                            );
                            emit("launch__shared_mem_config_size", "49152");
                            emit(
                                "launch__shared_mem_per_block_driver",
                                &smem_per_block.to_string(),
                            );
                            emit(
                                "launch__shared_mem_per_block_dynamic",
                                &activity.dynamic_shared_memory.to_string(),
                            );
                            emit(
                                "launch__shared_mem_per_block_static",
                                &activity.static_shared_memory.to_string(),
                            );
                            emit(
                                "launch__occupancy_limit_shared_mem",
                                &occupancy_limit_shared_mem.to_string(),
                            );
                            emit(
                                "launch__occupancy_limit_warps",
                                &occupancy_limit_warps.to_string(),
                            );
                            emit("launch__occupancy_limit_blocks", &max_blocks_sm.to_string());
                            emit(
                                "launch__occupancy_limit_registers",
                                &occupancy_limit_registers.to_string(),
                            );
                            emit(
                                "sm__maximum_warps_avg_per_active_cycle",
                                &max_active_warps.to_string(),
                            );
                            emit(
                                "sm__maximum_warps_per_active_cycle_pct",
                                &max_active_warps_pct.to_string(),
                            );
                        };
                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        extra_data(&mut |name: &str, value: &str| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        });

                        let got_first_renderstages =
                            GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                        ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                            let was_cleared =
                                std::mem::replace(&mut inc_state.was_cleared, false);
                            let emit_interned =
                                was_cleared || got_first_renderstages & (1 << inst_id) == 0;
                            let rs_ctx = RenderStageContext {
                                channels: &channels,
                                contexts: &contexts,
                                process_id,
                            };
                            emit_render_stage_event(
                                ctx,
                                activity,
                                timestamp,
                                duration_ns,
                                emit_interned,
                                &rs_ctx,
                                &extra_data_vec,
                            );
                        });
                    }
                    let mc_start =
                        start_offsets.memcpy_activities.get(ctx_id).copied().unwrap_or(0);
                    for activity in data.memcpy_activities[mc_start..].iter() {
                        let timestamp = activity.start;
                        let duration_ns = activity.end.saturating_sub(activity.start);
                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        activity.emit_extra_data(process_id, &process_name, &mut |name, value| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        });
                        let got_first_renderstages =
                            GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                        ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                            let was_cleared =
                                std::mem::replace(&mut inc_state.was_cleared, false);
                            let emit_interned =
                                was_cleared || got_first_renderstages & (1 << inst_id) == 0;
                            let rs_ctx = RenderStageContext {
                                channels: &channels,
                                contexts: &contexts,
                                process_id,
                            };
                            emit_render_stage_event(
                                ctx,
                                activity,
                                timestamp,
                                duration_ns,
                                emit_interned,
                                &rs_ctx,
                                &extra_data_vec,
                            );
                        });
                    }
                    let ms_start =
                        start_offsets.memset_activities.get(ctx_id).copied().unwrap_or(0);
                    for activity in data.memset_activities[ms_start..].iter() {
                        let timestamp = activity.start;
                        let duration_ns = activity.end.saturating_sub(activity.start);
                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        activity.emit_extra_data(process_id, &process_name, &mut |name, value| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        });
                        let got_first_renderstages =
                            GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                        ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                            let was_cleared =
                                std::mem::replace(&mut inc_state.was_cleared, false);
                            let emit_interned =
                                was_cleared || got_first_renderstages & (1 << inst_id) == 0;
                            let rs_ctx = RenderStageContext {
                                channels: &channels,
                                contexts: &contexts,
                                process_id,
                            };
                            emit_render_stage_event(
                                ctx,
                                activity,
                                timestamp,
                                duration_ns,
                                emit_interned,
                                &rs_ctx,
                                &extra_data_vec,
                            );
                        });
                    }
                }
                let mut sg = Some(stop_guard_opt.take());
                ctx.flush(move || drop(sg.take()));
            });
            drop(stop_guard_opt);
            let emitted: usize = state
                .context_data
                .iter()
                .map(|(ctx_id, data)| {
                    let ka = start_offsets
                        .kernel_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let mc = start_offsets
                        .memcpy_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let ms = start_offsets
                        .memset_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    data.kernel_activities.len().saturating_sub(ka)
                        + data.memcpy_activities.len().saturating_sub(mc)
                        + data.memset_activities.len().saturating_sub(ms)
                })
                .sum();
            injection_log!(
                "emitted {} render stage events (instance {})",
                emitted,
                inst_id
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Render stage helpers
// ---------------------------------------------------------------------------

fn emit_interned_specifications(
    packet: &mut TracePacket,
    channels: &std::collections::HashSet<(u32, u32)>,
    contexts: &std::collections::HashSet<u32>,
    process_id: i32,
) {
    packet.set_sequence_flags(TracePacketSequenceFlags::SeqIncrementalStateCleared as u32);
    packet.set_interned_data(|interned: &mut InternedData| {
        for context_id in contexts {
            interned.set_graphics_contexts(|ctx: &mut InternedGraphicsContext| {
                ctx.set_iid(*context_id as u64);
                ctx.set_pid(process_id);
                ctx.set_api(InternedGraphicsContextApi::Undefined);
            });
        }
        for (channel_id, channel_type) in channels {
            let queue_iid = *channel_id as u64 + HW_QUEUE_IID_OFFSET;
            let queue_category = match *channel_type {
                1 => InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                _ => InternedGpuRenderStageSpecificationRenderStageCategory::Other,
            };
            interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
                spec.set_iid(queue_iid);
                spec.set_name(format!("Channel ({})", channel_id));
                spec.set_category(queue_category);
            });
        }
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(KERNEL_STAGE_IID);
            spec.set_name("Kernel");
            spec.set_description("CUDA Kernel");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Compute);
        });
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(MEMCPY_STAGE_IID);
            spec.set_name("MemoryTransfer");
            spec.set_description("CUDA Memory Transfer");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Other);
        });
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(MEMSET_STAGE_IID);
            spec.set_name("MemorySet");
            spec.set_description("CUDA Memory Set");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Other);
        });
    });
}

struct RenderStageContext<'a> {
    channels: &'a std::collections::HashSet<(u32, u32)>,
    contexts: &'a std::collections::HashSet<u32>,
    process_id: i32,
}

fn emit_render_stage_event<T: GpuActivity>(
    ctx: &mut TraceContext,
    activity: &T,
    timestamp: u64,
    duration_ns: u64,
    emit_interned: bool,
    rs_ctx: &RenderStageContext,
    extra_data: &[(String, String)],
) {
    let hw_queue_iid = activity.channel_id() as u64 + HW_QUEUE_IID_OFFSET;
    let context_iid = activity.context_id() as u64;
    let gpu_id = activity.gpu_id() as i32;
    let stage_iid = activity.stage_iid();

    ctx.add_packet(|packet: &mut TracePacket| {
        packet
            .set_timestamp(timestamp)
            .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
            .set_gpu_render_stage_event(|event: &mut GpuRenderStageEvent| {
                event
                    .set_event_id(get_next_event_id())
                    .set_duration(duration_ns)
                    .set_gpu_id(gpu_id)
                    .set_hw_queue_iid(hw_queue_iid)
                    .set_stage_iid(stage_iid)
                    .set_context(context_iid);
                for (name, value) in extra_data {
                    event.set_extra_data(|extra_data: &mut GpuRenderStageEventExtraData| {
                        extra_data.set_name(name);
                        extra_data.set_value(value);
                    });
                }
            });
        if emit_interned {
            emit_interned_specifications(
                packet,
                rs_ctx.channels,
                rs_ctx.contexts,
                rs_ctx.process_id,
            );
        }
    });
}

// ---------------------------------------------------------------------------
// API call track event emission
// ---------------------------------------------------------------------------

fn emit_api_activities() {
    let (runtime_activities, driver_activities, thread_names) = {
        let mut state = match GLOBAL_STATE.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        let runtime = std::mem::take(&mut state.runtime_api_activities);
        let driver = std::mem::take(&mut state.driver_api_activities);
        let names = std::mem::take(&mut state.thread_names);
        (runtime, driver, names)
    };

    // Emit runtime API activities under "cudart" category
    emit_api_category_activities(
        &runtime_activities,
        CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
        perfetto_te_ns::category_index("cudart"),
        &thread_names,
    );

    // Emit driver API activities under "cuda" category
    emit_api_category_activities(
        &driver_activities,
        CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API,
        perfetto_te_ns::category_index("cuda"),
        &thread_names,
    );

    let total = runtime_activities.len() + driver_activities.len();
    if total > 0 {
        injection_log!(
            "emitted {} API call track events ({} runtime, {} driver)",
            total,
            runtime_activities.len(),
            driver_activities.len()
        );
    }
}

fn emit_api_category_activities(
    activities: &[state::ApiActivity],
    domain: CUpti_CallbackDomain,
    category_index: usize,
    thread_names: &std::collections::HashMap<u32, String>,
) {
    if activities.is_empty() || !perfetto_te_ns::is_category_enabled(category_index) {
        return;
    }

    let process_uuid = TrackEventTrack::process_track_uuid();

    // Sort by (thread_id, start) for proper nesting per thread
    let mut sorted: Vec<&state::ApiActivity> = activities.iter().collect();
    sorted.sort_by_key(|a| (a.thread_id, a.start));

    for activity in &sorted {
        let full_name = profiler::get_callback_name(domain, activity.cbid);
        // Strip version suffix (e.g. "_v7000") from the name, add as debug annotation
        let (base_name, version) = match full_name.rfind("_v") {
            Some(pos) if full_name[pos + 2..].chars().all(|c| c.is_ascii_digit()) => {
                (&full_name[..pos], Some(&full_name[pos + 2..]))
            }
            _ => (full_name.as_str(), None),
        };
        let c_name = CString::new(base_name).unwrap_or_else(|_| CString::new("unknown").unwrap());
        let name_ptr = c_name.as_ptr();

        // Target the correct thread track using Perfetto's UUID formula:
        // process_track_uuid ^ gettid(). Include parent_uuid and thread
        // descriptor so the descriptor is consistent with any SDK-emitted
        // descriptor for the same thread.
        let tname = thread_names.get(&activity.thread_id);
        let thread_fields_with_name;
        let thread_fields_without_name;
        let thread_fields: &[TrackEventProtoField] = if let Some(name) = tname {
            thread_fields_with_name = [
                TrackEventProtoField::VarInt(1, activity.process_id as u64), // pid
                TrackEventProtoField::VarInt(2, activity.thread_id as u64),  // tid
                TrackEventProtoField::Cstr(5, name),                         // thread_name
            ];
            &thread_fields_with_name
        } else {
            thread_fields_without_name = [
                TrackEventProtoField::VarInt(1, activity.process_id as u64), // pid
                TrackEventProtoField::VarInt(2, activity.thread_id as u64),  // tid
            ];
            &thread_fields_without_name
        };
        let track_fields = [
            TrackEventProtoField::VarInt(5, process_uuid), // parent_uuid
            TrackEventProtoField::Nested(4, thread_fields), // thread
        ];
        let thread_track = TrackEventProtoTrack {
            uuid: process_uuid ^ activity.thread_id as u64,
            fields: &track_fields,
        };

        // SliceBegin
        let mut ctx = EventContext::default();
        ctx.set_timestamp(TrackEventTimestamp::Boot(Duration::from_nanos(
            activity.start,
        )));
        ctx.set_proto_track(&thread_track);
        ctx.add_debug_arg(
            "correlation_id",
            perfetto_sdk::track_event::TrackEventDebugArg::Uint64(activity.correlation_id as u64),
        );
        if let Some(ver) = version {
            ctx.add_debug_arg(
                "version",
                perfetto_sdk::track_event::TrackEventDebugArg::String(ver),
            );
        }
        perfetto_te_ns::emit(
            category_index,
            TrackEventType::SliceBegin(name_ptr),
            &mut ctx,
        );

        // SliceEnd
        let mut ctx = EventContext::default();
        ctx.set_timestamp(TrackEventTimestamp::Boot(Duration::from_nanos(
            activity.end,
        )));
        ctx.set_proto_track(&thread_track);
        perfetto_te_ns::emit(category_index, TrackEventType::SliceEnd, &mut ctx);
    }
}

// ---------------------------------------------------------------------------
// Atexit fallback
// ---------------------------------------------------------------------------

extern "C" fn end_execution() {
    let _ = panic::catch_unwind(|| {
        let nvidia = CuptiBackend;
        nvidia.run_teardown();
        let counter_ids: Vec<u32> = GLOBAL_STATE
            .lock()
            .map(|s| s.counters_consumers.keys().copied().collect())
            .unwrap_or_default();
        let renderstage_ids: Vec<u32> = GLOBAL_STATE
            .lock()
            .map(|s| s.renderstages_consumers.keys().copied().collect())
            .unwrap_or_default();
        for inst_id in counter_ids {
            nvidia.emit_counter_events_for_instance(inst_id, None);
        }
        for inst_id in renderstage_ids {
            nvidia.emit_renderstage_events_for_instance(inst_id, None);
        }
        emit_api_activities();
    });
}

unsafe extern "C" fn activity_timestamp_callback() -> u64 {
    perfetto_gpu_compute_injection::tracing::trace_time_ns()
}

fn register_profiler_callbacks() -> Result<CUpti_SubscriberHandle, CUptiResult> {
    unsafe { profiler::activity_register_timestamp_callback(Some(activity_timestamp_callback)) }?;

    let subscriber =
        unsafe { profiler::subscribe(Some(profiler_callback_handler), ptr::null_mut()) }?;

    unsafe {
        profiler::enable_callback(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API,
            CUpti_driver_api_trace_cbid_enum_CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
        )
    }?;
    unsafe {
        profiler::enable_callback(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RESOURCE,
            CUpti_CallbackIdResource_CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
        )
    }?;
    unsafe {
        profiler::enable_callback(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RESOURCE,
            CUpti_CallbackIdResource_CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
        )
    }?;
    unsafe { profiler::enable_domain(1, subscriber, CUpti_CallbackDomain_CUPTI_CB_DOMAIN_STATE) }?;
    unsafe {
        profiler::enable_callback(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
            CUpti_runtime_api_trace_cbid_enum_CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020,
        )
    }?;
    unsafe {
        profiler::activity_register_callbacks(Some(buffer_requested), Some(buffer_completed))
    }?;

    profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)?;
    profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMCPY)?;
    profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_MEMSET)?;

    unsafe { libc::atexit(end_execution) };

    Ok(subscriber)
}

// ---------------------------------------------------------------------------
// Public C entry point
// ---------------------------------------------------------------------------

/// NVIDIA injection entry point (called via CUDA_INJECTION64_PATH mechanism).
#[no_mangle]
pub extern "C" fn InitializeInjection() -> i32 {
    let result = panic::catch_unwind(|| {
        let mut config = Config::from_env();

        // Log CUPTI version for debugging compatibility issues
        let mut cupti_version: u32 = 0;
        unsafe { profiler::bindings::cuptiGetVersion(&mut cupti_version) };
        injection_log!("CUPTI version: {}", cupti_version);

        // Use system thread IDs (gettid on Linux) so that activity record
        // threadId values match Perfetto's thread track UUIDs.
        let _ = unsafe {
            cuptiSetThreadIdType(CUpti_ActivityThreadIdType_CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM)
        };

        let metrics_str = std::env::var("INJECTION_METRICS").unwrap_or_default();
        config.metrics = parse_metrics(&metrics_str);

        register_backend(CuptiBackend);

        let producer_args = ProducerInitArgsBuilder::new().backends(Backends::SYSTEM);
        Producer::init(producer_args.build());
        let _ = get_counters_data_source();
        let _ = get_renderstages_data_source();

        // Initialize track event categories for API call tracing
        TrackEvent::init();
        let _ = perfetto_te_ns::register();

        // Enable/disable CUPTI activity kinds when categories are toggled
        track_event_set_category_callback!("cudart", |_inst_id, enabled, _changed| {
            if enabled {
                let _ = profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_RUNTIME);
            } else {
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_RUNTIME);
            }
        });
        track_event_set_category_callback!("cuda", |_inst_id, enabled, _changed| {
            if enabled {
                let _ = profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_DRIVER);
            } else {
                let _ = profiler::activity_disable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_DRIVER);
            }
        });

        if let Ok(mut state) = GLOBAL_STATE.lock() {
            if !state.injection_initialized {
                state.injection_initialized = true;
                state.config = config;

                match register_profiler_callbacks() {
                    Ok(subscriber) => {
                        state.subscriber_handle = subscriber;
                    }
                    Err(e) => {
                        injection_log!("Failed to register callbacks: {:?}", e);
                        return 0;
                    }
                }
            }
        }
        1
    });
    result.unwrap_or(0)
}
