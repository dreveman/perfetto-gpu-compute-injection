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
    track_event::TrackEvent,
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
    panic, ptr,
    sync::atomic::{AtomicU8, Ordering},
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
        // cuptiActivityFlushAll deadlocks when called concurrently from
        // multiple threads.  Use try_lock so that only one thread performs
        // the actual CUPTI flush; the other skips it (the first call already
        // delivers all pending records via the buffer_completed callback).
        static FLUSH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        if let Ok(_guard) = FLUSH_LOCK.try_lock() {
            let _ =
                profiler::activity_flush_all(CUpti_ActivityFlag_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        }
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
            // Collected counter event data.
            struct CollectedCounterEvent {
                timestamp_start: u64,
                timestamp_end: u64,
                gpu_id: i32,
                metrics: Vec<(String, f64)>,
            }

            // Phase 1: Collect data under GLOBAL_STATE lock, then release.
            let collected_events = {
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
                let mut events: Vec<CollectedCounterEvent> = Vec::new();
                for (ctx_id, data) in state.context_data.iter() {
                    let range_start = start_offsets.range_info.get(ctx_id).copied().unwrap_or(0);
                    let launch_start = start_offsets
                        .kernel_launches
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let activity_start = start_offsets
                        .kernel_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for ((range, launch), activity) in data.range_info[range_start..]
                        .iter()
                        .zip(data.kernel_launches[launch_start..].iter())
                        .zip(data.kernel_activities[activity_start..].iter())
                    {
                        if launch.end == 0 {
                            continue;
                        }
                        events.push(CollectedCounterEvent {
                            timestamp_start: launch.start,
                            timestamp_end: launch.end,
                            gpu_id: activity.gpu_id() as i32,
                            metrics: range
                                .metric_and_values
                                .iter()
                                .map(|m| (m.metric_name.clone(), m.value))
                                .collect(),
                        });
                    }
                }
                events
                // state (GLOBAL_STATE lock) dropped here
            };

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_completed callback.
            let mut stop_guard_opt = stop_guard;
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for event in &collected_events {
                    let got_first_counters =
                        GOT_FIRST_COUNTERS.fetch_or(1 << inst_id, Ordering::SeqCst);
                    if got_first_counters & (1 << inst_id) == 0 {
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(event.timestamp_start)
                                .set_timestamp_clock_id(
                                    BuiltinClock::BuiltinClockBoottime.into(),
                                )
                                .set_gpu_counter_event(|ce: &mut GpuCounterEvent| {
                                    ce.set_gpu_id(event.gpu_id).set_counter_descriptor(
                                        |desc: &mut GpuCounterDescriptor| {
                                            for (i, (metric_name, _)) in
                                                event.metrics.iter().enumerate()
                                            {
                                                desc.set_specs(|spec: &mut GpuCounterDescriptorGpuCounterSpec| {
                                                    spec.set_counter_id(i as u32);
                                                    spec.set_name(metric_name);
                                                    spec.set_groups(GpuCounterDescriptorGpuCounterGroup::Compute);
                                                });
                                            }
                                        },
                                    );
                                });
                        });
                    }
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(event.timestamp_start)
                            .set_timestamp_clock_id(
                                BuiltinClock::BuiltinClockBoottime.into(),
                            )
                            .set_gpu_counter_event(|ce: &mut GpuCounterEvent| {
                                ce.set_gpu_id(event.gpu_id);
                                for (i, _) in event.metrics.iter().enumerate() {
                                    ce.set_counters(
                                        |counter: &mut GpuCounterEventGpuCounter| {
                                            counter.set_counter_id(i as u32).set_int_value(0);
                                        },
                                    );
                                }
                            });
                    });
                    ctx.add_packet(|packet: &mut TracePacket| {
                        packet
                            .set_timestamp(event.timestamp_end)
                            .set_timestamp_clock_id(
                                BuiltinClock::BuiltinClockBoottime.into(),
                            )
                            .set_gpu_counter_event(|ce: &mut GpuCounterEvent| {
                                ce.set_gpu_id(event.gpu_id);
                                for (i, (_, value)) in event.metrics.iter().enumerate() {
                                    ce.set_counters(
                                        |counter: &mut GpuCounterEventGpuCounter| {
                                            counter
                                                .set_counter_id(i as u32)
                                                .set_double_value(*value);
                                        },
                                    );
                                }
                            });
                    });
                }
                if let Some(sg) = stop_guard_opt.take() {
                    let mut sg = Some(Some(sg));
                    ctx.flush(move || drop(sg.take()));
                }
            });
            drop(stop_guard_opt);
        });
    }

    fn emit_renderstage_events_for_instance(&self, inst_id: u32, stop_guard: Option<StopGuard>) {
        let _ = panic::catch_unwind(|| {
            let (process_id, process_name) =
                perfetto_gpu_compute_injection::config::get_process_info();

            // PendingEvent holds all data needed to emit a render stage event.
            struct PendingEvent {
                start: u64,
                end: u64,
                name: String,
                extra_data: Vec<(String, String)>,
                channel_id: u32,
                channel_type: u32,
                activity_device_id: u32,
                activity_context_id: u32,
                activity_stream_id: u32,
                stage_iid: u64,
            }

            impl state::GpuActivity for PendingEvent {
                fn start(&self) -> u64 {
                    self.start
                }
                fn end(&self) -> u64 {
                    self.end
                }
                fn device_id(&self) -> u32 {
                    self.activity_device_id
                }
                fn context_id(&self) -> u32 {
                    self.activity_context_id
                }
                fn stream_id(&self) -> u32 {
                    self.activity_stream_id
                }
                fn channel_id(&self) -> u32 {
                    self.channel_id
                }
                fn channel_type(&self) -> u32 {
                    self.channel_type
                }
                fn stage_iid(&self) -> u64 {
                    self.stage_iid
                }
                fn emit_extra_data(
                    &self,
                    _process_id: i32,
                    _process_name: &str,
                    _emit: &mut dyn FnMut(&str, &str),
                ) {
                }
            }

            // Phase 1: Collect all event data under GLOBAL_STATE lock, then release.
            let (all_events, channels, contexts, updated_prev_ends, is_flush) = {
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
                let is_flush = stop_guard.is_none();

                // Build channel and context sets.
                let mut channels: std::collections::HashSet<(u32, u32)> =
                    std::collections::HashSet::new();
                let mut contexts: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for (ctx_id, data) in state.context_data.iter() {
                    let ka_start = start_offsets
                        .kernel_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let mc_start = start_offsets
                        .memcpy_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let ms_start = start_offsets
                        .memset_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
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

                // Build per-channel event lists.
                let mut channel_events: std::collections::HashMap<(u32, u32), Vec<PendingEvent>> =
                    std::collections::HashMap::new();

                for (ctx_id, data) in state.context_data.iter() {
                    let launch_start = start_offsets
                        .kernel_launches
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let ka_start = start_offsets
                        .kernel_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for (launch, activity) in data.kernel_launches[launch_start..]
                        .iter()
                        .zip(data.kernel_activities[ka_start..].iter())
                    {
                        let (raw_start, raw_end) = if launch.profiled {
                            (launch.start, launch.end)
                        } else {
                            (activity.start, activity.end)
                        };
                        let demangled = perfetto_gpu_compute_injection::kernel::demangle_name(
                            &activity.kernel_name,
                        );
                        let grid_size =
                            activity.grid_size.0 * activity.grid_size.1 * activity.grid_size.2;
                        let block_size =
                            activity.block_size.0 * activity.block_size.1 * activity.block_size.2;
                        let thread_count = grid_size * block_size;
                        let cache_mode = launch.cache_mode;
                        let max_active_blocks = launch.max_active_blocks_per_sm;
                        let waves_per_multiprocessor = if data.num_sms > 0 && max_active_blocks > 0
                        {
                            grid_size as f64 / (data.num_sms * max_active_blocks) as f64
                        } else {
                            0.0
                        };
                        let regs_per_thread = activity.registers_per_thread as i32;
                        let smem_per_block =
                            activity.dynamic_shared_memory + activity.static_shared_memory;
                        let warp_size = data.warp_size;
                        let max_threads_sm = data.max_threads_per_sm;
                        let max_blocks_sm = data.max_blocks_per_sm;
                        let regs_per_sm = data.max_regs_per_sm;
                        let smem_per_sm = data.max_smem_per_sm;
                        let major = data.compute_capability.0;
                        let minor = data.compute_capability.1;
                        let warps_per_block = if warp_size > 0 {
                            block_size / warp_size
                        } else {
                            0
                        };
                        let max_active_warps = max_active_blocks * warps_per_block;
                        let regs_per_block = regs_per_thread * block_size;
                        let max_warps_sm = if warp_size > 0 {
                            max_threads_sm / warp_size
                        } else {
                            0
                        };
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
                        let occupancy_limit_warps = if warps_per_block > 0 {
                            max_warps_sm / warps_per_block
                        } else {
                            0
                        };
                        let occupancy_limit_registers = if regs_per_block != 0 {
                            regs_per_sm / regs_per_block
                        } else {
                            16
                        };

                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        let emit = &mut |name: &str, value: &str| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        };
                        emit("kernel_name", &activity.kernel_name);
                        emit("kernel_demangled_name", &demangled);
                        emit("kernel_type", "Compute");
                        emit("process_id", &process_id.to_string());
                        emit("process_name", &process_name);
                        emit("device_id", &activity.device_id.to_string());
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

                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: raw_start,
                                end: raw_end,
                                name: perfetto_gpu_compute_injection::kernel::simplify_name(
                                    &demangled,
                                )
                                .to_string(),
                                extra_data: extra_data_vec,
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::KERNEL_STAGE_IID,
                            });
                    }
                    let mc_start = start_offsets
                        .memcpy_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for activity in data.memcpy_activities[mc_start..].iter() {
                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        activity.emit_extra_data(process_id, &process_name, &mut |name, value| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        });
                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: activity.start,
                                end: activity.end,
                                name: format!("Memcpy {}", activity.direction_string()),
                                extra_data: extra_data_vec,
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::MEMCPY_STAGE_IID,
                            });
                    }
                    let ms_start = start_offsets
                        .memset_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for activity in data.memset_activities[ms_start..].iter() {
                        let mut extra_data_vec: Vec<(String, String)> = Vec::new();
                        activity.emit_extra_data(process_id, &process_name, &mut |name, value| {
                            extra_data_vec.push((name.to_string(), value.to_string()));
                        });
                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: activity.start,
                                end: activity.end,
                                name: "Memset".to_string(),
                                extra_data: extra_data_vec,
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::MEMSET_STAGE_IID,
                            });
                    }
                }

                // Sort each channel's events by start time, then clamp to
                // prevent overlaps caused by mixed timestamp sources.
                let mut updated_prev_ends: std::collections::HashMap<(u32, u32), u64> =
                    std::collections::HashMap::new();
                for (channel_key, events) in channel_events.iter_mut() {
                    events.sort_by_key(|e| e.start);
                    let mut prev_end: u64 = start_offsets
                        .channel_prev_end
                        .get(channel_key)
                        .copied()
                        .unwrap_or(0);
                    for event in events.iter_mut() {
                        if event.start < prev_end {
                            event.start = prev_end;
                        }
                        if event.end < event.start {
                            event.end = event.start;
                        }
                        prev_end = event.end;
                    }
                    updated_prev_ends.insert(*channel_key, prev_end);
                }

                // Merge all channel events into a single list sorted by timestamp.
                let mut all_events: Vec<PendingEvent> = channel_events
                    .into_values()
                    .flat_map(|v| v.into_iter())
                    .collect();
                all_events.sort_by_key(|e| e.start);

                (all_events, channels, contexts, updated_prev_ends, is_flush)
                // state (GLOBAL_STATE lock) dropped here
            };

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_completed callback.
            let mut stop_guard_opt = stop_guard;
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for event in &all_events {
                    let timestamp = event.start;
                    let duration_ns = event.end.saturating_sub(event.start);
                    let got_first_renderstages =
                        GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                    ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                        let was_cleared = std::mem::replace(&mut inc_state.was_cleared, false);
                        let emit_interned =
                            was_cleared || got_first_renderstages & (1 << inst_id) == 0;
                        let rs_ctx = RenderStageContext {
                            channels: &channels,
                            contexts: &contexts,
                            process_id,
                        };
                        emit_render_stage_event(
                            ctx,
                            event,
                            timestamp,
                            duration_ns,
                            emit_interned,
                            &rs_ctx,
                            &event.name,
                            &event.extra_data,
                        );
                    });
                }
                if let Some(sg) = stop_guard_opt.take() {
                    let mut sg = Some(Some(sg));
                    ctx.flush(move || drop(sg.take()));
                }
            });
            drop(stop_guard_opt);

            // Phase 3: Re-acquire lock to update channel_prev_end for flush path.
            if is_flush {
                if let Ok(mut state) = GLOBAL_STATE.lock() {
                    if let Some(offsets) = state.renderstages_consumers.get_mut(&inst_id) {
                        offsets.channel_prev_end = updated_prev_ends;
                    }
                }
            }
        });
    }

    fn flush_renderstage_events(&self) {
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
                ctx.set_api(InternedGraphicsContextApi::Cuda);
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

#[allow(clippy::too_many_arguments)]
fn emit_render_stage_event<T: GpuActivity>(
    ctx: &mut TraceContext,
    activity: &T,
    timestamp: u64,
    duration_ns: u64,
    emit_interned: bool,
    rs_ctx: &RenderStageContext,
    name: &str,
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
                    .set_context(context_iid)
                    .set_name(name);
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
