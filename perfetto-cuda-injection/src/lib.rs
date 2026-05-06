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
    get_counter_config, get_counters_data_source, get_renderstages_data_source,
    get_track_event_data_source, register_backend, trace_time_ns, GpuBackend,
};
use perfetto_sdk::{
    data_source::{StopGuard, TraceContext},
    producer::{Backends, Producer, ProducerInitArgsBuilder},
    protos::{
        common::builtin_clock::BuiltinClock,
        trace::{
            interned_data::interned_data::InternedData,
            trace_packet::{TracePacket, TracePacketSequenceFlags},
            track_event::track_event::EventName,
        },
    },
    track_event::TrackEvent,
};
use perfetto_sdk_protos_gpu::protos::trace::gpu::gpu_interned_data::InternedDataExt as _;
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
                GpuRenderStageEvent, GpuRenderStageEventComputeKernelLaunch,
                GpuRenderStageEventDim3, GpuRenderStageEventExtraComputeArg,
                InternedComputeArgName, InternedComputeKernel, InternedGpuRenderStageSpecification,
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
    collections::HashSet,
    panic, ptr,
    sync::atomic::{AtomicU8, Ordering},
};

use cupti_profiler as profiler;
use cupti_profiler::bindings::*;

// ---------------------------------------------------------------------------
// CuptiBackend implementation
// ---------------------------------------------------------------------------

/// 3-state teardown guard: 0=not started, 1=in progress, 2=done.
static CUPTI_TEARDOWN_STATE: AtomicU8 = AtomicU8::new(0);

struct CuptiBackend;

/// Collected counter event data for emission outside of GLOBAL_STATE lock.
struct CollectedCounterEvent {
    timestamp_start: u64,
    timestamp_end: u64,
    gpu_id: i32,
    metrics: Vec<(String, f64)>,
    /// Bitmask of instance IDs that want counters for this kernel.
    profiled_instances: u8,
}

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

    fn on_first_counters_start(&self) {
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
    }

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
            let _ = profiler::activity_flush_all(0);
        }
    }

    fn finalize_range_profiler(&self) {
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            for (_, data) in state.context_data.iter_mut() {
                data.finalize_profiler(false);
                if let Some(last_launch) = data
                    .kernel_launches
                    .iter_mut()
                    .rev()
                    .find(|l| l.profiled_instances != 0)
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
                    // `range_info` is dense (one entry per *profiled* kernel)
                    // while `kernel_launches` and `kernel_activities` are sparse
                    // (one entry per *every* cuLaunchKernel). Walk launches and
                    // advance the range_info iterator only on profiled kernels
                    // so each profiled kernel gets paired with its own metrics
                    // — zipping them by index would slide range_info forward
                    // by exactly `skip` positions and chop the last `skip`
                    // results off the emitted events.
                    let mut range_iter = data.range_info[range_start..].iter();
                    for (launch, activity) in data.kernel_launches[launch_start..]
                        .iter()
                        .zip(data.kernel_activities[activity_start..].iter())
                    {
                        if launch.end == 0 {
                            continue;
                        }
                        if launch.profiled_instances == 0 {
                            continue;
                        }
                        let Some(range) = range_iter.next() else {
                            break;
                        };
                        events.push(CollectedCounterEvent {
                            timestamp_start: launch.start,
                            timestamp_end: launch.end,
                            gpu_id: activity.gpu_id() as i32,
                            metrics: range
                                .metric_and_values
                                .iter()
                                .map(|m| (m.metric_name.clone(), m.value))
                                .collect(),
                            profiled_instances: launch.profiled_instances,
                        });
                    }
                }
                events
                // state (GLOBAL_STATE lock) dropped here
            };

            // Per-instance filtering using the profiled_instances bitmask:
            // 1. Skip kernels where this instance's bit is not set.
            // 2. Filter metrics by counter_names (only emit the counters this
            //    instance requested).
            let inst_bit = 1u8 << inst_id;
            let collected_events: Vec<_> = if let Some(cfg) = get_counter_config(inst_id) {
                let has_counter_filter = !cfg.counter_names.is_empty();
                let requested: HashSet<&str> =
                    cfg.counter_names.iter().map(|s| s.as_str()).collect();
                collected_events
                    .into_iter()
                    .filter(|e| e.profiled_instances & inst_bit != 0)
                    .map(|mut e| {
                        if has_counter_filter {
                            e.metrics
                                .retain(|(name, _)| requested.contains(name.as_str()));
                        }
                        e
                    })
                    .filter(|e| !e.metrics.is_empty())
                    .collect()
            } else {
                collected_events
                    .into_iter()
                    .filter(|e| e.profiled_instances & inst_bit != 0)
                    .collect()
            };

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_completed callback.
            // Collect the set of GPU IDs present in this batch.
            let gpu_ids: HashSet<i32> = collected_events.iter().map(|e| e.gpu_id).collect();
            let mut stop_guard_opt = stop_guard;
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for event in &collected_events {
                    ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                        let was_cleared = std::mem::replace(&mut inc_state.was_cleared, false);
                        if was_cleared {
                            emit_interned_counter_descriptors(ctx, &collected_events, &gpu_ids);
                        }
                        // Emit start sample (zero values).
                        let desc_iid = event.gpu_id as u64 + 1;
                        ctx.add_packet(|packet: &mut TracePacket| {
                            packet
                                .set_timestamp(event.timestamp_start)
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|ce: &mut GpuCounterEvent| {
                                    ce.set_counter_descriptor_iid(desc_iid);
                                    for (i, _) in event.metrics.iter().enumerate() {
                                        ce.set_counters(
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
                                .set_timestamp(event.timestamp_end)
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|ce: &mut GpuCounterEvent| {
                                    ce.set_counter_descriptor_iid(desc_iid);
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
                name_iid: u64,
                channel_id: u32,
                channel_type: u32,
                activity_device_id: u32,
                activity_context_id: u32,
                activity_stream_id: u32,
                stage_iid: u64,
                correlation_id: u32,
                // Structured compute kernel fields (kernel events only).
                kernel_iid: Option<u64>,
                kernel_mangled_name: Option<String>,
                kernel_demangled_name: Option<String>,
                kernel_arch: Option<String>,
                kernel_registers_per_thread: Option<u64>,
                kernel_shared_mem_static: Option<u64>,
                kernel_func_cache_config: Option<String>,
                kernel_shared_mem_config_size: Option<u64>,
                launch_grid: Option<(u32, u32, u32)>,
                launch_block: Option<(u32, u32, u32)>,
                launch_args: Vec<(u64, KernelArgValue)>,
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
                fn correlation_id(&self) -> u32 {
                    self.correlation_id
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
                    let ka_start = start_offsets
                        .kernel_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    let launch_map: std::collections::HashMap<u32, &state::KernelLaunch> = data
                        .kernel_launches
                        .iter()
                        .map(|l| (l.correlation_id, l))
                        .collect();
                    for activity in data.kernel_activities[ka_start..].iter() {
                        let launch = launch_map.get(&activity.correlation_id);
                        let (raw_start, raw_end) =
                            if launch.is_some_and(|l| l.profiled_instances != 0) {
                                let l = launch.unwrap();
                                (l.start, l.end)
                            } else {
                                (activity.start, activity.end)
                            };
                        if raw_start == 0 || raw_end < raw_start {
                            continue;
                        }
                        let demangled = perfetto_gpu_compute_injection::kernel::demangle_name(
                            &activity.kernel_name,
                        );
                        let grid_size =
                            activity.grid_size.0 * activity.grid_size.1 * activity.grid_size.2;
                        let block_size =
                            activity.block_size.0 * activity.block_size.1 * activity.block_size.2;
                        let thread_count = grid_size * block_size;
                        let cache_mode = launch.map_or(0, |l| l.cache_mode);
                        let max_active_blocks = launch.map_or(0, |l| l.max_active_blocks_per_sm);
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

                        // Build structured compute kernel fields.
                        let arch = format!("CC_{}{}", major, minor);
                        #[allow(nonstandard_style)]
                        let func_cache_config_str = match cache_mode as u32 {
                            CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_NONE => "CachePreferNone",
                            CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_SHARED => "CachePreferShared",
                            CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_L1 => "CachePreferL1",
                            CUfunc_cache_enum_CU_FUNC_CACHE_PREFER_EQUAL => "CachePreferEqual",
                            _ => "n/a",
                        };

                        let launch_args: Vec<(u64, KernelArgValue)> = vec![
                            (
                                arg_iid("workgroup_size"),
                                KernelArgValue::Uint(block_size as u64),
                            ),
                            (arg_iid("grid_size"), KernelArgValue::Uint(grid_size as u64)),
                            (
                                arg_iid("thread_count"),
                                KernelArgValue::Uint(thread_count as u64),
                            ),
                            (
                                arg_iid("shared_mem_dynamic"),
                                KernelArgValue::Uint(activity.dynamic_shared_memory as u64),
                            ),
                            (
                                arg_iid("waves_per_multiprocessor"),
                                KernelArgValue::Double(waves_per_multiprocessor),
                            ),
                            (
                                arg_iid("occupancy_limit_blocks"),
                                KernelArgValue::Uint(max_blocks_sm as u64),
                            ),
                            (
                                arg_iid("occupancy_limit_registers"),
                                KernelArgValue::Uint(occupancy_limit_registers as u64),
                            ),
                            (
                                arg_iid("occupancy_limit_shared_mem"),
                                KernelArgValue::Uint(occupancy_limit_shared_mem as u64),
                            ),
                            (
                                arg_iid("occupancy_limit_warps"),
                                KernelArgValue::Uint(occupancy_limit_warps as u64),
                            ),
                            (
                                arg_iid("sm__maximum_warps_per_active_cycle_pct"),
                                KernelArgValue::Double(max_active_warps_pct),
                            ),
                            (
                                arg_iid("sm__maximum_warps_avg_per_active_cycle"),
                                KernelArgValue::Uint(max_active_warps as u64),
                            ),
                            (
                                arg_iid("shared_mem_driver"),
                                KernelArgValue::Uint(smem_per_block as u64),
                            ),
                        ];

                        let simplified_name =
                            perfetto_gpu_compute_injection::kernel::simplify_name(&demangled);
                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: raw_start,
                                end: raw_end,
                                name: simplified_name.to_string(),
                                name_iid: kernel_name_iid(simplified_name),
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::KERNEL_STAGE_IID,
                                correlation_id: activity.correlation_id,
                                kernel_iid: Some(kernel_name_iid(&activity.kernel_name)),
                                kernel_mangled_name: Some(activity.kernel_name.clone()),
                                kernel_demangled_name: Some(demangled.clone()),
                                kernel_arch: Some(arch),
                                kernel_registers_per_thread: Some(
                                    activity.registers_per_thread as u64,
                                ),
                                kernel_shared_mem_static: Some(
                                    activity.static_shared_memory as u64,
                                ),
                                kernel_func_cache_config: Some(func_cache_config_str.to_string()),
                                kernel_shared_mem_config_size: Some(49152),
                                launch_grid: Some((
                                    activity.grid_size.0 as u32,
                                    activity.grid_size.1 as u32,
                                    activity.grid_size.2 as u32,
                                )),
                                launch_block: Some((
                                    activity.block_size.0 as u32,
                                    activity.block_size.1 as u32,
                                    activity.block_size.2 as u32,
                                )),
                                launch_args,
                            });
                    }
                    let mc_start = start_offsets
                        .memcpy_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for activity in data.memcpy_activities[mc_start..].iter() {
                        if activity.start == 0 || activity.end < activity.start {
                            injection_log!(
                                "WARNING: memcpy activity with invalid timestamps \
                                 (start={}, end={}), skipping: {} bytes",
                                activity.start,
                                activity.end,
                                activity.bytes
                            );
                            continue;
                        }
                        let memcpy_name = format!("Memcpy {}", activity.direction_string());
                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: activity.start,
                                end: activity.end,
                                name: memcpy_name.clone(),
                                name_iid: kernel_name_iid(&memcpy_name),
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::MEMCPY_STAGE_IID,
                                correlation_id: activity.correlation_id,
                                kernel_iid: None,
                                kernel_mangled_name: None,
                                kernel_demangled_name: None,
                                kernel_arch: None,
                                kernel_registers_per_thread: None,
                                kernel_shared_mem_static: None,
                                kernel_func_cache_config: None,
                                kernel_shared_mem_config_size: None,
                                launch_grid: None,
                                launch_block: None,
                                launch_args: Vec::new(),
                            });
                    }
                    let ms_start = start_offsets
                        .memset_activities
                        .get(ctx_id)
                        .copied()
                        .unwrap_or(0);
                    for activity in data.memset_activities[ms_start..].iter() {
                        if activity.start == 0 || activity.end < activity.start {
                            injection_log!(
                                "WARNING: memset activity with invalid timestamps \
                                 (start={}, end={}), skipping: {} bytes",
                                activity.start,
                                activity.end,
                                activity.bytes
                            );
                            continue;
                        }
                        channel_events
                            .entry((activity.channel_id, activity.channel_type))
                            .or_default()
                            .push(PendingEvent {
                                start: activity.start,
                                end: activity.end,
                                name: "Memset".to_string(),
                                name_iid: kernel_name_iid("Memset"),
                                channel_id: activity.channel_id,
                                channel_type: activity.channel_type,
                                activity_device_id: activity.device_id,
                                activity_context_id: activity.context_id,
                                activity_stream_id: activity.stream_id,
                                stage_iid: state::MEMSET_STAGE_IID,
                                correlation_id: activity.correlation_id,
                                kernel_iid: None,
                                kernel_mangled_name: None,
                                kernel_demangled_name: None,
                                kernel_arch: None,
                                kernel_registers_per_thread: None,
                                kernel_shared_mem_static: None,
                                kernel_func_cache_config: None,
                                kernel_shared_mem_config_size: None,
                                launch_grid: None,
                                launch_block: None,
                                launch_args: Vec::new(),
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
                        if event.end <= event.start {
                            event.end = event.start + 1;
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

                // Advance consumer offsets NOW (while holding the lock)
                // to exactly what we consumed. This prevents the race
                // where advance_and_drain sets offsets to a length that
                // includes activities added by buffer_completed between
                // emit and drain.
                if is_flush {
                    let lengths: Vec<(u32, usize, usize, usize, usize)> = state
                        .context_data
                        .iter()
                        .map(|(&ctx_id, data)| {
                            (
                                ctx_id,
                                data.kernel_launches.len(),
                                data.kernel_activities.len(),
                                data.memcpy_activities.len(),
                                data.memset_activities.len(),
                            )
                        })
                        .collect();
                    if let Some(offsets) = state.renderstages_consumers.get_mut(&inst_id) {
                        for (ctx_id, kl, ka, mc, ms) in &lengths {
                            offsets.kernel_launches.insert(*ctx_id, *kl);
                            offsets.kernel_activities.insert(*ctx_id, *ka);
                            offsets.memcpy_activities.insert(*ctx_id, *mc);
                            offsets.memset_activities.insert(*ctx_id, *ms);
                        }
                    }
                }

                (all_events, channels, contexts, updated_prev_ends, is_flush)
                // state (GLOBAL_STATE lock) dropped here
            };

            // Phase 2: Emit collected events without holding GLOBAL_STATE.
            // This prevents deadlock with buffer_completed callback.

            // Collect unique event names and kernels for interning.
            let mut unique_event_names: Vec<(u64, String)> = Vec::new();
            {
                let mut seen_name_iids: HashSet<u64> = HashSet::new();
                for event in &all_events {
                    if seen_name_iids.insert(event.name_iid) {
                        unique_event_names.push((event.name_iid, event.name.clone()));
                    }
                }
            }
            let mut unique_kernels: Vec<UniqueKernel> = Vec::new();
            {
                let mut seen_kernel_iids: HashSet<u64> = HashSet::new();
                for event in &all_events {
                    if let Some(kiid) = event.kernel_iid {
                        if seen_kernel_iids.insert(kiid) {
                            unique_kernels.push(UniqueKernel {
                                iid: kiid,
                                mangled_name: event.kernel_mangled_name.clone().unwrap_or_default(),
                                demangled_name: event
                                    .kernel_demangled_name
                                    .clone()
                                    .unwrap_or_default(),
                                arch: event.kernel_arch.clone().unwrap_or_default(),
                                registers_per_thread: event
                                    .kernel_registers_per_thread
                                    .unwrap_or(0),
                                shared_mem_static: event.kernel_shared_mem_static.unwrap_or(0),
                                func_cache_config: event
                                    .kernel_func_cache_config
                                    .clone()
                                    .unwrap_or_default(),
                                shared_mem_config_size: event
                                    .kernel_shared_mem_config_size
                                    .unwrap_or(0),
                                process_name: process_name.clone(),
                                process_id: process_id as u64,
                            });
                        }
                    }
                }
            }

            let mut stop_guard_opt = stop_guard;
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                for event in &all_events {
                    let timestamp = event.start;
                    let duration_ns = event.end.saturating_sub(event.start);
                    ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                        let was_cleared = std::mem::replace(&mut inc_state.was_cleared, false);
                        let rs_ctx = RenderStageContext {
                            channels: &channels,
                            contexts: &contexts,
                            process_id,
                            unique_kernels: &unique_kernels,
                            unique_event_names: &unique_event_names,
                        };
                        emit_render_stage_event(
                            ctx,
                            event,
                            timestamp,
                            duration_ns,
                            was_cleared,
                            &rs_ctx,
                            event.name_iid,
                            event.kernel_iid,
                            event.launch_grid,
                            event.launch_block,
                            &event.launch_args,
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
// Compute kernel structured proto constants and helpers
// ---------------------------------------------------------------------------

const COMPUTE_ARG_NAMES: &[(u64, &str)] = &[
    (1, "workgroup_size"),
    (2, "grid_size"),
    (3, "thread_count"),
    (4, "shared_mem_dynamic"),
    (5, "waves_per_multiprocessor"),
    (6, "occupancy_limit_blocks"),
    (7, "occupancy_limit_registers"),
    (8, "occupancy_limit_shared_mem"),
    (9, "occupancy_limit_warps"),
    (10, "sm__maximum_warps_per_active_cycle_pct"),
    (11, "sm__maximum_warps_avg_per_active_cycle"),
    (12, "process_name"),
    (13, "process_id"),
    (14, "registers_per_thread"),
    (15, "shared_mem_static"),
    (16, "func_cache_config"),
    (19, "shared_mem_config_size"),
    (20, "shared_mem_driver"),
];

fn arg_iid(name: &str) -> u64 {
    COMPUTE_ARG_NAMES
        .iter()
        .find(|(_, n)| *n == name)
        .unwrap_or_else(|| panic!("unknown compute arg name: {name}"))
        .0
}

/// Simple hash of kernel name to produce a stable IID for InternedComputeKernel.
fn kernel_name_iid(name: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in name.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    // Avoid 0 (reserved) by ensuring non-zero
    if h == 0 {
        1
    } else {
        h
    }
}

enum KernelArgValue {
    Uint(u64),
    Double(f64),
    #[allow(dead_code)]
    Str(String),
}

fn set_kernel_arg_uint(kernel: &mut InternedComputeKernel, name: &str, value: u64) {
    kernel.set_args(|arg: &mut GpuRenderStageEventExtraComputeArg| {
        arg.set_name_iid(arg_iid(name));
        arg.set_uint_value(value);
    });
}

fn set_kernel_arg_string(kernel: &mut InternedComputeKernel, name: &str, value: &str) {
    kernel.set_args(|arg: &mut GpuRenderStageEventExtraComputeArg| {
        arg.set_name_iid(arg_iid(name));
        arg.set_string_value(value);
    });
}

// ---------------------------------------------------------------------------
// Render stage helpers
// ---------------------------------------------------------------------------

/// Information about a unique kernel for interning.
struct UniqueKernel {
    iid: u64,
    mangled_name: String,
    demangled_name: String,
    arch: String,
    registers_per_thread: u64,
    shared_mem_static: u64,
    func_cache_config: String,
    shared_mem_config_size: u64,
    process_name: String,
    process_id: u64,
}

fn emit_interned_specifications(
    packet: &mut TracePacket,
    channels: &std::collections::HashSet<(u32, u32)>,
    contexts: &std::collections::HashSet<u32>,
    process_id: i32,
    unique_kernels: &[UniqueKernel],
    unique_event_names: &[(u64, String)],
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
                spec.set_name(format!("Channel #{}", channel_id + 1));
                spec.set_category(queue_category);
            });
        }
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(KERNEL_STAGE_IID);
            spec.set_name("Kernel");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Compute);
        });
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(MEMCPY_STAGE_IID);
            spec.set_name("MemoryTransfer");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Other);
        });
        interned.set_gpu_specifications(|spec: &mut InternedGpuRenderStageSpecification| {
            spec.set_iid(MEMSET_STAGE_IID);
            spec.set_name("MemorySet");
            spec.set_category(InternedGpuRenderStageSpecificationRenderStageCategory::Other);
        });
        for &(iid, name) in COMPUTE_ARG_NAMES {
            interned.set_compute_arg_names(|an: &mut InternedComputeArgName| {
                an.set_iid(iid);
                an.set_name(name);
            });
        }
        for (iid, name) in unique_event_names {
            interned.set_event_names(|en: &mut EventName| {
                en.set_iid(*iid);
                en.set_name(name);
            });
        }
        for kernel in unique_kernels {
            interned.set_compute_kernels(|ck: &mut InternedComputeKernel| {
                ck.set_iid(kernel.iid);
                ck.set_name(&kernel.mangled_name);
                ck.set_demangled_name(&kernel.demangled_name);
                ck.set_arch(&kernel.arch);
                set_kernel_arg_uint(ck, "registers_per_thread", kernel.registers_per_thread);
                set_kernel_arg_uint(ck, "shared_mem_static", kernel.shared_mem_static);
                set_kernel_arg_string(ck, "func_cache_config", &kernel.func_cache_config);
                set_kernel_arg_uint(ck, "shared_mem_config_size", kernel.shared_mem_config_size);
                set_kernel_arg_string(ck, "process_name", &kernel.process_name);
                set_kernel_arg_uint(ck, "process_id", kernel.process_id);
            });
        }
    });
}

/// Hardware block prefix groups for organizing instrumented counters.
/// Each entry is (metric_name_prefix, display_name).
const COUNTER_GROUPS: &[(&str, &str)] = &[
    ("dram__", "DRAM"),
    ("gpc__", "GPC"),
    ("sm__", "SM"),
    ("gpu__", "GPU"),
    ("l1tex__", "L1TEX"),
    ("lts__", "LTS"),
];

/// Emit interned counter descriptors for all GPUs in the batch.
///
/// Each GPU gets one `InternedGpuCounterDescriptor` with iid = gpu_id + 1,
/// containing all counter specs with simple 0-based counter_ids. The gpu_id
/// on the interned descriptor handles per-GPU track separation.
///
/// Counters are organized into hardware block groups based on their metric
/// name prefix (e.g. `sm__` → "SM", `dram__` → "DRAM").
fn emit_interned_counter_descriptors(
    ctx: &mut TraceContext,
    collected_events: &[CollectedCounterEvent],
    gpu_ids: &HashSet<i32>,
) {
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_sequence_flags(TracePacketSequenceFlags::SeqIncrementalStateCleared as u32);
        packet.set_interned_data(|interned: &mut InternedData| {
            for &gpu_id in gpu_ids {
                // Find the first event for this GPU to get metric names.
                let Some(sample) = collected_events.iter().find(|e| e.gpu_id == gpu_id) else {
                    continue;
                };
                interned.set_gpu_counter_descriptors(|desc: &mut InternedGpuCounterDescriptor| {
                    desc.set_iid(gpu_id as u64 + 1);
                    desc.set_gpu_id(gpu_id);
                    desc.set_counter_descriptor(|cd: &mut GpuCounterDescriptor| {
                        for (i, (metric_name, _)) in sample.metrics.iter().enumerate() {
                            cd.set_specs(|spec: &mut GpuCounterDescriptorGpuCounterSpec| {
                                spec.set_counter_id(i as u32);
                                spec.set_name(metric_name);
                                spec.set_groups(GpuCounterDescriptorGpuCounterGroup::Compute);
                            });
                        }
                        // Group counters by hardware block prefix.
                        for (group_id, &(prefix, group_name)) in COUNTER_GROUPS.iter().enumerate() {
                            let member_ids: Vec<u32> = sample
                                .metrics
                                .iter()
                                .enumerate()
                                .filter(|(_, (name, _))| name.starts_with(prefix))
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

struct RenderStageContext<'a> {
    channels: &'a std::collections::HashSet<(u32, u32)>,
    contexts: &'a std::collections::HashSet<u32>,
    process_id: i32,
    unique_kernels: &'a [UniqueKernel],
    unique_event_names: &'a [(u64, String)],
}

#[allow(clippy::too_many_arguments)]
fn emit_render_stage_event<T: GpuActivity>(
    ctx: &mut TraceContext,
    activity: &T,
    timestamp: u64,
    duration_ns: u64,
    emit_interned: bool,
    rs_ctx: &RenderStageContext,
    name_iid: u64,
    kernel_iid: Option<u64>,
    launch_grid: Option<(u32, u32, u32)>,
    launch_block: Option<(u32, u32, u32)>,
    launch_args: &[(u64, KernelArgValue)],
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
                    .set_event_id(activity.correlation_id() as u64)
                    .set_duration(duration_ns)
                    .set_gpu_id(gpu_id)
                    .set_hw_queue_iid(hw_queue_iid)
                    .set_stage_iid(stage_iid)
                    .set_context(context_iid)
                    .set_name_iid(name_iid);
                if let Some(kiid) = kernel_iid {
                    // Structured compute kernel event.
                    event.set_kernel_iid(kiid);
                    event.set_launch(|launch: &mut GpuRenderStageEventComputeKernelLaunch| {
                        if let Some((gx, gy, gz)) = launch_grid {
                            launch.set_grid_size(|d: &mut GpuRenderStageEventDim3| {
                                d.set_x(gx);
                                d.set_y(gy);
                                d.set_z(gz);
                            });
                        }
                        if let Some((bx, by, bz)) = launch_block {
                            launch.set_workgroup_size(|d: &mut GpuRenderStageEventDim3| {
                                d.set_x(bx);
                                d.set_y(by);
                                d.set_z(bz);
                            });
                        }
                        for (name_iid, value) in launch_args {
                            launch.set_args(|arg: &mut GpuRenderStageEventExtraComputeArg| {
                                arg.set_name_iid(*name_iid);
                                match value {
                                    KernelArgValue::Uint(v) => {
                                        arg.set_uint_value(*v);
                                    }
                                    KernelArgValue::Double(v) => {
                                        arg.set_double_value(*v);
                                    }
                                    KernelArgValue::Str(v) => {
                                        arg.set_string_value(v);
                                    }
                                }
                            });
                        }
                    });
                }
            });
        if emit_interned {
            emit_interned_specifications(
                packet,
                rs_ctx.channels,
                rs_ctx.contexts,
                rs_ctx.process_id,
                rs_ctx.unique_kernels,
                rs_ctx.unique_event_names,
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

    // Enable entire RUNTIME and DRIVER API domains so we can emit
    // track events directly from callbacks for all API calls.
    unsafe {
        profiler::enable_domain(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_RUNTIME_API,
        )
    }?;
    unsafe {
        profiler::enable_domain(
            1,
            subscriber,
            CUpti_CallbackDomain_CUPTI_CB_DOMAIN_DRIVER_API,
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
    unsafe { profiler::enable_domain(1, subscriber, CUpti_CallbackDomain_CUPTI_CB_DOMAIN_NVTX) }?;
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
        TrackEvent::init();
        let _ = get_counters_data_source();
        let _ = get_renderstages_data_source();
        let _ = get_track_event_data_source();

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
