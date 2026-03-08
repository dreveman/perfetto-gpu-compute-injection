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
pub mod config;
pub mod metrics;
pub mod state;
pub mod tracing;

use crate::tracing::trace_time_ns;
use callbacks::{buffer_completed, buffer_requested, profiler_callback_handler};
use config::Config;
use state::GLOBAL_STATE;
use tracing::{
    get_counters_data_source, get_next_event_id, get_renderstages_data_source, is_counters_enabled,
    is_renderstages_enabled, GOT_FIRST_COUNTERS, GOT_FIRST_RENDERSTAGES,
};

use cpp_demangle::Symbol;
use cupti_profiler as profiler;
use cupti_profiler::bindings::*;
use perfetto_sdk::{
    data_source::TraceContext,
    producer::{Backends, Producer, ProducerInitArgsBuilder},
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
                InternedGpuRenderStageSpecificationRenderStageCategory,
            },
        },
        interned_data::interned_data::prelude::*,
        trace_packet::prelude::*,
    },
};
use std::{panic, ptr, sync::atomic::Ordering};

extern "C" fn end_execution() {
    let _ = panic::catch_unwind(|| {
        let _ = profiler::activity_flush_all(CUpti_ActivityFlag_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        let process_id = unsafe { libc::getpid() };
        let process_name = std::fs::read_to_string("/proc/self/comm")
            .unwrap_or_else(|_| "unknown".to_string())
            .trim_end_matches('\n')
            .to_owned();
        let mut state = match GLOBAL_STATE.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        let metric_names = state.config.metrics.clone();
        let counters_enabled = is_counters_enabled();
        let renderstages_enabled = is_renderstages_enabled();
        let verbose = state.config.verbose;

        // Finalize the last kernel launch for each context (set end timestamp)
        // and only decode counter data and evaluate metrics if counters are enabled
        if counters_enabled {
            for (_, data) in state.context_data.iter_mut() {
                if data.is_active {
                    if let Some(rp) = &mut data.range_profiler {
                        let _ = rp.stop();
                        let _ = rp.decode_counter_data();
                        if let Some(me) = &data.metric_evaluator {
                            if let Ok(infos) =
                                me.evaluate_all_ranges(&data.counter_data_image, &metric_names)
                            {
                                data.range_info.extend(infos);
                            }
                        }
                    }
                }
                // Finalize the last kernel launch with end timestamp
                if let Some(last_launch) = data.kernel_launches.last_mut() {
                    if last_launch.end == 0 {
                        last_launch.end = trace_time_ns();
                    }
                }
            }
        }

        // Emit counters data (only if counters data source enabled)
        if counters_enabled {
            get_counters_data_source().trace(|ctx: &mut TraceContext| {
                let inst_id = ctx.instance_index();
                for (_, data) in state.context_data.iter() {
                    for (range, launch) in data.range_info.iter().zip(data.kernel_launches.iter()) {
                        let duration_ns = launch.end.saturating_sub(launch.start);

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
                                        event.set_counter_descriptor(
                                            |desc: &mut GpuCounterDescriptor| {
                                                for (i, metric) in
                                                    range.metric_and_values.iter().enumerate()
                                                {
                                                    desc.set_specs(|desc: &mut GpuCounterDescriptorGpuCounterSpec| {
                                                        desc.set_counter_id(i as u32);
                                                        desc.set_name(&metric.metric_name);
                                                        desc.set_groups(
                                                        GpuCounterDescriptorGpuCounterGroup::Compute,
                                                    );
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
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
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
                                .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                .set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                                    for (i, metric) in range.metric_and_values.iter().enumerate() {
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

                        if verbose {
                            println!("Range Name: {}", range.range_name);
                            println!("Timestamp: {}", launch.start);
                            println!("Duration: {}", duration_ns);
                            println!("-----------------------------------------------------------------------------------");
                            for metric in &range.metric_and_values {
                                println!("{}: {}", metric.metric_name, metric.value);
                            }
                            println!("-----------------------------------------------------------------------------------\n");
                        }
                    }
                }
            });
        }

        // Emit renderstages data (only if renderstages data source enabled)
        if renderstages_enabled {
            get_renderstages_data_source().trace(|ctx: &mut TraceContext| {
                let inst_id = ctx.instance_index();
                for (_, data) in state.context_data.iter() {
                for (launch, activity) in data
                        .kernel_launches
                        .iter()
                        .zip(data.kernel_activities.iter())
                    {
                        // When counters are enabled, use launch timestamps captured around profiler
                        // When counters are disabled, use activity timestamps directly (they are valid)
                        let (timestamp, duration_ns) = if counters_enabled {
                            (
                                launch.start,
                                launch.end.saturating_sub(launch.start),
                            )
                        } else {
                            (
                                activity.start,
                                activity.end.saturating_sub(activity.start),
                            )
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
                        let waves_per_multiprocessor = if data.num_sms > 0 && max_active_blocks > 0
                        {
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
                        let warps_per_block = if warp_size > 0 { block_size / warp_size } else { 0 };
                        let max_active_warps = max_active_blocks * warps_per_block;
                        let regs_per_block = (regs_per_thread as i32) * block_size;
                        let max_warps_sm = if warp_size > 0 { max_threads_sm / warp_size } else { 0 };
                        let max_active_warps_pct = if max_warps_sm > 0 {
                            100.0 * max_active_warps as f64 / max_warps_sm as f64
                        } else {
                            0.0
                        };
                        let occupancy_limit_shared_mem =
                            if smem_per_block != 0 { smem_per_sm / smem_per_block } else { 16 };
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

                        if verbose {
                            println!("Timestamp: {}", timestamp);
                            println!(
                                "-----------------------------------------------------------------------------------"
                            );
                            extra_data(&mut |name: &str, value: &str| {
                                println!("{}: {}", name, value);
                            });
                            println!(
                                "-----------------------------------------------------------------------------------\n"
                            );
                        }

                        let got_first_renderstages =
                            GOT_FIRST_RENDERSTAGES.fetch_or(1 << inst_id, Ordering::SeqCst);
                        ctx.with_incremental_state(|ctx: &mut TraceContext, inc_state| {
                            let was_cleared = std::mem::replace(&mut inc_state.was_cleared, false);
                            ctx.add_packet(|packet: &mut TracePacket| {
                                packet
                                    .set_timestamp(timestamp)
                                    .set_timestamp_clock_id(BuiltinClock::BuiltinClockBoottime.into())
                                    .set_gpu_render_stage_event(|event: &mut GpuRenderStageEvent| {
                                        event
                                            .set_event_id(get_next_event_id())
                                            .set_duration(duration_ns)
                                            .set_hw_queue_iid(1)
                                            .set_stage_iid(2);
                                        extra_data(&mut |name: &str, value: &str| {
                                            event.set_extra_data(|extra_data: &mut GpuRenderStageEventExtraData| {
                                                extra_data.set_name(name);
                                                extra_data.set_value(value);
                                            });
                                        });
                                    });
                                if was_cleared || got_first_renderstages & (1 << inst_id) == 0 {
                                    packet.set_sequence_flags(
                                        TracePacketSequenceFlags::SeqIncrementalStateCleared as u32,
                                    );
                                    packet.set_interned_data(|interned: &mut InternedData| {
                                        interned.set_gpu_specifications(
                                            |spec: &mut InternedGpuRenderStageSpecification| {
                                                spec.set_iid(1);
                                                spec.set_name("Queue (0)");
                                                spec.set_category(
                                                    InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                                                );
                                            },
                                        );
                                        interned.set_gpu_specifications(
                                            |spec: &mut InternedGpuRenderStageSpecification| {
                                                spec.set_iid(2);
                                                spec.set_name("Kernel");
                                                spec.set_description("CUDA Kernel");
                                                spec.set_category(
                                                    InternedGpuRenderStageSpecificationRenderStageCategory::Compute,
                                                );
                                            },
                                        );
                                    });
                                }
                            });
                        });
                    }
                }
            });
        }
    });
}

/// Timestamp callback for CUPTI activity records.
///
/// This function is called by CUPTI to get timestamps for activity records,
/// ensuring they use the same clock as our trace timestamps.
unsafe extern "C" fn activity_timestamp_callback() -> u64 {
    trace_time_ns()
}

fn register_profiler_callbacks() -> Result<(), CUptiResult> {
    // Register custom timestamp callback so activity timestamps use our trace clock
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
    profiler::activity_enable(CUpti_ActivityKind_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)?;
    unsafe {
        profiler::activity_register_callbacks(Some(buffer_requested), Some(buffer_completed))
    }?;
    unsafe { libc::atexit(end_execution) };
    Ok(())
}

/// Entry point for the injection library.
///
/// Initializes the Perfetto producer, sets up global state, and registers CUPTI callbacks.
/// This function is intended to be called by a preload mechanism or manually at the start of the application.
#[no_mangle]
pub extern "C" fn InitializeInjection() -> i32 {
    let result = panic::catch_unwind(|| {
        let producer_args = ProducerInitArgsBuilder::new().backends(Backends::SYSTEM);
        Producer::init(producer_args.build());
        let _ = get_counters_data_source();
        let _ = get_renderstages_data_source();
        if let Ok(mut state) = GLOBAL_STATE.lock() {
            if !state.injection_initialized {
                state.injection_initialized = true;
                state.config = Config::from_env();

                if let Err(e) = register_profiler_callbacks() {
                    eprintln!("Failed to register callbacks: {:?}", e);
                    return 0;
                }
            }
        }
        1
    });
    result.unwrap_or(0)
}
