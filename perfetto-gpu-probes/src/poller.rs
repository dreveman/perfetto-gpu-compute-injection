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

//! Shared GPU polling infrastructure.
//!
//! Provides a [`PollableGpu`] trait and generic poll/emit functions
//! used by both the NVML and AMD pollers.

use crate::perfetto_dlog;
use crate::perfetto_elog;
use libc::{clock_gettime, timespec};

/// Clock ID for timestamps. Use `CLOCK_BOOTTIME` on Linux (includes suspend
/// time, matching Perfetto's default), fall back to `CLOCK_MONOTONIC` elsewhere.
#[cfg(target_os = "linux")]
const TRACE_CLOCK_ID: libc::clockid_t = libc::CLOCK_BOOTTIME;
#[cfg(not(target_os = "linux"))]
const TRACE_CLOCK_ID: libc::clockid_t = libc::CLOCK_MONOTONIC;

/// Perfetto builtin clock ID corresponding to `TRACE_CLOCK_ID`.
/// See `perfetto/common/builtin_clock.proto`.
#[cfg(target_os = "linux")]
const PERFETTO_CLOCK_ID: u32 = 6; // BUILTIN_CLOCK_BOOTTIME
#[cfg(not(target_os = "linux"))]
const PERFETTO_CLOCK_ID: u32 = 3; // BUILTIN_CLOCK_MONOTONIC
use perfetto_sdk::data_source::{DataSource, TraceContext};
use perfetto_sdk::protos::trace::{
    interned_data::interned_data::InternedData,
    trace_packet::{TracePacket, TracePacketSequenceFlags},
};
use perfetto_sdk_protos_gpu::protos::{
    common::gpu_counter_descriptor::{
        GpuCounterDescriptor, GpuCounterDescriptorGpuCounterGroup,
        GpuCounterDescriptorGpuCounterSpec, GpuCounterDescriptorMeasureUnit,
    },
    trace::generic_kernel::generic_gpu_frequency::GenericGpuFrequencyEvent,
    trace::gpu::gpu_counter_event::{
        GpuCounterEvent, GpuCounterEventGpuCounter, InternedGpuCounterDescriptor,
    },
    trace::gpu::gpu_mem_event::GpuMemTotalEvent,
    trace::interned_data::interned_data::prelude::*,
    trace::system_info::gpu_info::{GpuInfo, GpuInfoGpu},
    trace::trace_packet::prelude::TracePacketExt as GpuTracePacketExt,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// Static GPU metadata for the GpuInfo TracePacket.
pub(crate) struct GpuMetadata {
    pub name: String,
    pub vendor: String,
    pub pci_bdf: String,
    pub uuid: String,
    pub model: String,
    pub architecture: String,
    pub extra_info: Vec<(String, String)>,
}

/// A GPU that can be polled for frequency and memory usage.
pub(crate) trait PollableGpu {
    fn index(&self) -> u32;
    fn metadata(&self) -> GpuMetadata;
    fn read_frequency(&self) -> Option<u32>;
    fn read_memory_used(&self) -> Option<u64>;
    fn read_temperature(&self) -> Option<u32> {
        None
    }
    fn read_power_usage_mw(&self) -> Option<u32> {
        None
    }
    fn read_gpu_utilization(&self) -> Option<u32> {
        None
    }
    fn read_mem_utilization(&self) -> Option<u32> {
        None
    }
}

/// Per-instance stop handle used to signal a poll loop to exit.
pub(crate) struct InstanceStop {
    flag: AtomicBool,
    cvar: (Mutex<bool>, Condvar),
}

impl InstanceStop {
    pub fn new() -> Self {
        Self {
            flag: AtomicBool::new(false),
            cvar: (Mutex::new(false), Condvar::new()),
        }
    }

    pub fn signal(&self) {
        self.flag.store(true, Ordering::SeqCst);
        let (lock, cvar) = &self.cvar;
        *lock.lock().expect("mutex poisoned") = true;
        cvar.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    fn wait(&self, timeout: Duration) {
        let (lock, cvar) = &self.cvar;
        let guard = lock.lock().expect("mutex poisoned");
        let _ = cvar.wait_timeout_while(guard, timeout, |stopped| !*stopped);
    }
}

/// Returns the current timestamp in nanoseconds from the trace clock.
pub(crate) fn trace_time_ns() -> u64 {
    let mut ts = timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { clock_gettime(TRACE_CLOCK_ID, &mut ts) };
    if ret != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000u64 + (ts.tv_nsec as u64)
}

/// A GPU frequency sample that passed change-detection.
struct FreqSample {
    gpu_index: u32,
    freq_khz: u64,
}

/// A GPU memory sample that passed change-detection.
struct MemSample {
    gpu_index: u32,
    size_bytes: u64,
}

/// Emits one `GenericGpuFrequencyEvent` packet per changed GPU.
fn emit_gpu_frequency(ctx: &mut TraceContext, samples: &[FreqSample], timestamp: u64) {
    for s in samples {
        ctx.add_packet(|packet: &mut TracePacket| {
            packet.set_timestamp(timestamp);
            packet.set_timestamp_clock_id(PERFETTO_CLOCK_ID);
            GpuTracePacketExt::set_generic_gpu_frequency_event(
                packet,
                |event: &mut GenericGpuFrequencyEvent| {
                    event.set_gpu_id(s.gpu_index);
                    event.set_frequency_khz(s.freq_khz as u32);
                },
            );
        });
    }
}

/// Emits one `GpuMemTotalEvent` packet per changed GPU.
fn emit_gpu_mem_total(ctx: &mut TraceContext, samples: &[MemSample], timestamp: u64) {
    for s in samples {
        ctx.add_packet(|packet: &mut TracePacket| {
            packet.set_timestamp(timestamp);
            packet.set_timestamp_clock_id(PERFETTO_CLOCK_ID);
            GpuTracePacketExt::set_gpu_mem_total_event(packet, |event: &mut GpuMemTotalEvent| {
                event.set_gpu_id(s.gpu_index);
                event.set_pid(0);
                event.set_size(s.size_bytes);
            });
        });
    }
}

/// Counter identifier for a specific GPU counter.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum CounterKind {
    Temperature,
    PowerW,
    Utilization,
    MemUtilization,
}

const ALL_COUNTER_KINDS: [CounterKind; 4] = [
    CounterKind::Temperature,
    CounterKind::PowerW,
    CounterKind::Utilization,
    CounterKind::MemUtilization,
];

impl CounterKind {
    fn counter_id(&self) -> u32 {
        match self {
            Self::Temperature => 1,
            Self::PowerW => 2,
            Self::Utilization => 3,
            Self::MemUtilization => 4,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::PowerW => "Power",
            Self::Utilization => "Utilization",
            Self::MemUtilization => "Memory Utilization",
        }
    }

    fn unit(&self) -> GpuCounterDescriptorMeasureUnit {
        match self {
            Self::Temperature => GpuCounterDescriptorMeasureUnit::Celsius,
            Self::PowerW => GpuCounterDescriptorMeasureUnit::Watt,
            Self::Utilization | Self::MemUtilization => GpuCounterDescriptorMeasureUnit::Percent,
        }
    }

    fn group(&self) -> GpuCounterDescriptorGpuCounterGroup {
        GpuCounterDescriptorGpuCounterGroup::System
    }
}

/// A counter sample from a single GPU.
struct CounterSample {
    gpu_index: u32,
    kind: CounterKind,
    value: f64,
}

/// Emits interned counter descriptors for all GPUs.
///
/// Each GPU gets one `InternedGpuCounterDescriptor` with iid = gpu_index + 1,
/// containing all counter specs. The gpu_id on the interned descriptor handles
/// per-GPU track separation in the trace processor.
fn emit_interned_counter_descriptors(ctx: &mut TraceContext, gpu_indices: &HashSet<u32>) {
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_sequence_flags(TracePacketSequenceFlags::SeqIncrementalStateCleared as u32);
        packet.set_interned_data(|interned: &mut InternedData| {
            for &gpu_index in gpu_indices {
                interned.set_gpu_counter_descriptors(|desc: &mut InternedGpuCounterDescriptor| {
                    desc.set_iid(gpu_index as u64 + 1);
                    desc.set_gpu_id(gpu_index as i32);
                    desc.set_counter_descriptor(|cd: &mut GpuCounterDescriptor| {
                        for kind in &ALL_COUNTER_KINDS {
                            cd.set_specs(|spec: &mut GpuCounterDescriptorGpuCounterSpec| {
                                spec.set_counter_id(kind.counter_id());
                                spec.set_name(kind.name());
                                spec.set_numerator_units(kind.unit());
                                spec.set_groups(kind.group());
                            });
                        }
                    });
                });
            }
        });
    });
}

/// Emits GpuCounterEvent packets for changed counter values.
///
/// Counter events reference the interned descriptor via `counter_descriptor_iid`.
fn emit_gpu_counter_events(
    ctx: &mut TraceContext,
    samples: &[CounterSample],
    descriptors_emitted: &mut bool,
    gpu_indices: &HashSet<u32>,
    timestamp: u64,
) {
    if !*descriptors_emitted {
        emit_interned_counter_descriptors(ctx, gpu_indices);
        *descriptors_emitted = true;
    }

    // Group samples by gpu_index so each GPU gets one GpuCounterEvent packet.
    let mut by_gpu: HashMap<u32, Vec<&CounterSample>> = HashMap::new();
    for s in samples {
        by_gpu.entry(s.gpu_index).or_default().push(s);
    }

    for (&gpu_index, gpu_samples) in &by_gpu {
        let desc_iid = gpu_index as u64 + 1;
        ctx.add_packet(|packet: &mut TracePacket| {
            packet.set_timestamp(timestamp);
            packet.set_timestamp_clock_id(PERFETTO_CLOCK_ID);
            packet.set_gpu_counter_event(|event: &mut GpuCounterEvent| {
                event.set_counter_descriptor_iid(desc_iid);
                for s in gpu_samples {
                    event.set_counters(|counter: &mut GpuCounterEventGpuCounter| {
                        counter.set_counter_id(s.kind.counter_id());
                        counter.set_double_value(s.value);
                    });
                }
            });
        });
    }
}

/// Builds display names for GPUs, appending `#N` suffixes when multiple
/// GPUs share the same name (e.g., "NVIDIA A100 #0", "NVIDIA A100 #1").
pub(crate) fn build_display_names(metadata: &[GpuMetadata]) -> Vec<String> {
    // Count how many GPUs share each name.
    let mut name_counts: HashMap<&str, u32> = HashMap::new();
    for meta in metadata {
        *name_counts.entry(&meta.name).or_default() += 1;
    }

    // Assign display names, appending #N only for duplicates.
    let mut name_indices: HashMap<&str, u32> = HashMap::new();
    metadata
        .iter()
        .map(|meta| {
            if name_counts[meta.name.as_str()] > 1 {
                let idx = name_indices.entry(&meta.name).or_default();
                let display = format!("{} #{}", meta.name, idx);
                *idx += 1;
                display
            } else {
                meta.name.clone()
            }
        })
        .collect()
}

/// Emits a single `GpuInfo` TracePacket containing metadata for all GPUs.
pub(crate) fn emit_gpu_info(ctx: &mut TraceContext, metadata: &[GpuMetadata]) {
    let display_names = build_display_names(metadata);
    let timestamp = trace_time_ns();
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_timestamp(timestamp);
        packet.set_timestamp_clock_id(PERFETTO_CLOCK_ID);
        packet.set_gpu_info(|info: &mut GpuInfo| {
            for (meta, display_name) in metadata.iter().zip(display_names.iter()) {
                info.set_gpus(|g: &mut GpuInfoGpu| {
                    g.set_name(display_name);
                    g.set_vendor(&meta.vendor);
                    if !meta.pci_bdf.is_empty() {
                        g.set_pci_bdf(&meta.pci_bdf);
                    }
                    if !meta.uuid.is_empty() {
                        g.set_uuid(&meta.uuid);
                    }
                    if !meta.model.is_empty() {
                        g.set_model(&meta.model);
                    }
                    if !meta.architecture.is_empty() {
                        g.set_architecture(&meta.architecture);
                    }
                    for (k, v) in &meta.extra_info {
                        g.set_extra_info(|kv| {
                            kv.set_key(k);
                            kv.set_value(v);
                        });
                    }
                });
            }
        });
    });
}

/// Runs a generic polling loop for a single data source instance.
/// Blocks until `stop` is signaled.
///
/// 1. Enumerates GPUs via `enumerate`
/// 2. Emits a one-shot `GpuInfo` packet
/// 3. Polls GPU frequencies and memory, emitting ftrace events on change
pub(crate) fn run_poll_loop<G: PollableGpu>(
    enumerate: fn() -> Vec<G>,
    backend_name: &str,
    data_source: &'static DataSource<'static>,
    inst_id: u32,
    stop: &InstanceStop,
    poll_us: u64,
) {
    let gpus = enumerate();
    if gpus.is_empty() {
        perfetto_elog!("no {} GPUs found, polling loop exiting", backend_name);
        return;
    }

    let mut last_freq: Vec<Option<u32>> = vec![None; gpus.len()];
    let mut last_mem: Vec<Option<u64>> = vec![None; gpus.len()];
    let mut last_temp: Vec<Option<u32>> = vec![None; gpus.len()];
    let mut last_power: Vec<Option<u32>> = vec![None; gpus.len()];
    let mut last_gpu_util: Vec<Option<u32>> = vec![None; gpus.len()];
    let mut last_mem_util: Vec<Option<u32>> = vec![None; gpus.len()];
    let gpu_indices: HashSet<u32> = gpus.iter().map(|g| g.index()).collect();
    let mut descriptors_emitted = false;
    // GPU counters are "backwards looking" in the trace processor: a value
    // emitted at T2 gets assigned to T1. To compensate, we buffer the
    // previous counter samples and emit them at the current timestamp.
    let mut prev_counter_samples: Vec<CounterSample> = Vec::new();

    while !stop.is_stopped() {
        // Collect changed samples before entering the trace closure.
        let mut freq_samples = Vec::new();
        for (i, gpu) in gpus.iter().enumerate() {
            if let Some(freq_mhz) = gpu.read_frequency() {
                if last_freq[i] != Some(freq_mhz) {
                    last_freq[i] = Some(freq_mhz);
                    perfetto_dlog!(
                        "{} GPU {}: frequency {} MHz",
                        backend_name,
                        gpu.index(),
                        freq_mhz
                    );
                    freq_samples.push(FreqSample {
                        gpu_index: gpu.index(),
                        freq_khz: freq_mhz as u64 * 1000,
                    });
                }
            }
        }
        let mut mem_samples = Vec::new();
        for (i, gpu) in gpus.iter().enumerate() {
            if let Some(mem_used) = gpu.read_memory_used() {
                if last_mem[i] != Some(mem_used) {
                    last_mem[i] = Some(mem_used);
                    perfetto_dlog!(
                        "{} GPU {}: memory {} bytes",
                        backend_name,
                        gpu.index(),
                        mem_used
                    );
                    mem_samples.push(MemSample {
                        gpu_index: gpu.index(),
                        size_bytes: mem_used,
                    });
                }
            }
        }

        // Read current counter values and update last-known state.
        for (i, gpu) in gpus.iter().enumerate() {
            if let Some(temp) = gpu.read_temperature() {
                if last_temp[i] != Some(temp) {
                    perfetto_dlog!(
                        "{} GPU {}: temperature {} C",
                        backend_name,
                        gpu.index(),
                        temp
                    );
                }
                last_temp[i] = Some(temp);
            }
            if let Some(power_mw) = gpu.read_power_usage_mw() {
                if last_power[i] != Some(power_mw) {
                    perfetto_dlog!(
                        "{} GPU {}: power {} mW",
                        backend_name,
                        gpu.index(),
                        power_mw
                    );
                }
                last_power[i] = Some(power_mw);
            }
            if let Some(gpu_util) = gpu.read_gpu_utilization() {
                if last_gpu_util[i] != Some(gpu_util) {
                    perfetto_dlog!(
                        "{} GPU {}: utilization {}%",
                        backend_name,
                        gpu.index(),
                        gpu_util
                    );
                }
                last_gpu_util[i] = Some(gpu_util);
            }
            if let Some(mem_util) = gpu.read_mem_utilization() {
                if last_mem_util[i] != Some(mem_util) {
                    perfetto_dlog!(
                        "{} GPU {}: memory utilization {}%",
                        backend_name,
                        gpu.index(),
                        mem_util
                    );
                }
                last_mem_util[i] = Some(mem_util);
            }
        }

        let has_freq_mem = !freq_samples.is_empty() || !mem_samples.is_empty();
        let has_prev_counters = !prev_counter_samples.is_empty();

        if has_freq_mem || has_prev_counters {
            let timestamp = trace_time_ns();
            data_source.trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                if !freq_samples.is_empty() {
                    emit_gpu_frequency(ctx, &freq_samples, timestamp);
                }
                if !mem_samples.is_empty() {
                    emit_gpu_mem_total(ctx, &mem_samples, timestamp);
                }
                if has_prev_counters {
                    emit_gpu_counter_events(
                        ctx,
                        &prev_counter_samples,
                        &mut descriptors_emitted,
                        &gpu_indices,
                        timestamp,
                    );
                }
            });
        }
        // Build buffered samples from last-known values for next iteration.
        prev_counter_samples.clear();
        for (i, gpu) in gpus.iter().enumerate() {
            if let Some(temp) = last_temp[i] {
                prev_counter_samples.push(CounterSample {
                    gpu_index: gpu.index(),
                    kind: CounterKind::Temperature,
                    value: temp as f64,
                });
            }
            if let Some(power_mw) = last_power[i] {
                prev_counter_samples.push(CounterSample {
                    gpu_index: gpu.index(),
                    kind: CounterKind::PowerW,
                    value: power_mw as f64 / 1000.0,
                });
            }
            if let Some(gpu_util) = last_gpu_util[i] {
                prev_counter_samples.push(CounterSample {
                    gpu_index: gpu.index(),
                    kind: CounterKind::Utilization,
                    value: gpu_util as f64,
                });
            }
            if let Some(mem_util) = last_mem_util[i] {
                prev_counter_samples.push(CounterSample {
                    gpu_index: gpu.index(),
                    kind: CounterKind::MemUtilization,
                    value: mem_util as f64,
                });
            }
        }

        stop.wait(Duration::from_micros(poll_us));
    }

    // Emit one final sample with all last-known values so counter tracks
    // extend to the end of the trace session (counters are backwards-looking).
    let mut final_freq_samples = Vec::new();
    let mut final_mem_samples = Vec::new();
    let mut final_counter_samples = Vec::new();
    for (i, gpu) in gpus.iter().enumerate() {
        if let Some(freq_mhz) = last_freq[i] {
            final_freq_samples.push(FreqSample {
                gpu_index: gpu.index(),
                freq_khz: freq_mhz as u64 * 1000,
            });
        }
        if let Some(mem_used) = last_mem[i] {
            final_mem_samples.push(MemSample {
                gpu_index: gpu.index(),
                size_bytes: mem_used,
            });
        }
        if let Some(temp) = last_temp[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::Temperature,
                value: temp as f64,
            });
        }
        if let Some(power_mw) = last_power[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::PowerW,
                value: power_mw as f64 / 1000.0,
            });
        }
        if let Some(gpu_util) = last_gpu_util[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::Utilization,
                value: gpu_util as f64,
            });
        }
        if let Some(mem_util) = last_mem_util[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::MemUtilization,
                value: mem_util as f64,
            });
        }
    }

    // Emit any remaining buffered counter samples plus final values.
    let mut combined_counter_samples = prev_counter_samples;
    combined_counter_samples.extend(final_counter_samples);
    let has_final = !final_freq_samples.is_empty()
        || !final_mem_samples.is_empty()
        || !combined_counter_samples.is_empty();
    if has_final {
        let timestamp = trace_time_ns();
        data_source.trace(|ctx: &mut TraceContext| {
            if ctx.instance_index() != inst_id {
                return;
            }
            if !final_freq_samples.is_empty() {
                emit_gpu_frequency(ctx, &final_freq_samples, timestamp);
            }
            if !final_mem_samples.is_empty() {
                emit_gpu_mem_total(ctx, &final_mem_samples, timestamp);
            }
            if !combined_counter_samples.is_empty() {
                emit_gpu_counter_events(
                    ctx,
                    &combined_counter_samples,
                    &mut descriptors_emitted,
                    &gpu_indices,
                    timestamp,
                );
            }
        });
    }

    perfetto_dlog!("{} polling loop stopped (id={})", backend_name, inst_id);
}
