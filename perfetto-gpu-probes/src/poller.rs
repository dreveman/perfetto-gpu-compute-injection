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

/// Declared encoding type for a counter — drives both wire encoding and the
/// only-emit-on-change comparison key. Quantization runs *before* the
/// comparison so two raw samples that round to the same integer dedupe
/// against each other instead of being treated as distinct.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CounterValueType {
    /// Counter is integer-valued at the source — cast straight to i64 and
    /// emit as int_value (no rounding). NVML utilization (% as u32),
    /// temperature (°C as u32) live here.
    Integer,
    /// Counter arrives fractional but 1-unit precision is the natural
    /// granularity — round to the nearest whole unit before encoding as
    /// int_value. Power lives here: NVML returns milliwatts, we convert to
    /// watts and then round so consecutive samples within ±0.5 W of each
    /// other dedupe to the same int.
    IntegerRounded,
    /// Floating-point counter — encode as double_value, full f64 precision.
    /// No NVML counter currently maps here, but reserved for future
    /// fractional probes (e.g. fan RPM as a fractional ratio).
    #[expect(dead_code, reason = "reserved for future fractional counters")]
    Double,
}

/// Pre-encoding form of a counter sample. Implements PartialEq so the
/// only-emit-on-change comparison can compare directly without re-doing
/// the f64 maths.
#[derive(Clone, Copy, PartialEq)]
enum CounterValue {
    Int(i64),
    Double(f64),
}

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

    /// Declared value type for this counter. Stable per kind, so the type
    /// table doubles as the heuristic the future telemetry pipeline can
    /// consult without re-deriving it from sample values.
    fn value_type(&self) -> CounterValueType {
        match self {
            Self::Temperature | Self::Utilization | Self::MemUtilization => {
                CounterValueType::Integer
            }
            Self::PowerW => CounterValueType::IntegerRounded,
        }
    }

    /// Quantize a raw sample (in the counter's natural unit — °C for
    /// Temperature, W for PowerW, % for Utilization / MemUtilization) to
    /// its on-wire form. Apply this before comparing to the previously
    /// emitted value so dedup hits even when the raw f64 drifted in noise.
    fn quantize(&self, value: f64) -> CounterValue {
        match self.value_type() {
            CounterValueType::Integer => {
                if value.is_finite() && value >= 0.0 && value < (1u64 << 56) as f64 {
                    CounterValue::Int(value as i64)
                } else {
                    CounterValue::Double(value)
                }
            }
            CounterValueType::IntegerRounded => {
                let rounded = value.round();
                if rounded.is_finite() && rounded >= 0.0 && rounded < (1u64 << 56) as f64 {
                    CounterValue::Int(rounded as i64)
                } else {
                    CounterValue::Double(value)
                }
            }
            CounterValueType::Double => CounterValue::Double(value),
        }
    }
}

/// A counter sample from a single GPU. Holds the *quantized* value so the
/// emit path is a straight unpack and the only-emit-on-change path can
/// compare CounterValue directly.
struct CounterSample {
    gpu_index: u32,
    kind: CounterKind,
    value: CounterValue,
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

/// Compare a fresh counter reading against the last-emitted quantized value
/// and, on change, push a single backwards-looking cap packet carrying the
/// OLD value at T_now. The trace processor's backwards-looking interpretation
/// turns that cap into "OLD held until T_now", which closes the previous
/// interval at the actual change point. The NEW value is *not* emitted here —
/// it becomes the OLD value that gets capped at the next change (or at the
/// trace-end pass), so emitting it now would be redundant. The polling loop
/// initialises last_emitted to Some(CounterValue::Int(0)) and emits a
/// dedicated T_0 anchor packet so the very first cap has a meaningful T_prev.
#[allow(clippy::too_many_arguments)]
fn handle_counter_change(
    kind: CounterKind,
    raw: Option<f64>,
    last_emitted: &mut Option<CounterValue>,
    gpu_index: u32,
    backend_name: &str,
    counter_label: &str,
    unit_suffix: &str,
    caps: &mut Vec<CounterSample>,
) {
    let Some(raw) = raw else {
        return;
    };
    let new_value = kind.quantize(raw);
    if *last_emitted == Some(new_value) {
        return;
    }
    perfetto_dlog!(
        "{} GPU {}: {} {}{}",
        backend_name,
        gpu_index,
        counter_label,
        raw,
        unit_suffix
    );
    if let Some(prev) = *last_emitted {
        caps.push(CounterSample {
            gpu_index,
            kind,
            value: prev,
        });
    }
    *last_emitted = Some(new_value);
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
                        match s.value {
                            CounterValue::Int(v) => {
                                counter.set_int_value(v);
                            }
                            CounterValue::Double(v) => {
                                counter.set_double_value(v);
                            }
                        }
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
    // Per-GPU quantized "last emitted" state for the four counter kinds.
    // Initialised to Some(0) instead of None so handle_counter_change always
    // has an OLD value to push as a cap when the reading changes — paired
    // with the T_0 anchor packet emitted immediately below, this gives every
    // counter a starting point of 0 at trace-source-instance-start that the
    // first real cap will close at the iteration where the counter actually
    // produced a non-zero reading.
    let zero = CounterValue::Int(0);
    let mut last_emitted_temp: Vec<Option<CounterValue>> = vec![Some(zero); gpus.len()];
    let mut last_emitted_power: Vec<Option<CounterValue>> = vec![Some(zero); gpus.len()];
    let mut last_emitted_gpu_util: Vec<Option<CounterValue>> = vec![Some(zero); gpus.len()];
    let mut last_emitted_mem_util: Vec<Option<CounterValue>> = vec![Some(zero); gpus.len()];
    let gpu_indices: HashSet<u32> = gpus.iter().map(|g| g.index()).collect();
    let mut descriptors_emitted = false;
    // GPU counters are "backwards looking" in the trace processor: a packet
    // at timestamp T_pkt assigns its value to the interval [T_prev_pkt,
    // T_pkt]. To make this work with dedup we (1) emit a single anchor
    // packet at the data source instance start carrying value 0 for every
    // counter (gives the first real cap a meaningful T_prev so its
    // backwards-looking interval starts at T_0 instead of stretching to
    // -infinity) and (2) on every change emit one cap packet at T_now
    // carrying the OLD value, which closes the previous interval at the
    // actual change point. The NEW value implicitly becomes the next OLD
    // value and gets capped at the iteration where it changes again — or
    // at the trace-end pass for the final value. No paired "start" packet
    // is needed because the backwards-looking interpretation makes the next
    // cap retroactively establish the interval anyway.

    // T_0 anchor: one packet per GPU carrying value 0 for all four counter
    // kinds. Also triggers the lazy interned-counter-descriptor emission
    // (and the SEQ_INCREMENTAL_STATE_CLEARED flag) inside
    // emit_gpu_counter_events, since this is the first counter packet of
    // the sequence.
    let anchor_samples: Vec<CounterSample> = gpus
        .iter()
        .flat_map(|gpu| {
            ALL_COUNTER_KINDS.iter().map(move |kind| CounterSample {
                gpu_index: gpu.index(),
                kind: *kind,
                value: zero,
            })
        })
        .collect();
    let anchor_ts = trace_time_ns();
    data_source.trace(|ctx: &mut TraceContext| {
        if ctx.instance_index() != inst_id {
            return;
        }
        emit_gpu_counter_events(
            ctx,
            &anchor_samples,
            &mut descriptors_emitted,
            &gpu_indices,
            anchor_ts,
        );
    });

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

        // Read current counter values, quantize per declared CounterKind
        // type, and on each change push a single backwards-looking cap
        // carrying the OLD value at T_now. The NEW value is held in
        // last_emitted_* until it changes again or the trace ends — no
        // separate "start" packet is needed because the trace processor's
        // backwards-looking interpretation will retroactively establish the
        // NEW value's interval at the next cap (or final) emission.
        // Quantization runs *before* the comparison so two raw samples that
        // round to the same integer (PowerW samples within ±0.5 W, etc.)
        // dedupe to a no-op.
        let mut counter_caps: Vec<CounterSample> = Vec::new();
        for (i, gpu) in gpus.iter().enumerate() {
            handle_counter_change(
                CounterKind::Temperature,
                gpu.read_temperature().map(|t| t as f64),
                &mut last_emitted_temp[i],
                gpu.index(),
                backend_name,
                "temperature",
                "C",
                &mut counter_caps,
            );
            handle_counter_change(
                CounterKind::PowerW,
                gpu.read_power_usage_mw().map(|mw| mw as f64 / 1000.0),
                &mut last_emitted_power[i],
                gpu.index(),
                backend_name,
                "power",
                "W",
                &mut counter_caps,
            );
            handle_counter_change(
                CounterKind::Utilization,
                gpu.read_gpu_utilization().map(|u| u as f64),
                &mut last_emitted_gpu_util[i],
                gpu.index(),
                backend_name,
                "utilization",
                "%",
                &mut counter_caps,
            );
            handle_counter_change(
                CounterKind::MemUtilization,
                gpu.read_mem_utilization().map(|u| u as f64),
                &mut last_emitted_mem_util[i],
                gpu.index(),
                backend_name,
                "memory utilization",
                "%",
                &mut counter_caps,
            );
        }

        let has_freq_mem = !freq_samples.is_empty() || !mem_samples.is_empty();
        let has_counters = !counter_caps.is_empty();

        if has_freq_mem || has_counters {
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
                if !counter_caps.is_empty() {
                    emit_gpu_counter_events(
                        ctx,
                        &counter_caps,
                        &mut descriptors_emitted,
                        &gpu_indices,
                        timestamp,
                    );
                }
            });
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
        // Final samples carry the last-emitted quantized value so the
        // counter track extends cleanly to the trace end timestamp. No
        // dedup needed here — these always emit one packet per known
        // counter so the consumer sees the closing edge.
        if let Some(q) = last_emitted_temp[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::Temperature,
                value: q,
            });
        }
        if let Some(q) = last_emitted_power[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::PowerW,
                value: q,
            });
        }
        if let Some(q) = last_emitted_gpu_util[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::Utilization,
                value: q,
            });
        }
        if let Some(q) = last_emitted_mem_util[i] {
            final_counter_samples.push(CounterSample {
                gpu_index: gpu.index(),
                kind: CounterKind::MemUtilization,
                value: q,
            });
        }
    }

    // Closing edge: emit one packet per known counter at trace-end timestamp
    // so the backwards-looking value extends right up to the trace end. No
    // pair-emission needed here — the final packet for each counter caps
    // its current value at the trace boundary; nothing follows it.
    let has_final = !final_freq_samples.is_empty()
        || !final_mem_samples.is_empty()
        || !final_counter_samples.is_empty();
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
            if !final_counter_samples.is_empty() {
                emit_gpu_counter_events(
                    ctx,
                    &final_counter_samples,
                    &mut descriptors_emitted,
                    &gpu_indices,
                    timestamp,
                );
            }
        });
    }

    perfetto_dlog!("{} polling loop stopped (id={})", backend_name, inst_id);
}
