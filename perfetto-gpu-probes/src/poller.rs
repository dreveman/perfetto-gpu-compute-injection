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
use crate::protos::trace::trace_packet::prelude::*;
use libc::{clock_gettime, timespec, CLOCK_BOOTTIME};
use perfetto_sdk::data_source::{DataSource, TraceContext};
use perfetto_sdk::protos::trace::trace_packet::TracePacket;
use perfetto_sdk_protos_gpu::protos::trace::system_info::gpu_info::{GpuInfo, GpuInfoGpu};
use perfetto_sdk_protos_gpu::protos::trace::trace_packet::prelude::TracePacketExt as GpuTracePacketExt;
use std::collections::HashMap;
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

/// Returns the current timestamp in nanoseconds from `CLOCK_BOOTTIME`.
pub(crate) fn trace_time_ns() -> u64 {
    let mut ts = timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { clock_gettime(CLOCK_BOOTTIME, &mut ts) };
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

/// Emits a single `FtraceEventBundle` packet containing one
/// `GpuFrequencyFtraceEvent` per changed GPU.
///
/// Frequency values from NVML/sysfs are in MHz; the ftrace `state` field
/// expects kHz, so we multiply by 1000.
fn emit_ftrace_gpu_frequency(ctx: &mut TraceContext, samples: &[FreqSample], timestamp: u64) {
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_ftrace_events(|bundle| {
            bundle.set_cpu(0);
            for s in samples {
                bundle.set_event(|event| {
                    event.set_timestamp(timestamp);
                    event.set_pid(0);
                    event.set_gpu_frequency(|gpu_freq| {
                        gpu_freq.set_gpu_id(s.gpu_index);
                        gpu_freq.set_state(s.freq_khz as u32);
                    });
                });
            }
        });
    });
}

/// Emits a single `FtraceEventBundle` packet containing one
/// `GpuMemTotalFtraceEvent` per changed GPU.
///
/// Uses pid=0 (global total). Size is in bytes.
fn emit_ftrace_gpu_mem_total(ctx: &mut TraceContext, samples: &[MemSample], timestamp: u64) {
    ctx.add_packet(|packet: &mut TracePacket| {
        packet.set_ftrace_events(|bundle| {
            bundle.set_cpu(0);
            for s in samples {
                bundle.set_event(|event| {
                    event.set_timestamp(timestamp);
                    event.set_pid(0);
                    event.set_gpu_mem_total(|gpu_mem| {
                        gpu_mem.set_gpu_id(s.gpu_index);
                        gpu_mem.set_pid(0);
                        gpu_mem.set_size(s.size_bytes);
                    });
                });
            }
        });
    });
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
    ctx.add_packet(|packet: &mut TracePacket| {
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
    poll_ms: u64,
) {
    let gpus = enumerate();
    if gpus.is_empty() {
        perfetto_elog!("no {} GPUs found, polling loop exiting", backend_name);
        return;
    }

    let mut last_freq: Vec<Option<u32>> = vec![None; gpus.len()];
    let mut last_mem: Vec<Option<u64>> = vec![None; gpus.len()];

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

        if !freq_samples.is_empty() || !mem_samples.is_empty() {
            let timestamp = trace_time_ns();
            data_source.trace(|ctx: &mut TraceContext| {
                if ctx.instance_index() != inst_id {
                    return;
                }
                if !freq_samples.is_empty() {
                    emit_ftrace_gpu_frequency(ctx, &freq_samples, timestamp);
                }
                if !mem_samples.is_empty() {
                    emit_ftrace_gpu_mem_total(ctx, &mem_samples, timestamp);
                }
            });
        }

        stop.wait(Duration::from_millis(poll_ms));
    }

    perfetto_dlog!("{} polling loop stopped (id={})", backend_name, inst_id);
}
