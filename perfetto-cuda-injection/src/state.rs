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

use crate::cupti_profiler::bindings::*;
use crate::cupti_profiler::cuda;
use crate::cupti_profiler::*;
use once_cell::sync::Lazy;
use perfetto_gpu_compute_injection::config::Config;
use std::{collections::HashMap, sync::Mutex};

/// Common interface for GPU activity types that can emit render stage events.
pub trait GpuActivity {
    /// Activity start timestamp (nanoseconds).
    fn start(&self) -> u64;
    /// Activity end timestamp (nanoseconds).
    fn end(&self) -> u64;
    /// CUDA device ID.
    fn device_id(&self) -> u32;
    /// CUDA context ID.
    fn context_id(&self) -> u32;
    /// CUDA stream ID.
    fn stream_id(&self) -> u32;
    /// Channel ID for work submission.
    fn channel_id(&self) -> u32;
    /// Channel type.
    fn channel_type(&self) -> u32;
    /// Stage IID for this activity type.
    fn stage_iid(&self) -> u64;
    /// Emit extra_data fields specific to this activity type.
    fn emit_extra_data(
        &self,
        process_id: i32,
        process_name: &str,
        emit: &mut dyn FnMut(&str, &str),
    );

    /// Returns the nvidia-smi device index as gpu_id.
    /// Falls back to the CUDA ordinal if NVML is unavailable.
    fn gpu_id(&self) -> u32 {
        cuda::get_nvidia_smi_index(self.device_id() as CUdevice)
    }
}

/// Represents a specific kernel launch event.
pub struct KernelLaunch {
    pub function: CUfunction,
    /// Trace timestamp captured just before range profiler push (when counters enabled).
    pub start: u64,
    /// Trace timestamp captured after range profiler pop completes all passes (when counters enabled).
    pub end: u64,
    /// Whether this launch was profiled by the range profiler (counters were enabled).
    /// When true, `start`/`end` are valid CPU-side timestamps to use for renderstage events.
    /// When false, activity timestamps from CUPTI activity records should be used instead.
    pub profiled: bool,
    /// Pre-computed cache mode from `CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`.
    pub cache_mode: i32,
    /// Pre-computed max active blocks per SM from `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
    pub max_active_blocks_per_sm: i32,
}

/// Detailed activity information for a kernel execution.
///
/// Gathered from CUPTI activity records.
pub struct KernelActivity {
    pub kernel_name: String,
    pub grid_size: (i32, i32, i32),
    pub block_size: (i32, i32, i32),
    pub registers_per_thread: u16,
    pub dynamic_shared_memory: i32,
    pub static_shared_memory: i32,
    /// Kernel execution start timestamp (nanoseconds).
    pub start: u64,
    /// Kernel execution end timestamp (nanoseconds).
    pub end: u64,
    /// CUDA device ID.
    pub device_id: u32,
    /// CUDA context ID.
    pub context_id: u32,
    /// CUDA stream ID.
    pub stream_id: u32,
    /// Channel ID for the work submission channel.
    pub channel_id: u32,
    /// Channel type (compute, async memcpy, etc.).
    pub channel_type: u32,
}

/// Stage IID constants for render stage events.
pub const KERNEL_STAGE_IID: u64 = 1;
pub const MEMCPY_STAGE_IID: u64 = 2;
pub const MEMSET_STAGE_IID: u64 = 3;
pub const HW_QUEUE_IID_OFFSET: u64 = 1000;

impl GpuActivity for KernelActivity {
    fn start(&self) -> u64 {
        self.start
    }
    fn end(&self) -> u64 {
        self.end
    }
    fn device_id(&self) -> u32 {
        self.device_id
    }
    fn context_id(&self) -> u32 {
        self.context_id
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn channel_id(&self) -> u32 {
        self.channel_id
    }
    fn channel_type(&self) -> u32 {
        self.channel_type
    }
    fn stage_iid(&self) -> u64 {
        KERNEL_STAGE_IID
    }
    fn emit_extra_data(
        &self,
        process_id: i32,
        process_name: &str,
        emit: &mut dyn FnMut(&str, &str),
    ) {
        emit("kernel_name", &self.kernel_name);
        emit("process_id", &process_id.to_string());
        emit("process_name", process_name);
        emit("device_id", &self.device_id.to_string());
        emit("context_id", &self.context_id.to_string());
        emit("stream_id", &self.stream_id.to_string());
        emit("channel_id", &self.channel_id.to_string());
        emit("channel_type", &self.channel_type.to_string());
    }
}

/// Detailed activity information for a memory transfer operation.
pub struct MemcpyActivity {
    /// Copy direction (HtoD, DtoH, DtoD, etc.)
    pub copy_kind: u8,
    /// Number of bytes transferred.
    pub bytes: u64,
    /// Transfer start timestamp (nanoseconds).
    pub start: u64,
    /// Transfer end timestamp (nanoseconds).
    pub end: u64,
    /// CUDA device ID.
    pub device_id: u32,
    /// CUDA context ID.
    pub context_id: u32,
    /// CUDA stream ID.
    pub stream_id: u32,
    /// Channel ID for the work submission channel.
    pub channel_id: u32,
    /// Channel type.
    pub channel_type: u32,
}

impl MemcpyActivity {
    /// Returns the copy direction as a human-readable string.
    pub fn direction_string(&self) -> &'static str {
        match self.copy_kind {
            1 => "HtoD",
            2 => "DtoH",
            3 => "HtoA",
            4 => "AtoH",
            5 => "AtoA",
            6 => "AtoD",
            7 => "DtoA",
            8 => "DtoD",
            9 => "HtoH",
            10 => "PtoP",
            _ => "Unknown",
        }
    }
}

impl GpuActivity for MemcpyActivity {
    fn start(&self) -> u64 {
        self.start
    }
    fn end(&self) -> u64 {
        self.end
    }
    fn device_id(&self) -> u32 {
        self.device_id
    }
    fn context_id(&self) -> u32 {
        self.context_id
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn channel_id(&self) -> u32 {
        self.channel_id
    }
    fn channel_type(&self) -> u32 {
        self.channel_type
    }
    fn stage_iid(&self) -> u64 {
        MEMCPY_STAGE_IID
    }
    fn emit_extra_data(
        &self,
        process_id: i32,
        process_name: &str,
        emit: &mut dyn FnMut(&str, &str),
    ) {
        emit("direction", self.direction_string());
        emit("size_bytes", &self.bytes.to_string());
        emit("process_id", &process_id.to_string());
        emit("process_name", process_name);
        emit("device_id", &self.device_id.to_string());
        emit("context_id", &self.context_id.to_string());
        emit("stream_id", &self.stream_id.to_string());
        emit("channel_id", &self.channel_id.to_string());
        emit("channel_type", &self.channel_type.to_string());
    }
}

/// Detailed activity information for a memory set operation.
pub struct MemsetActivity {
    /// Value being set (typically 0 for zeroing memory).
    pub value: u32,
    /// Number of bytes being set.
    pub bytes: u64,
    /// Memory kind (device, array, etc.).
    pub memory_kind: u16,
    /// Memset start timestamp (nanoseconds).
    pub start: u64,
    /// Memset end timestamp (nanoseconds).
    pub end: u64,
    /// CUDA device ID.
    pub device_id: u32,
    /// CUDA context ID.
    pub context_id: u32,
    /// CUDA stream ID.
    pub stream_id: u32,
    /// Channel ID for the work submission channel.
    pub channel_id: u32,
    /// Channel type.
    pub channel_type: u32,
}

impl MemsetActivity {
    /// Returns the memory kind as a human-readable string.
    pub fn memory_kind_string(&self) -> &'static str {
        match self.memory_kind {
            0 => "Unknown",
            1 => "Pageable",
            2 => "Pinned",
            3 => "Device",
            4 => "Array",
            5 => "Managed",
            6 => "DeviceStatic",
            7 => "ManagedStatic",
            _ => "Unknown",
        }
    }
}

impl GpuActivity for MemsetActivity {
    fn start(&self) -> u64 {
        self.start
    }
    fn end(&self) -> u64 {
        self.end
    }
    fn device_id(&self) -> u32 {
        self.device_id
    }
    fn context_id(&self) -> u32 {
        self.context_id
    }
    fn stream_id(&self) -> u32 {
        self.stream_id
    }
    fn channel_id(&self) -> u32 {
        self.channel_id
    }
    fn channel_type(&self) -> u32 {
        self.channel_type
    }
    fn stage_iid(&self) -> u64 {
        MEMSET_STAGE_IID
    }
    fn emit_extra_data(
        &self,
        process_id: i32,
        process_name: &str,
        emit: &mut dyn FnMut(&str, &str),
    ) {
        emit("value", &format!("0x{:08X}", self.value));
        emit("size_bytes", &self.bytes.to_string());
        emit("memory_kind", self.memory_kind_string());
        emit("process_id", &process_id.to_string());
        emit("process_name", process_name);
        emit("device_id", &self.device_id.to_string());
        emit("context_id", &self.context_id.to_string());
        emit("stream_id", &self.stream_id.to_string());
        emit("channel_id", &self.channel_id.to_string());
        emit("channel_type", &self.channel_type.to_string());
    }
}

/// Records the starting index in each per-context data vector at the moment a
/// data-source consumer (instance) started.  Emission for that consumer spans
/// from these offsets to the current end of each vector.
///
/// Contexts that did not exist when the consumer started are absent from the
/// maps; an offset of 0 is used for them (emit everything, since all of their
/// data post-dates the consumer start).
#[derive(Clone, Default)]
pub struct ConsumerStartOffsets {
    pub range_info: HashMap<u32, usize>,
    pub kernel_launches: HashMap<u32, usize>,
    pub kernel_activities: HashMap<u32, usize>,
    pub memcpy_activities: HashMap<u32, usize>,
    pub memset_activities: HashMap<u32, usize>,
}

impl ConsumerStartOffsets {
    /// Snapshot the current vector lengths for all existing contexts.
    pub fn snapshot(context_data: &HashMap<u32, Box<CtxProfilerData>>) -> Self {
        let mut s = Self::default();
        for (&ctx_id, data) in context_data {
            s.range_info.insert(ctx_id, data.range_info.len());
            s.kernel_launches.insert(ctx_id, data.kernel_launches.len());
            s.kernel_activities
                .insert(ctx_id, data.kernel_activities.len());
            s.memcpy_activities
                .insert(ctx_id, data.memcpy_activities.len());
            s.memset_activities
                .insert(ctx_id, data.memset_activities.len());
        }
        s
    }
}

/// Profiling data associated with a specific CUDA context.
///
/// Handles the lifecycle of the range profiler, metric evaluator, and stores collected
/// ranges and kernel launch metadata.
pub struct CtxProfilerData {
    pub device_id: i32,
    pub num_sms: i32,
    pub warp_size: i32,
    pub max_threads_per_sm: i32,
    pub max_blocks_per_sm: i32,
    pub max_regs_per_sm: i32,
    pub max_smem_per_sm: i32,
    pub compute_capability: (i32, i32),
    pub max_num_ranges: usize,
    pub is_active: bool,
    pub counter_data_image: Vec<u8>,
    pub metric_evaluator: Option<MetricEvaluator>,
    pub range_profiler: Option<RangeProfiler>,
    pub range_info: Vec<RangeInfo>,
    pub kernel_launches: Vec<KernelLaunch>,
    pub kernel_activities: Vec<KernelActivity>,
    pub memcpy_activities: Vec<MemcpyActivity>,
    pub memset_activities: Vec<MemsetActivity>,
}

impl CtxProfilerData {
    /// Finalizes the range profiler session by stopping it, decoding counter data,
    /// and evaluating metrics. Optionally disables the profiler and clears state.
    ///
    /// Returns early if the context is not active or has no range profiler.
    pub fn finalize_profiler(&mut self, disable: bool) {
        if !self.is_active {
            return;
        }
        if let Some(rp) = &mut self.range_profiler {
            let _ = rp.stop();
            let _ = rp.decode_counter_data();
            let metric_names = rp.validated_metric_names().to_vec();
            if let Some(me) = &self.metric_evaluator {
                if let Ok(infos) = me.evaluate_all_ranges(&self.counter_data_image, &metric_names) {
                    self.range_info.extend(infos);
                }
            }
            if disable {
                let _ = rp.disable();
            }
        }
        if disable {
            self.range_profiler = None;
            self.is_active = false;
        }
    }
}

unsafe impl Send for CtxProfilerData {}
unsafe impl Sync for CtxProfilerData {}

/// Global state shared across the application.
///
/// Manages per-context profiler data, the currently active context, and global configuration.
pub struct GlobalState {
    pub context_data: HashMap<u32, Box<CtxProfilerData>>,
    pub active_ctx: Option<CUcontext>,
    pub injection_initialized: bool,
    pub config: Config,
    pub subscriber_handle: CUpti_SubscriberHandle,
    /// Start offsets per active counters consumer instance (keyed by inst_id 0-7).
    pub counters_consumers: HashMap<u32, ConsumerStartOffsets>,
    /// Start offsets per active renderstages consumer instance (keyed by inst_id 0-7).
    pub renderstages_consumers: HashMap<u32, ConsumerStartOffsets>,
    /// thread_id → thread name, captured from /proc when first seen.
    pub thread_names: HashMap<u32, String>,
}

unsafe impl Send for GlobalState {}

impl GlobalState {
    /// Advance all renderstages consumer offsets to the current vector lengths,
    /// then drain the prefix of each vector that all consumers have consumed.
    pub fn advance_and_drain_renderstage_events(&mut self) {
        // Advance all consumer offsets to current lengths.
        for offsets in self.renderstages_consumers.values_mut() {
            for (&ctx_id, data) in self.context_data.iter() {
                offsets
                    .kernel_launches
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.kernel_launches.len())
                    .or_insert(data.kernel_launches.len());
                offsets
                    .kernel_activities
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.kernel_activities.len())
                    .or_insert(data.kernel_activities.len());
                offsets
                    .memcpy_activities
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.memcpy_activities.len())
                    .or_insert(data.memcpy_activities.len());
                offsets
                    .memset_activities
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.memset_activities.len())
                    .or_insert(data.memset_activities.len());
            }
        }

        // Drain consumed prefix for each context vector.
        for (&ctx_id, data) in self.context_data.iter_mut() {
            // kernel_launches and kernel_activities are shared with counter consumers,
            // so take the minimum across BOTH consumer types to avoid draining entries
            // that counter consumers haven't yet processed.
            let min_kl = self
                .renderstages_consumers
                .values()
                .chain(self.counters_consumers.values())
                .map(|o| o.kernel_launches.get(&ctx_id).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);
            let min_ka = self
                .renderstages_consumers
                .values()
                .chain(self.counters_consumers.values())
                .map(|o| o.kernel_activities.get(&ctx_id).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);
            let min_mc = self
                .renderstages_consumers
                .values()
                .map(|o| o.memcpy_activities.get(&ctx_id).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);
            let min_ms = self
                .renderstages_consumers
                .values()
                .map(|o| o.memset_activities.get(&ctx_id).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);

            if min_kl > 0 {
                data.kernel_launches.drain(..min_kl);
                for offsets in self.renderstages_consumers.values_mut() {
                    if let Some(o) = offsets.kernel_launches.get_mut(&ctx_id) {
                        *o -= min_kl;
                    }
                }
                // Also adjust counters consumer offsets for kernel_launches.
                for offsets in self.counters_consumers.values_mut() {
                    if let Some(o) = offsets.kernel_launches.get_mut(&ctx_id) {
                        *o = o.saturating_sub(min_kl);
                    }
                }
            }
            if min_ka > 0 {
                data.kernel_activities.drain(..min_ka);
                for offsets in self.renderstages_consumers.values_mut() {
                    if let Some(o) = offsets.kernel_activities.get_mut(&ctx_id) {
                        *o -= min_ka;
                    }
                }
                // Also adjust counters consumer offsets for kernel_activities.
                for offsets in self.counters_consumers.values_mut() {
                    if let Some(o) = offsets.kernel_activities.get_mut(&ctx_id) {
                        *o = o.saturating_sub(min_ka);
                    }
                }
            }
            if min_mc > 0 {
                data.memcpy_activities.drain(..min_mc);
                for offsets in self.renderstages_consumers.values_mut() {
                    if let Some(o) = offsets.memcpy_activities.get_mut(&ctx_id) {
                        *o -= min_mc;
                    }
                }
            }
            if min_ms > 0 {
                data.memset_activities.drain(..min_ms);
                for offsets in self.renderstages_consumers.values_mut() {
                    if let Some(o) = offsets.memset_activities.get_mut(&ctx_id) {
                        *o -= min_ms;
                    }
                }
            }
        }
    }

    /// Advance all counters consumer offsets to the current vector lengths,
    /// then drain the prefix of each vector that all consumers have consumed.
    pub fn advance_and_drain_counter_events(&mut self) {
        // Advance all consumer offsets to current lengths.
        for offsets in self.counters_consumers.values_mut() {
            for (&ctx_id, data) in self.context_data.iter() {
                offsets
                    .range_info
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.range_info.len())
                    .or_insert(data.range_info.len());
                offsets
                    .kernel_launches
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.kernel_launches.len())
                    .or_insert(data.kernel_launches.len());
                offsets
                    .kernel_activities
                    .entry(ctx_id)
                    .and_modify(|o| *o = data.kernel_activities.len())
                    .or_insert(data.kernel_activities.len());
            }
        }

        // Drain consumed prefix for each context vector.
        for (&ctx_id, data) in self.context_data.iter_mut() {
            let min_ri = self
                .counters_consumers
                .values()
                .map(|o| o.range_info.get(&ctx_id).copied().unwrap_or(0))
                .min()
                .unwrap_or(0);

            if min_ri > 0 {
                data.range_info.drain(..min_ri);
                for offsets in self.counters_consumers.values_mut() {
                    if let Some(o) = offsets.range_info.get_mut(&ctx_id) {
                        *o -= min_ri;
                    }
                }
            }

            // kernel_launches and kernel_activities are shared with renderstages,
            // so only drain if no renderstages consumers exist (otherwise
            // advance_and_drain_renderstage_events handles them).
            if self.renderstages_consumers.is_empty() {
                let min_kl = self
                    .counters_consumers
                    .values()
                    .map(|o| o.kernel_launches.get(&ctx_id).copied().unwrap_or(0))
                    .min()
                    .unwrap_or(0);
                let min_ka = self
                    .counters_consumers
                    .values()
                    .map(|o| o.kernel_activities.get(&ctx_id).copied().unwrap_or(0))
                    .min()
                    .unwrap_or(0);

                if min_kl > 0 {
                    data.kernel_launches.drain(..min_kl);
                    for offsets in self.counters_consumers.values_mut() {
                        if let Some(o) = offsets.kernel_launches.get_mut(&ctx_id) {
                            *o -= min_kl;
                        }
                    }
                }
                if min_ka > 0 {
                    data.kernel_activities.drain(..min_ka);
                    for offsets in self.counters_consumers.values_mut() {
                        if let Some(o) = offsets.kernel_activities.get_mut(&ctx_id) {
                            *o -= min_ka;
                        }
                    }
                }
            }
        }
    }
}

/// The singleton global state instance.
///
/// Protected by a Mutex to ensure thread-safe access from callback handlers.
pub static GLOBAL_STATE: Lazy<Mutex<GlobalState>> = Lazy::new(|| {
    Mutex::new(GlobalState {
        context_data: HashMap::new(),
        active_ctx: None,
        injection_initialized: false,
        config: Config::default(),
        subscriber_handle: std::ptr::null_mut(),
        counters_consumers: HashMap::new(),
        renderstages_consumers: HashMap::new(),
        thread_names: HashMap::new(),
    })
});
