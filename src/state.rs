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

use crate::config::Config;
use cupti_profiler::bindings::*;
use cupti_profiler::*;
use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};

/// Represents a specific kernel launch event.
pub struct KernelLaunch {
    pub function: CUfunction,
    /// Trace timestamp captured just before range profiler push (when counters enabled).
    pub start: u64,
    /// Trace timestamp captured after range profiler pop completes all passes (when counters enabled).
    pub end: u64,
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
    /// CUDA context ID.
    pub context_id: u32,
    /// Channel ID for the work submission channel.
    pub channel_id: u32,
    /// Channel type (compute, async memcpy, etc.).
    pub channel_type: u32,
}

/// Profiling data associated with a specific CUDA context.
///
/// Handles the lifecycle of the range profiler, metric evaluator, and stores collected
/// ranges and kernel launch metadata.
pub struct CtxProfilerData {
    pub device_id: i32,
    pub num_sms: i32,
    pub max_num_ranges: usize,
    pub is_active: bool,
    pub counter_data_image: Vec<u8>,
    pub metric_evaluator: Option<MetricEvaluator>,
    pub range_profiler: Option<RangeProfiler>,
    pub range_info: Vec<RangeInfo>,
    pub kernel_launches: Vec<KernelLaunch>,
    pub kernel_activities: Vec<KernelActivity>,
}

impl CtxProfilerData {
    /// Finalizes the range profiler session by stopping it, decoding counter data,
    /// and evaluating metrics. Optionally disables the profiler and clears state.
    ///
    /// Returns early if the context is not active or has no range profiler.
    pub fn finalize_profiler(&mut self, metric_names: &[String], disable: bool) {
        if !self.is_active {
            return;
        }
        if let Some(rp) = &mut self.range_profiler {
            let _ = rp.stop();
            let _ = rp.decode_counter_data();
            if let Some(me) = &self.metric_evaluator {
                if let Ok(infos) = me.evaluate_all_ranges(&self.counter_data_image, metric_names) {
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
}

unsafe impl Send for GlobalState {}

/// The singleton global state instance.
///
/// Protected by a Mutex to ensure thread-safe access from callback handlers.
pub static GLOBAL_STATE: Lazy<Mutex<GlobalState>> = Lazy::new(|| {
    Mutex::new(GlobalState {
        context_data: HashMap::new(),
        active_ctx: None,
        injection_initialized: false,
        config: Config::default(),
    })
});
