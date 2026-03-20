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

use once_cell::sync::Lazy;
use perfetto_gpu_compute_injection::config::Config;
use std::{collections::HashMap, sync::Mutex};

/// A captured set of counter values for a single kernel dispatch.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CounterResult {
    pub kernel_name: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub device_index: i32,
    pub gpu_id: u32,
    pub values: Vec<f64>, // values in same order as counter_names
}

/// Snapshot of counter_results length when a Perfetto counter consumer starts.
#[derive(Debug, Clone, Default)]
pub struct CounterConsumerStartOffsets {
    pub counter_results: usize,
}

/// A captured kernel dispatch event from rocprofiler buffer tracing.
#[derive(Debug, Clone)]
pub struct KernelDispatch {
    pub kernel_name: String,
    pub grid: (u32, u32, u32),
    pub workgroup: (u32, u32, u32),
    pub start_ns: u64,
    pub end_ns: u64,
    /// `rocprofiler_queue_id_t.handle` of the dispatch queue.
    pub queue_handle: u64,
    /// `logical_node_type_id` (0-based GPU index) for this agent.
    pub device_index: i32,
    /// Pseudo GPU-ID derived from the agent handle (for Perfetto gpu_id field).
    pub gpu_id: u32,
    /// GPU architecture name (e.g. `gfx90a`).
    pub arch: String,
}

/// A captured memory copy event.
#[derive(Debug, Clone)]
pub struct MemcopyActivity {
    pub bytes: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub device_index: i32,
    pub gpu_id: u32,
    /// Memory copy direction (raw `rocprofiler_memory_copy_operation_t` value).
    pub direction: i32,
}

/// Snapshot of buffer lengths taken when a Perfetto consumer starts.
/// Used to emit only events recorded after the consumer started.
#[derive(Debug, Clone, Default)]
pub struct ConsumerStartOffsets {
    pub kernel_dispatches: usize,
    pub memcopies: usize,
}

impl ConsumerStartOffsets {
    pub fn snapshot(state: &GlobalState) -> Self {
        Self {
            kernel_dispatches: state.kernel_dispatches.len(),
            memcopies: state.memcopies.len(),
        }
    }
}

pub struct GlobalState {
    /// Captured kernel dispatch events (grows throughout the trace).
    pub kernel_dispatches: Vec<KernelDispatch>,
    /// Captured memory copy events.
    pub memcopies: Vec<MemcopyActivity>,
    /// Captured counter results from dispatch counting callbacks.
    pub counter_results: Vec<CounterResult>,
    /// kernel_id → kernel_name, populated by code object callbacks.
    pub kernel_names: HashMap<u64, String>,
    /// agent_id.handle → (logical_node_type_id, arch name) for GPU agents.
    pub agents: HashMap<u64, (i32, String)>,
    /// agent_id.handle → counter_config_id.handle (per-agent counter configs).
    pub counter_configs: HashMap<u64, u64>,
    /// Ordered list of configured counter names (same order as CounterResult.values).
    pub counter_names: Vec<String>,
    /// counter_id.handle → index in counter_names.
    pub counter_id_to_index: HashMap<u64, usize>,
    /// Perfetto tracing context (started on first consumer, stopped on last).
    pub tracing_context: Option<u64>, // rocprofiler_context_id_t.handle
    /// Buffer used for kernel dispatch and memory copy records.
    pub tracing_buffer: Option<u64>, // rocprofiler_buffer_id_t.handle
    /// Utility context (always active) for code object callbacks.
    pub utility_context: Option<u64>, // rocprofiler_context_id_t.handle
    /// Per-instance start offsets for renderstages consumers.
    pub renderstages_consumers: HashMap<u32, ConsumerStartOffsets>,
    /// Per-instance start offsets for counters consumers.
    pub counters_consumers: HashMap<u32, CounterConsumerStartOffsets>,
    pub config: Config,
    pub initialized: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for GlobalState {
    fn default() -> Self {
        Self {
            kernel_dispatches: Vec::new(),
            memcopies: Vec::new(),
            counter_results: Vec::new(),
            kernel_names: HashMap::new(),
            agents: HashMap::new(),
            counter_configs: HashMap::new(),
            counter_names: Vec::new(),
            counter_id_to_index: HashMap::new(),
            tracing_context: None,
            tracing_buffer: None,
            utility_context: None,
            renderstages_consumers: HashMap::new(),
            counters_consumers: HashMap::new(),
            config: Config::default(),
            initialized: false,
        }
    }
}

pub static GLOBAL_STATE: Lazy<Mutex<GlobalState>> =
    Lazy::new(|| Mutex::new(GlobalState::default()));
