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

/// Default NVIDIA CUPTI metrics to collect if none are specified via environment variable.
///
/// These metrics are selected to provide a broad overview of GPU performance,
/// covering compute usage, memory bandwidth, cache efficiency, and instruction throughput.
pub const DEFAULT_METRICS: &[&str] = &[
    "gpu__time_duration.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.avg",
    "sm__cycles_elapsed.avg",
    "gpc__cycles_elapsed.avg.per_second",
    "gpc__cycles_elapsed.max",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__cycles_elapsed.avg.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_elapsed",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
];

/// Parses a comma or semicolon separated string of metrics.
///
/// If input is empty or whitespace-only, returns `DEFAULT_METRICS`.
pub fn parse_metrics(input: &str) -> Vec<String> {
    perfetto_gpu_compute_injection::config::parse_metrics(input, DEFAULT_METRICS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics_empty() {
        let metrics = parse_metrics("");
        assert_eq!(metrics.len(), DEFAULT_METRICS.len());
        assert_eq!(metrics[0], "gpu__time_duration.sum");
    }
}
