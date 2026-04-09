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

/// Default AMD GPU hardware counter names to collect if none are specified via environment variable.
///
/// These are raw rocprofiler counter names available on most RDNA/CDNA architectures.
pub const DEFAULT_METRICS: &[&str] = &[
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
    "SQ_INSTS_FLAT",
    "SQ_INSTS_GDS",
    "GRBM_GUI_ACTIVE_avr",
    "SQ_BUSY_CYCLES_avr",
    "TA_BUSY_avr",
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
];

/// Synthetic counter names computed from hardware counters and agent
/// properties. These are appended to counter_names after the hardware
/// counters and have their values computed in record_counting_callback.
pub const SYNTHETIC_COUNTERS: &[&str] = &["GRBM_TIME_DUR_max"];

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
        assert_eq!(metrics[0], "SQ_WAVES");
    }
}
