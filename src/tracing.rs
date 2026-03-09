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

use crate::injection_log;
use libc::{clock_gettime, timespec};
use perfetto_sdk::data_source::{
    DataSource, DataSourceArgsBuilder, DataSourceBufferExhaustedPolicy,
};
use std::{
    env,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering},
        OnceLock,
    },
};

#[cfg(target_os = "linux")]
use libc::CLOCK_BOOTTIME as TRACE_TIME_CLOCK;
#[cfg(target_os = "macos")]
use libc::CLOCK_MONOTONIC as TRACE_TIME_CLOCK;

/// Monotonically increasing counter for trace event IDs.
pub static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// Returns the next unique event ID for tracing events.
pub fn get_next_event_id() -> u64 {
    NEXT_EVENT_ID.fetch_add(1, Ordering::SeqCst)
}

/// Tracks whether the first counters have been received for a given data source instance.
pub static GOT_FIRST_COUNTERS: AtomicU8 = AtomicU8::new(0);

/// Tracks whether the first renderstages have been emitted for a given data source instance.
pub static GOT_FIRST_RENDERSTAGES: AtomicU8 = AtomicU8::new(0);

/// Atomic state tracking for whether counters data source is enabled.
static COUNTERS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Atomic state tracking for whether renderstages data source is enabled.
static RENDERSTAGES_ENABLED: AtomicBool = AtomicBool::new(false);

static GPU_COUNTERS_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();
static GPU_RENDERSTAGES_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();

static DATA_SOURCE_NAME_SUFFIX: OnceLock<String> = OnceLock::new();
static COUNTERS_DATA_SOURCE_NAME: OnceLock<String> = OnceLock::new();
static RENDERSTAGES_DATA_SOURCE_NAME: OnceLock<String> = OnceLock::new();

/// Default suffix for data source names.
/// Since this library uses CUPTI (CUDA Profiling Tools Interface), which is
/// exclusively for NVIDIA GPUs, the default vendor suffix is "nv".
const DEFAULT_DATA_SOURCE_NAME_SUFFIX: &str = "nv";

/// Returns the data source name suffix, reading from `INJECTION_DATA_SOURCE_NAME_SUFFIX` env var or using default.
fn get_data_source_name_suffix() -> &'static str {
    DATA_SOURCE_NAME_SUFFIX.get_or_init(|| {
        env::var("INJECTION_DATA_SOURCE_NAME_SUFFIX")
            .unwrap_or_else(|_| DEFAULT_DATA_SOURCE_NAME_SUFFIX.to_string())
    })
}

/// Returns the counters data source name in the format `gpu.counters.SUFFIX`.
fn get_counters_data_source_name() -> &'static str {
    COUNTERS_DATA_SOURCE_NAME
        .get_or_init(|| format!("gpu.counters.{}", get_data_source_name_suffix()))
}

/// Returns the renderstages data source name in the format `gpu.renderstages.SUFFIX`.
fn get_renderstages_data_source_name() -> &'static str {
    RENDERSTAGES_DATA_SOURCE_NAME
        .get_or_init(|| format!("gpu.renderstages.{}", get_data_source_name_suffix()))
}

/// Initializes and retrieves the static Perfetto counters data source.
///
/// This function is thread-safe and ensures the data source is registered only once.
/// The data source name suffix can be overridden via the `INJECTION_DATA_SOURCE_NAME_SUFFIX`
/// environment variable (default: "nv", resulting in "gpu.counters.nv").
pub fn get_counters_data_source() -> &'static DataSource<'static> {
    GPU_COUNTERS_DATA_SOURCE.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .buffer_exhausted_policy(DataSourceBufferExhaustedPolicy::StallAndAbort)
            .on_start(move |inst_id, _| {
                GOT_FIRST_COUNTERS.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                COUNTERS_ENABLED.store(true, Ordering::SeqCst);
                injection_log!("counters data source started");
            })
            .on_stop(move |_inst_id, _| {
                COUNTERS_ENABLED.store(false, Ordering::SeqCst);
            });
        let mut data_source = DataSource::new();
        data_source
            .register(get_counters_data_source_name(), data_source_args.build())
            .expect("failed to register counters data source");
        data_source
    })
}

/// Initializes and retrieves the static Perfetto renderstages data source.
///
/// This function is thread-safe and ensures the data source is registered only once.
/// The data source name suffix can be overridden via the `INJECTION_DATA_SOURCE_NAME_SUFFIX`
/// environment variable (default: "nv", resulting in "gpu.renderstages.nv").
pub fn get_renderstages_data_source() -> &'static DataSource<'static> {
    GPU_RENDERSTAGES_DATA_SOURCE.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .buffer_exhausted_policy(DataSourceBufferExhaustedPolicy::StallAndAbort)
            .on_start(move |inst_id, _| {
                GOT_FIRST_RENDERSTAGES.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                RENDERSTAGES_ENABLED.store(true, Ordering::SeqCst);
                injection_log!("renderstages data source started");
            })
            .on_stop(move |_inst_id, _| {
                RENDERSTAGES_ENABLED.store(false, Ordering::SeqCst);
            });
        let mut data_source = DataSource::new();
        data_source
            .register(
                get_renderstages_data_source_name(),
                data_source_args.build(),
            )
            .expect("failed to register renderstages data source");
        data_source
    })
}

/// Checks if counters collection is active.
pub fn is_counters_enabled() -> bool {
    COUNTERS_ENABLED.load(Ordering::SeqCst)
}

/// Returns the current timestamp in nanoseconds from the trace clock.
///
/// Uses `CLOCK_BOOTTIME` on Linux and `CLOCK_MONOTONIC` on macOS.
pub fn trace_time_ns() -> u64 {
    let mut ts = timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ret = unsafe { clock_gettime(TRACE_TIME_CLOCK, &mut ts) };
    if ret != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000u64 + (ts.tv_nsec as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_next_event_id() {
        let id1 = get_next_event_id();
        let id2 = get_next_event_id();
        assert_eq!(id2, id1 + 1);
        assert!(id1 > 0);
    }

    #[test]
    fn test_counters_enabled_default() {
        assert!(!is_counters_enabled());
    }
}
