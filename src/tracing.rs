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

use crate::config::trace_startup_has;
use crate::injection_log;
use libc::{clock_gettime, timespec};
use perfetto_sdk::data_source::{
    DataSource, DataSourceArgsBuilder, DataSourceBufferExhaustedPolicy,
};
use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Condvar, Mutex, OnceLock,
};
use std::time::Duration;

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

/// Bitmask of currently active counters data source instances (one bit per inst_id 0-7).
/// Non-zero means at least one counters consumer is running.
static COUNTERS_ACTIVE_MASK: AtomicU8 = AtomicU8::new(0);

/// Bitmask of currently active renderstages data source instances (one bit per inst_id 0-7).
/// Non-zero means at least one renderstages consumer is running.
static RENDERSTAGES_ACTIVE_MASK: AtomicU8 = AtomicU8::new(0);

/// Condvar flag for waiting on counters data source to start.
static COUNTERS_STARTED: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Condvar flag for waiting on renderstages data source to start.
static RENDERSTAGES_STARTED: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

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
        std::env::var("INJECTION_DATA_SOURCE_NAME_SUFFIX")
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

/// Waits for a `(Mutex<bool>, Condvar)` flag to become true, with a 30s timeout.
/// Logs a warning every 10s while waiting. Exits the process on timeout.
fn wait_for_start(flag: &(Mutex<bool>, Condvar), label: &str) {
    injection_log!("waiting for {} to start...", label);
    let (lock, cvar) = flag;
    let mut guard = lock.lock().expect("mutex poisoned");
    for attempt in 1..=3 {
        let (g, _timeout) = cvar
            .wait_timeout_while(guard, Duration::from_secs(10), |started| !*started)
            .expect("mutex poisoned");
        guard = g;
        if *guard {
            injection_log!("{} started", label);
            return;
        }
        if attempt < 3 {
            injection_log!(
                "WARNING: {} not started after {}s, still waiting...",
                label,
                attempt * 10
            );
        } else {
            injection_log!("ERROR: {} never started (timed out after 30s)", label);
            std::process::exit(1);
        }
    }
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
            .will_notify_on_stop(true)
            .on_start(move |inst_id, _| {
                GOT_FIRST_COUNTERS.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                let prev = COUNTERS_ACTIVE_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                // Snapshot start offsets before any kernels are recorded for this instance.
                crate::register_counters_consumer(inst_id);
                if prev == 0 && !is_renderstages_enabled() {
                    // Transitioning from no consumers at all: enable base CUPTI activities.
                    crate::on_first_consumer_start();
                }
                if prev == 0 {
                    // First counters consumer: disable MEMSET to avoid Range Profiler conflict.
                    crate::on_first_counters_start();
                }
                injection_log!("counters data source started (instance {})", inst_id);
                let (lock, cvar) = &COUNTERS_STARTED;
                *lock.lock().expect("mutex poisoned") = true;
                cvar.notify_all();
            })
            .on_stop(move |inst_id, args| {
                let stop_guard = args.postpone();
                let prev = COUNTERS_ACTIVE_MASK.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                let remaining = prev & !(1 << inst_id);
                if remaining == 0 && is_renderstages_enabled() {
                    // Last counters consumer stopped while renderstages still runs:
                    // re-enable MEMSET now that there is no Range Profiler conflict.
                    crate::on_last_counters_stop();
                }
                injection_log!(
                    "counters data source stopping (instance {}), emitting buffered events...",
                    inst_id
                );
                std::thread::spawn(move || {
                    if !is_counters_enabled() && !is_renderstages_enabled() {
                        // All consumers gone: full CUPTI teardown (flush + finalize).
                        crate::run_cupti_teardown();
                    } else if !is_counters_enabled() {
                        // Last counters consumer stopped; renderstages still running.
                        // Finalize the range profiler without disabling activities.
                        crate::finalize_range_profiler();
                    }
                    // else: other counters consumers still running; no finalization needed.
                    crate::emit_counter_events_for_instance(inst_id, Some(stop_guard));
                });
            });
        let mut data_source = DataSource::new();
        let ds_name = get_counters_data_source_name();
        data_source
            .register(ds_name, data_source_args.build())
            .expect("failed to register counters data source");

        if trace_startup_has(ds_name) {
            wait_for_start(&COUNTERS_STARTED, ds_name);
        }

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
            .will_notify_on_stop(true)
            .on_start(move |inst_id, _| {
                GOT_FIRST_RENDERSTAGES.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                let prev = RENDERSTAGES_ACTIVE_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                crate::register_renderstages_consumer(inst_id);
                if prev == 0 && !is_counters_enabled() {
                    // First consumer of any type AND no counters: enable activities + MEMSET.
                    crate::on_first_consumer_start();
                    crate::on_renderstages_start_no_counters();
                }
                // If counters is already running: activities are already enabled and MEMSET
                // is already disabled — nothing extra to do.
                injection_log!("renderstages data source started (instance {})", inst_id);
                let (lock, cvar) = &RENDERSTAGES_STARTED;
                *lock.lock().expect("mutex poisoned") = true;
                cvar.notify_all();
            })
            .on_stop(move |inst_id, args| {
                let stop_guard = args.postpone();
                RENDERSTAGES_ACTIVE_MASK.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                injection_log!(
                    "renderstages data source stopping (instance {}), emitting buffered events...",
                    inst_id
                );
                std::thread::spawn(move || {
                    if !is_counters_enabled() && !is_renderstages_enabled() {
                        // All consumers gone: full teardown.
                        crate::run_cupti_teardown();
                    } else {
                        // Other consumers still running: flush activity buffers so this
                        // instance sees all pending records without disabling activities.
                        crate::flush_activity_buffers();
                    }
                    crate::emit_renderstage_events_for_instance(inst_id, Some(stop_guard));
                });
            });
        let mut data_source = DataSource::new();
        let ds_name = get_renderstages_data_source_name();
        data_source
            .register(ds_name, data_source_args.build())
            .expect("failed to register renderstages data source");

        if trace_startup_has(ds_name) {
            wait_for_start(&RENDERSTAGES_STARTED, ds_name);
        }

        data_source
    })
}

/// Returns true if at least one counters data source consumer is currently active.
pub fn is_counters_enabled() -> bool {
    COUNTERS_ACTIVE_MASK.load(Ordering::SeqCst) != 0
}

/// Returns true if at least one renderstages data source consumer is currently active.
pub fn is_renderstages_enabled() -> bool {
    RENDERSTAGES_ACTIVE_MASK.load(Ordering::SeqCst) != 0
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
