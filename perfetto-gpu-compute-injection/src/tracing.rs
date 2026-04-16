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
    DataSource, DataSourceArgsBuilder, DataSourceBufferExhaustedPolicy, StopGuard,
};
use perfetto_sdk::pb_decoder::{PbDecoder, PbDecoderField};
use perfetto_sdk_protos_gpu::protos::config::data_source_config::DataSourceConfigExtFieldNumber;
use perfetto_sdk_protos_gpu::protos::config::gpu::gpu_counter_config::GpuCounterConfigFieldNumber;
use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Condvar, Mutex, OnceLock,
};
use std::time::Duration;

#[cfg(target_os = "linux")]
use libc::CLOCK_BOOTTIME as TRACE_TIME_CLOCK;
#[cfg(target_os = "macos")]
use libc::CLOCK_MONOTONIC as TRACE_TIME_CLOCK;

// ---------------------------------------------------------------------------
// GpuBackend trait
// ---------------------------------------------------------------------------

/// Abstraction over NVIDIA and AMD GPU tracing backends.
///
/// Each backend crate (`perfetto-cuda-injection`, `perfetto-hip-injection`) provides
/// one implementation and registers it at startup via [`register_backend`].
pub trait GpuBackend: Send + Sync {
    /// Short suffix used in data source names, e.g. `"nv"` or `"amd"`.
    fn default_data_source_suffix(&self) -> &'static str;

    /// Called when the first consumer of any type (counters or renderstages) starts.
    fn on_first_consumer_start(&self);

    /// Called when a renderstages consumer starts and no counters consumer is active.
    fn on_renderstages_start_no_counters(&self);

    /// Full teardown when all consumers are gone.
    fn run_teardown(&self);

    /// Flush buffered GPU activity records to the in-memory store.
    fn flush_activity_buffers(&self);

    /// Emit renderstage Perfetto events for one consumer instance and drop the stop guard.
    fn emit_renderstage_events_for_instance(&self, inst_id: u32, stop_guard: Option<StopGuard>);

    /// Snapshot start offsets for a renderstages consumer instance.
    fn register_renderstages_consumer(&self, inst_id: u32);

    // ----- NVIDIA counters-only hooks — default to no-ops for AMD -----

    /// Called when the first counters consumer starts.
    fn on_first_counters_start(&self) {}

    /// Called when the last counters consumer stops (while renderstages is still running).
    fn on_last_counters_stop(&self) {}

    /// Emit counter Perfetto events for one consumer instance and drop the stop guard.
    fn emit_counter_events_for_instance(&self, _inst_id: u32, _stop_guard: Option<StopGuard>) {}

    /// Snapshot start offsets for a counters consumer instance.
    fn register_counters_consumer(&self, _inst_id: u32) {}

    /// Finalize the range profiler without disabling activities.
    fn finalize_range_profiler(&self) {}

    /// Flush and emit renderstage events for all active instances (called on periodic flush).
    fn flush_renderstage_events(&self) {}

    /// Flush and emit counter events for all active instances (called on periodic flush).
    fn flush_counter_events(&self) {}
}

static BACKEND: OnceLock<Box<dyn GpuBackend + Send + Sync>> = OnceLock::new();

/// Register the GPU backend. Must be called exactly once before any data source is used.
pub fn register_backend<B: GpuBackend + Send + Sync + 'static>(backend: B) {
    let _ = BACKEND.set(Box::new(backend));
}

fn backend() -> &'static dyn GpuBackend {
    &**BACKEND.get().expect("no GPU backend registered")
}

// ---------------------------------------------------------------------------
// Shared tracing state
// ---------------------------------------------------------------------------

/// Monotonically increasing counter for trace event IDs.
pub static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

/// Returns the next unique event ID for tracing events.
pub fn get_next_event_id() -> u64 {
    NEXT_EVENT_ID.fetch_add(1, Ordering::SeqCst)
}

/// Bitmask of currently active counters data source instances (one bit per inst_id 0-7).
/// Non-zero means at least one counters consumer is running.
/// Always stays 0 for AMD since the AMD backend never registers the counters data source.
static COUNTERS_ACTIVE_MASK: AtomicU8 = AtomicU8::new(0);

/// Bitmask of currently active renderstages data source instances (one bit per inst_id 0-7).
/// Non-zero means at least one renderstages consumer is running.
static RENDERSTAGES_ACTIVE_MASK: AtomicU8 = AtomicU8::new(0);

/// Bitmask of instances whose config had `instrumented_sampling: true` (set in on_setup).
static INSTRUMENTED_SETUP_MASK: AtomicU8 = AtomicU8::new(0);

/// Bitmask of currently active instances with instrumented sampling enabled.
/// A subset of `COUNTERS_ACTIVE_MASK` — only instances where the consumer requested
/// `instrumented_sampling: true` in `GpuCounterConfig`.
static INSTRUMENTED_ACTIVE_MASK: AtomicU8 = AtomicU8::new(0);

/// Condvar flag for waiting on counters data source to start.
static COUNTERS_STARTED: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Condvar flag for waiting on renderstages data source to start.
static RENDERSTAGES_STARTED: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

static GPU_COUNTERS_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();
static GPU_RENDERSTAGES_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();

static DATA_SOURCE_NAME_SUFFIX: OnceLock<String> = OnceLock::new();
static COUNTERS_DATA_SOURCE_NAME: OnceLock<String> = OnceLock::new();
static RENDERSTAGES_DATA_SOURCE_NAME: OnceLock<String> = OnceLock::new();

/// Returns the data source name suffix, reading from `INJECTION_DATA_SOURCE_NAME_SUFFIX`
/// env var or using the registered backend's default.
fn get_data_source_name_suffix() -> &'static str {
    DATA_SOURCE_NAME_SUFFIX.get_or_init(|| {
        std::env::var("INJECTION_DATA_SOURCE_NAME_SUFFIX")
            .unwrap_or_else(|_| backend().default_data_source_suffix().to_string())
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
/// environment variable (default determined by the registered backend).
pub fn get_counters_data_source() -> &'static DataSource<'static> {
    GPU_COUNTERS_DATA_SOURCE.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .buffer_exhausted_policy(DataSourceBufferExhaustedPolicy::StallAndAbort)
            .will_notify_on_stop(true)
            .on_setup(move |inst_id, config, _| {
                // Parse DataSourceConfig for gpu_counter_config, then instrumented_sampling.
                const GPU_COUNTER_CONFIG_FIELD: u32 =
                    DataSourceConfigExtFieldNumber::GpuCounterConfig as u32;
                const INSTRUMENTED_SAMPLING_FIELD: u32 =
                    GpuCounterConfigFieldNumber::InstrumentedSampling as u32;
                let mut instrumented = false;
                for item in PbDecoder::new(config) {
                    if let Ok((
                        GPU_COUNTER_CONFIG_FIELD,
                        PbDecoderField::Delimited(gpu_counter_bytes),
                    )) = item
                    {
                        for sub_item in PbDecoder::new(gpu_counter_bytes) {
                            if let Ok((INSTRUMENTED_SAMPLING_FIELD, PbDecoderField::Varint(v))) =
                                sub_item
                            {
                                instrumented = v != 0;
                            }
                        }
                    }
                }
                if instrumented {
                    INSTRUMENTED_SETUP_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                } else {
                    INSTRUMENTED_SETUP_MASK.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                }
                injection_log!(
                    "counters data source setup (instance {}): instrumented_sampling={}",
                    inst_id,
                    instrumented
                );
            })
            .on_start(move |inst_id, _| {
                let prev = COUNTERS_ACTIVE_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                // Snapshot start offsets before any kernels are recorded for this instance.
                backend().register_counters_consumer(inst_id);
                if prev == 0 && !is_renderstages_enabled() {
                    // Transitioning from no consumers at all: enable base activities.
                    backend().on_first_consumer_start();
                }
                let instrumented =
                    INSTRUMENTED_SETUP_MASK.load(Ordering::SeqCst) & (1 << inst_id) != 0;
                if instrumented {
                    let prev_instr =
                        INSTRUMENTED_ACTIVE_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                    if prev_instr == 0 {
                        // First instrumented consumer: configure range profiling.
                        backend().on_first_counters_start();
                    }
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
                let was_instrumented = INSTRUMENTED_ACTIVE_MASK
                    .fetch_and(!(1 << inst_id), Ordering::SeqCst)
                    & (1 << inst_id)
                    != 0;
                INSTRUMENTED_SETUP_MASK.fetch_and(!(1 << inst_id), Ordering::SeqCst);
                if remaining == 0 && is_renderstages_enabled() && was_instrumented {
                    // Last counters consumer stopped while renderstages still runs:
                    // re-enable MEMSET now that there is no Range Profiler conflict.
                    backend().on_last_counters_stop();
                }
                injection_log!(
                    "counters data source stopping (instance {}), emitting buffered events...",
                    inst_id
                );
                std::thread::spawn(move || {
                    if !is_counters_enabled() && !is_renderstages_enabled() {
                        // All consumers gone: full teardown (flush + finalize).
                        backend().run_teardown();
                    } else if !is_counters_enabled() && !is_instrumented_enabled() {
                        // Last counters consumer stopped; renderstages still running.
                        // Only finalize range profiler if instrumented sampling was active.
                        if was_instrumented {
                            backend().finalize_range_profiler();
                        }
                    }
                    // else: other counters consumers still running; no finalization needed.
                    backend().emit_counter_events_for_instance(inst_id, Some(stop_guard));
                });
            })
            .on_flush(move |inst_id, args| {
                injection_log!(
                    "counters data source flush requested (instance {})",
                    inst_id
                );
                let flush_guard = args.postpone();
                std::thread::spawn(move || {
                    backend().flush_counter_events();
                    injection_log!("counters data source flush complete (instance {})", inst_id);
                    drop(flush_guard);
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
/// environment variable (default determined by the registered backend).
pub fn get_renderstages_data_source() -> &'static DataSource<'static> {
    GPU_RENDERSTAGES_DATA_SOURCE.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .buffer_exhausted_policy(DataSourceBufferExhaustedPolicy::StallAndAbort)
            .will_notify_on_stop(true)
            .on_start(move |inst_id, _| {
                let prev = RENDERSTAGES_ACTIVE_MASK.fetch_or(1 << inst_id, Ordering::SeqCst);
                backend().register_renderstages_consumer(inst_id);
                if prev == 0 && !is_counters_enabled() {
                    // First consumer of any type AND no counters: enable activities + MEMSET.
                    backend().on_first_consumer_start();
                    backend().on_renderstages_start_no_counters();
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
                        backend().run_teardown();
                    } else {
                        // Other consumers still running: flush activity buffers so this
                        // instance sees all pending records without disabling activities.
                        backend().flush_activity_buffers();
                    }
                    backend().emit_renderstage_events_for_instance(inst_id, Some(stop_guard));
                });
            })
            .on_flush(move |inst_id, args| {
                injection_log!(
                    "renderstages data source flush requested (instance {})",
                    inst_id
                );
                let flush_guard = args.postpone();
                std::thread::spawn(move || {
                    backend().flush_renderstage_events();
                    injection_log!(
                        "renderstages data source flush complete (instance {})",
                        inst_id
                    );
                    drop(flush_guard);
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

/// Returns true if at least one counters consumer with `instrumented_sampling: true` is active.
///
/// When false, the backends should skip range profiling / dispatch counting to avoid overhead.
pub fn is_instrumented_enabled() -> bool {
    INSTRUMENTED_ACTIVE_MASK.load(Ordering::SeqCst) != 0
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

/// Builds a thread-track descriptor for API call track events.
///
/// Creates local bindings for `TrackEventProtoField` arrays and a
/// `TrackEventProtoTrack` that borrows them.  The track `uuid` is
/// `process_uuid ^ thread_id`.
///
/// # Parameters
///
/// * `$process_uuid` — the process track UUID (`TrackEventTrack::process_track_uuid()`)
/// * `$process_id`   — numeric pid (as `u64`)
/// * `$thread_id`    — numeric tid (as `u64`)
/// * `$thread_name`  — `Option<&str>` thread name
/// * `$thread_fields_named`, `$thread_fields_unnamed`, `$track_fields`, `$thread_track`
///   — identifiers for the local bindings that will be created
///
/// After the macro expands, `$thread_track` is a `TrackEventProtoTrack` you
/// can pass to `ctx.set_proto_track()`.
#[macro_export]
macro_rules! build_thread_track {
    (
        process_uuid: $process_uuid:expr,
        process_id: $process_id:expr,
        thread_id: $thread_id:expr,
        thread_name: $thread_name:expr,
        => $thread_fields_named:ident,
           $thread_fields_unnamed:ident,
           $track_fields:ident,
           $thread_track:ident
    ) => {
        let __pu = $process_uuid;
        let __pid: u64 = $process_id;
        let __tid: u64 = $thread_id;
        let __tname: Option<&str> = $thread_name;
        let $thread_fields_named;
        let $thread_fields_unnamed;
        let thread_fields_ref: &[perfetto_sdk::track_event::TrackEventProtoField] =
            if let Some(name) = __tname {
                $thread_fields_named = [
                    perfetto_sdk::track_event::TrackEventProtoField::VarInt(1, __pid),
                    perfetto_sdk::track_event::TrackEventProtoField::VarInt(2, __tid),
                    perfetto_sdk::track_event::TrackEventProtoField::Cstr(5, name),
                ];
                &$thread_fields_named
            } else {
                $thread_fields_unnamed = [
                    perfetto_sdk::track_event::TrackEventProtoField::VarInt(1, __pid),
                    perfetto_sdk::track_event::TrackEventProtoField::VarInt(2, __tid),
                ];
                &$thread_fields_unnamed
            };
        let $track_fields = [
            perfetto_sdk::track_event::TrackEventProtoField::VarInt(5, __pu),
            perfetto_sdk::track_event::TrackEventProtoField::Nested(4, thread_fields_ref),
        ];
        let $thread_track = perfetto_sdk::track_event::TrackEventProtoTrack {
            uuid: __pu ^ __tid,
            fields: &$track_fields,
        };
    };
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
