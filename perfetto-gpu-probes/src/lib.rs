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

//! Perfetto GPU Probes — global GPU counter polling service.
//!
//! This crate provides a standalone Perfetto producer that polls GPU metrics
//! (frequency, etc.) and emits them as ftrace event packets. It supports
//! NVIDIA GPUs via NVML and AMD GPUs via sysfs. Both backends register
//! independently, gated on hardware availability.

pub mod amd_poller;
pub mod amd_sysfs;
pub mod nvml;
pub mod nvml_poller;
pub mod poller;
pub mod protos;

use crate::poller::{GpuMetadata, InstanceStop, PollableGpu};
use clap::Parser;
use perfetto_sdk::data_source::{DataSource, DataSourceArgsBuilder};
use perfetto_sdk::pb_decoder::{PbDecoder, PbDecoderField};
use perfetto_sdk::producer::{Backends, Producer, ProducerInitArgsBuilder};
use perfetto_sdk_protos_gpu::protos::config::data_source_config::DataSourceConfigExtFieldNumber;
use perfetto_sdk_protos_gpu::protos::config::gpu::gpu_counter_config::GpuCounterConfigFieldNumber;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

pub static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Formats the `[SSS.mmm] file:line` prefix matching upstream Perfetto's
/// `LogMessage()` output format. Wall time modulo 1000 seconds.
#[doc(hidden)]
pub fn log_prefix(file: &str, line: u32) -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u32;
    let t_sec = (ms / 1000) % 1000;
    let t_ms = ms % 1000;

    // Extract basename from file path.
    let basename = file.rsplit('/').next().unwrap_or(file);

    let line_str = format!("{}", line);
    // 24-char wide file:line column, right-aligned filename.
    let fname_max = 24usize.saturating_sub(line_str.len() + 1);
    let basename_display = if basename.len() <= fname_max {
        format!("{:>width$}", basename, width = fname_max)
    } else {
        basename[basename.len() - fname_max..].to_string()
    };

    format!("[{:03}.{:03}] {}:{}", t_sec, t_ms, basename_display, line)
}

/// Always-on informational log (like `PERFETTO_LOG`).
#[macro_export]
macro_rules! perfetto_log {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::log_prefix(file!(), line!()), format!($($arg)*));
    };
}

/// Always-on error log (like `PERFETTO_ELOG`).
#[macro_export]
macro_rules! perfetto_elog {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::log_prefix(file!(), line!()), format!($($arg)*));
    };
}

/// Debug log, only emitted when `--verbose` is passed (like `PERFETTO_DLOG`).
#[macro_export]
macro_rules! perfetto_dlog {
    ($($arg:tt)*) => {
        if $crate::VERBOSE.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("{} {}", $crate::log_prefix(file!(), line!()), format!($($arg)*));
        }
    };
}

/// Perfetto GPU Probes — global GPU counter polling service.
#[derive(Parser)]
#[command(name = "traced_gpu_probes")]
struct Args {
    /// Poll interval in milliseconds.
    #[arg(long, default_value_t = 100)]
    poll_interval_ms: u64,

    /// Enable verbose logging.
    #[arg(long)]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// NVML statics
// ---------------------------------------------------------------------------

static NVML_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();
static NVML_INSTANCE_CONFIGS: LazyLock<Mutex<HashMap<u32, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NVML_INSTANCES: LazyLock<Mutex<HashMap<u32, Arc<InstanceStop>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// AMD statics
// ---------------------------------------------------------------------------

static AMD_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();
static AMD_INSTANCE_CONFIGS: LazyLock<Mutex<HashMap<u32, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static AMD_INSTANCES: LazyLock<Mutex<HashMap<u32, Arc<InstanceStop>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// GPU info statics
// ---------------------------------------------------------------------------

static GPU_INFO_DATA_SOURCE: OnceLock<DataSource> = OnceLock::new();

/// Set to true after NVML is successfully initialized.
static NVML_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Parses `CounterPeriodNs` from the raw `DataSourceConfig` protobuf bytes.
fn parse_counter_period_ns(config: &[u8]) -> Option<u64> {
    const GPU_COUNTER_CONFIG_ID: u32 = DataSourceConfigExtFieldNumber::GpuCounterConfig as u32;
    const COUNTER_PERIOD_NS_ID: u32 = GpuCounterConfigFieldNumber::CounterPeriodNs as u32;

    for item in PbDecoder::new(config) {
        if let Ok((GPU_COUNTER_CONFIG_ID, PbDecoderField::Delimited(value))) = &item {
            for inner in PbDecoder::new(value) {
                if let Ok((COUNTER_PERIOD_NS_ID, PbDecoderField::Varint(v))) = &inner {
                    if *v > 0 {
                        return Some(*v);
                    }
                }
            }
        }
    }
    None
}

/// Function pointer for backend-specific poll operation.
type PollFn = fn(&'static DataSource<'static>, u32, &InstanceStop, u64);

/// Registers a polled `gpu.counters.<suffix>` data source with the given
/// backend-specific statics and function pointers.
fn register_polled_data_source(
    ds_cell: &'static OnceLock<DataSource>,
    instance_configs: &'static LazyLock<Mutex<HashMap<u32, u64>>>,
    instances: &'static LazyLock<Mutex<HashMap<u32, Arc<InstanceStop>>>>,
    poll_fn: PollFn,
    default_poll_ms: u64,
    ds_name: &str,
) -> &'static DataSource<'static> {
    ds_cell.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .will_notify_on_stop(true)
            .on_setup(|inst_id, config, _args| {
                if let Some(period_ns) = parse_counter_period_ns(config) {
                    perfetto_dlog!(
                        "CounterPeriodNs from config: {} (id={})",
                        period_ns,
                        inst_id
                    );
                    let poll_ms = period_ns.div_ceil(1_000_000);
                    instance_configs
                        .lock()
                        .expect("mutex poisoned")
                        .insert(inst_id, poll_ms);
                }
            })
            .on_start(move |inst_id, _| {
                perfetto_dlog!("StartDataSource(id={})", inst_id);
                let poll_ms = instance_configs
                    .lock()
                    .expect("mutex poisoned")
                    .remove(&inst_id)
                    .unwrap_or(default_poll_ms);
                let stop = Arc::new(InstanceStop::new());
                instances
                    .lock()
                    .expect("mutex poisoned")
                    .insert(inst_id, Arc::clone(&stop));
                let ds = ds_cell.get().expect("data source not registered");
                std::thread::spawn(move || {
                    poll_fn(ds, inst_id, &stop, poll_ms);
                });
            })
            .on_stop(move |inst_id, args| {
                perfetto_log!("Producer stop (id={})", inst_id);
                if let Some(stop) = instances.lock().expect("mutex poisoned").remove(&inst_id) {
                    stop.signal();
                }
                let stop_guard = args.postpone();
                let ds = ds_cell.get().expect("data source not registered");
                std::thread::spawn(move || {
                    let mut stop_guard = Some(stop_guard);
                    ds.trace(|ctx| {
                        if ctx.instance_index() != inst_id {
                            return;
                        }
                        let mut sg = stop_guard.take().map(Some);
                        ctx.flush(move || drop(sg.take()));
                    });
                });
            });

        let mut data_source = DataSource::new();
        data_source
            .register(ds_name, data_source_args.build())
            .expect("failed to register gpu probes data source");

        data_source
    })
}

/// Tries to initialize NVML and register the `gpu.counters.nv` data source.
/// Returns true if successful, false if NVML is unavailable.
fn try_register_nvml_data_source(poll_ms: u64) -> bool {
    let ret = unsafe { nvml::nvmlInit_v2() };
    if ret != nvml::NVML_SUCCESS {
        perfetto_dlog!(
            "NVML not available (error {}), skipping gpu.counters.nv",
            ret
        );
        return false;
    }

    register_polled_data_source(
        &NVML_DATA_SOURCE,
        &NVML_INSTANCE_CONFIGS,
        &NVML_INSTANCES,
        nvml_poller::run_poll_loop,
        poll_ms,
        "gpu.counters.nv",
    );
    NVML_AVAILABLE.store(true, Ordering::Relaxed);
    true
}

/// Tries to discover AMD GPUs and register the `gpu.counters.amd` data source.
/// Returns true if AMD GPUs were found, false otherwise.
fn try_register_amd_data_source(poll_ms: u64) -> bool {
    let gpus = amd_sysfs::enumerate_amd_gpus();
    if gpus.is_empty() {
        perfetto_dlog!("No AMD GPUs found, skipping gpu.counters.amd");
        return false;
    }

    register_polled_data_source(
        &AMD_DATA_SOURCE,
        &AMD_INSTANCE_CONFIGS,
        &AMD_INSTANCES,
        amd_poller::run_poll_loop,
        poll_ms,
        "gpu.counters.amd",
    );
    true
}

/// Collects GPU metadata from all available backends (NVML + AMD sysfs).
fn collect_all_gpu_metadata() -> Vec<GpuMetadata> {
    let mut metadata = Vec::new();

    if NVML_AVAILABLE.load(Ordering::Relaxed) {
        for gpu in nvml_poller::enumerate_gpus() {
            metadata.push(gpu.metadata());
        }
    }

    for gpu in amd_sysfs::enumerate_amd_gpus() {
        metadata.push(gpu.metadata());
    }

    metadata
}

/// Registers the `linux.gpu_info` data source.
///
/// This is a one-shot data source: on start it enumerates all GPUs from all
/// backends and emits a single `GpuInfo` TracePacket with static metadata.
fn register_gpu_info_data_source() {
    GPU_INFO_DATA_SOURCE.get_or_init(|| {
        let data_source_args = DataSourceArgsBuilder::new()
            .on_start(|inst_id, _| {
                perfetto_dlog!("linux.gpu_info StartDataSource(id={})", inst_id);
                let metadata = collect_all_gpu_metadata();
                if metadata.is_empty() {
                    perfetto_dlog!("linux.gpu_info: no GPUs found");
                    return;
                }
                let ds = GPU_INFO_DATA_SOURCE
                    .get()
                    .expect("data source not registered");
                ds.trace(|ctx| {
                    if ctx.instance_index() != inst_id {
                        return;
                    }
                    poller::emit_gpu_info(ctx, &metadata);
                });
            })
            .build();

        let mut data_source = DataSource::new();
        data_source
            .register("linux.gpu_info", data_source_args)
            .expect("failed to register linux.gpu_info data source");

        data_source
    });
}

/// Blocks until SIGINT or SIGTERM is received.
fn wait_for_signal() {
    use std::sync::atomic::AtomicBool;

    static SIGNALED: AtomicBool = AtomicBool::new(false);

    // Install signal handlers.
    unsafe {
        libc::signal(
            libc::SIGINT,
            signal_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            signal_handler as *const () as libc::sighandler_t,
        );
    }

    // Park until signaled.
    while !SIGNALED.load(Ordering::SeqCst) {
        std::thread::park_timeout(std::time::Duration::from_secs(1));
    }

    extern "C" fn signal_handler(_sig: libc::c_int) {
        SIGNALED.store(true, Ordering::SeqCst);
    }
}

/// Runs the GPU probes service. Blocks until interrupted.
///
/// This is the main entry point for the `traced_gpu_probes` binary.
/// It connects to the system Perfetto tracing service, registers
/// available data sources, and waits for trace sessions.
///
/// # Arguments
///
/// - `args`: Command-line arguments (including the program name).
///
/// # Returns
///
/// Exit code: 0 on clean shutdown, 1 on error.
pub fn run(args: &[String]) -> i32 {
    let parsed = match Args::try_parse_from(args) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { 1 } else { 0 };
        }
    };
    let poll_interval_ms = parsed.poll_interval_ms;
    VERBOSE.store(parsed.verbose, Ordering::Relaxed);

    // Initialize Perfetto producer (connects to system tracing service).
    Producer::init(
        ProducerInitArgsBuilder::new()
            .backends(Backends::SYSTEM)
            .build(),
    );

    // Register available data sources.
    try_register_nvml_data_source(poll_interval_ms);
    try_register_amd_data_source(poll_interval_ms);
    register_gpu_info_data_source();

    // Notify parent process that all data sources are registered, matching
    // upstream Perfetto's TRACED_PROBES_NOTIFY_FD protocol: write "1" to
    // the fd specified in the env var, then close it.
    if let Ok(fd_str) = std::env::var("TRACED_GPU_PROBES_NOTIFY_FD") {
        if let Ok(fd) = fd_str.parse::<i32>() {
            let ret = unsafe { libc::write(fd, b"1".as_ptr() as *const libc::c_void, 1) };
            if ret != 1 {
                perfetto_elog!("Failed to write to TRACED_GPU_PROBES_NOTIFY_FD");
            }
            unsafe {
                libc::close(fd);
            }
        }
    }

    perfetto_log!(
        "Starting {} service",
        args.first().map_or("traced_gpu_probes", |s| s.as_str())
    );

    // Block until SIGINT/SIGTERM.
    wait_for_signal();

    0
}
