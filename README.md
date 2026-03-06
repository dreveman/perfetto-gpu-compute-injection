# Perfetto GPU Compute Injection

This Rust crate implements a profiling injection library that bridges NVIDIA CUPTI (CUDA Profiling Tools Interface) with the Perfetto tracing system. It is designed to be injected into CUDA applications to capture GPU compute metrics and trace events.

## Overview

The library intercepts CUDA driver API calls (specifically `cuLaunchKernel`) and CUPTI callbacks to:

1.  **Track Kernel Launches**: Captures timestamps and details of kernel executions.
2.  **Collect Metrics**: Uses the CUPTI Range Profiler to gather hardware performance counters (e.g., SM cycles, throughput, cache hit rates) for each kernel.
3.  **Emit Perfetto Traces**: Converts collected data into Perfetto trace packets (`TracePacket`), enabling visualization in the Perfetto UI.

## Features

- **Automated Injection**: Initializes itself via `InitializeInjection` (likely called by a preload mechanism or explicit integration).
- **Metric Configuration**: Supports customizable metrics via the `INJECTION_METRICS` environment variable.
- **Verbose Logging**: Debug output can be enabled with `INJECTION_VERBOSE=1`.
- **Concurrency Support**: Thread-safe global state handling for multi-threaded applications.

## Usage

The library can be injected into any unmodified CUDA application using the `CUDA_INJECTION64_PATH` environment variable.

```bash
CUDA_INJECTION64_PATH=target/release/libperfetto_gpu_compute_injection.so /path/to/example_cuda_app
```

## Environment Variables

- `INJECTION_METRICS`: A comma-separated list of CUPTI metric names to collect (e.g., `sm__cycles_elapsed.avg`). If unset, a default set of useful metrics is used.
- `INJECTION_VERBOSE`: Set to any value to enable detailed stdout logging of profiling events.

## Architecture

This crate depends on the internal `cupti-profiler` crate for safe interactions with the NVIDIA CUPTI API. It manages:
- **Global State**: Tracks active contexts and profiling sessions.
- **Perfetto Producer**: Registers a data source (`gpu.counters`) to stream data to the system Perfetto service.

## Build Requirements

- **CUDA Toolkit**: Must be installed.
- **Rust**: Stable toolchain.
- `CUDA_HOME`: Environment variable pointing to the CUDA installation (defaults to `/usr/local/cuda` on Linux).
