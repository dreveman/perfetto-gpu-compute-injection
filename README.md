# Perfetto GPU Compute Injection

This Rust crate implements a profiling injection library that bridges NVIDIA CUPTI (CUDA Profiling Tools Interface) with the Perfetto tracing system. It is designed to be injected into CUDA applications to capture GPU compute metrics and trace events.

## Overview

The library intercepts CUDA driver API calls (specifically `cuLaunchKernel`) and CUPTI callbacks to:

1.  **Track Kernel Launches**: Captures timestamps and details of kernel executions.
2.  **Collect Metrics**: Uses the CUPTI Range Profiler to gather hardware performance counters (e.g., SM cycles, throughput, cache hit rates) for each kernel.
3.  **Emit Perfetto Traces**: Converts collected data into Perfetto trace packets (`TracePacket`), enabling visualization in the Perfetto UI.

## Features

- **Automated Injection**: Initializes itself via `InitializeInjection` (likely called by a preload mechanism or explicit integration).
- **Dual Data Sources**: Supports two independent Perfetto data sources for flexible tracing:
  - `gpu.counters`: Hardware performance counter metrics
  - `gpu.renderstages`: Kernel execution events with detailed metadata
- **Metric Configuration**: Supports customizable metrics via the `INJECTION_METRICS` environment variable.
- **Verbose Logging**: Debug output can be enabled with `INJECTION_VERBOSE=1`.
- **Concurrency Support**: Thread-safe global state handling for multi-threaded applications.

## Usage

The library can be injected into any unmodified CUDA application using the `CUDA_INJECTION64_PATH` environment variable.

```bash
CUDA_INJECTION64_PATH=target/release/libperfetto_gpu_compute_injection.so /path/to/example_cuda_app
```

## Data Sources

The library registers two independent Perfetto data sources with names in the format `gpu.<type>.<suffix>`:

- **`gpu.counters.nv`** - Hardware performance counter metrics
- **`gpu.renderstages.nv`** - Kernel execution events

The suffix defaults to "nv" (NVIDIA) since CUPTI is exclusively for NVIDIA GPUs. It can be customized via the `INJECTION_DATA_SOURCE_NAME_SUFFIX` environment variable.

### `gpu.counters.<suffix>`

Emits hardware performance counter data (`GpuCounterEvent` packets). Requires the CUPTI Range Profiler and is useful for detailed performance analysis. When enabled, the library will:
- Initialize the CUPTI profiler
- Collect hardware counter data for each kernel launch
- Emit `GpuCounterDescriptor` and `GpuCounterEvent` trace packets

### `gpu.renderstages.<suffix>`

Emits kernel execution events (`GpuRenderStageEvent` packets). This is a lightweight option that captures:
- Kernel timestamps and names
- Grid/block dimensions
- Thread counts and occupancy metrics
- Shared memory and register usage

**Performance Benefit**: When only `gpu.renderstages.<suffix>` is enabled (without `gpu.counters.<suffix>`), the library skips CUPTI profiler initialization and range profiling, resulting in significantly lower overhead.

## Perfetto Trace Config Examples

Enable only renderstages for low-overhead tracing:

```proto
data_sources: {
    config {
        name: "gpu.renderstages.nv"
    }
}
```

Enable only counters for detailed performance metrics:

```proto
data_sources: {
    config {
        name: "gpu.counters.nv"
    }
}
```

Enable both data sources:

```proto
data_sources: {
    config {
        name: "gpu.counters.nv"
    }
}
data_sources: {
    config {
        name: "gpu.renderstages.nv"
    }
}
```

## Environment Variables

- `INJECTION_METRICS`: A comma-separated list of CUPTI metric names to collect (e.g., `sm__cycles_elapsed.avg`). If unset, a default set of useful metrics is used. **Note**: Only used when the `gpu.counters` data source is enabled.
- `INJECTION_VERBOSE`: Set to any value to enable detailed stdout logging of profiling events.
- `INJECTION_DATA_SOURCE_NAME_SUFFIX`: Set the suffix for data source names (default: `nv`). Data sources will be named `gpu.counters.<suffix>` and `gpu.renderstages.<suffix>`.

## Architecture

This crate depends on the internal `cupti-profiler` crate for safe interactions with the NVIDIA CUPTI API. It manages:
- **Global State**: Tracks active contexts and profiling sessions.
- **Perfetto Producer**: Registers two data sources (`gpu.counters` and `gpu.renderstages`) to stream data to the system Perfetto service.

## Build Requirements

- **CUDA Toolkit**: Must be installed.
- **Rust**: Stable toolchain.
- `CUDA_HOME`: Environment variable pointing to the CUDA installation (defaults to `/usr/local/cuda` on Linux).
