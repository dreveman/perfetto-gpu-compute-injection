# Perfetto GPU Compute Injection

A Rust workspace that bridges GPU profiling APIs with the Perfetto tracing system. It injects into GPU applications to capture compute metrics and trace events, supporting both NVIDIA (CUPTI) and AMD (rocprofiler) GPUs.

## Overview

The libraries intercept GPU API calls and profiling callbacks to:

1.  **Track Kernel Launches**: Capture timestamps and details of kernel executions.
2.  **Collect Metrics**: Gather hardware performance counters for each kernel.
3.  **Emit Perfetto Traces**: Convert collected data into Perfetto trace packets (`TracePacket`), enabling visualization in the Perfetto UI.

## Workspace Structure

This is a 3-crate Cargo workspace:

- **`perfetto-gpu-compute-injection/`** — Shared library crate (rlib) containing common infrastructure: `GpuBackend` trait, Perfetto data source registration, configuration, and the `injection_log!` macro.
- **`perfetto-cuda-injection/`** — NVIDIA backend (cdylib). Uses CUPTI for profiling. Injected via `CUDA_INJECTION64_PATH`.
- **`perfetto-hip-injection/`** — AMD backend (cdylib). Uses rocprofiler for profiling. Loaded via `HSA_TOOLS_LIB`.

## Features

- **Automated Injection**: Each backend exports a C entry point called by the GPU runtime when the library is loaded.
- **Dual Data Sources**: Two independent Perfetto data sources per backend:
  - `gpu.counters.<suffix>`: Hardware performance counter metrics
  - `gpu.renderstages.<suffix>`: Kernel execution events with detailed metadata
- **Per-Metric Validation**: Invalid metric names are individually filtered out rather than failing the entire batch.
- **Metric Configuration**: Customizable metrics via the `INJECTION_METRICS` environment variable.
- **Verbose Logging**: Debug output can be enabled with `INJECTION_VERBOSE=1`.
- **Concurrency Support**: Thread-safe global state handling for multi-threaded applications.

## Usage

### NVIDIA (CUDA)

```bash
CUDA_INJECTION64_PATH=target/release/libperfetto_cuda_injection.so \
INJECTION_VERBOSE=1 \
/path/to/cuda_app
```

### AMD (HIP/ROCm)

```bash
ROCP_TOOL_LIB=target/release/libperfetto_hip_injection.so \
INJECTION_VERBOSE=1 \
/path/to/hip_app
```

## Data Sources

Each backend registers two independent Perfetto data sources with names in the format `gpu.<type>.<suffix>`:

- **NVIDIA**: `gpu.counters.nv` and `gpu.renderstages.nv`
- **AMD**: `gpu.counters.amd` and `gpu.renderstages.amd`

The suffix can be customized via the `INJECTION_DATA_SOURCE_NAME_SUFFIX` environment variable.

### `gpu.counters.<suffix>`

Emits hardware performance counter data (`GpuCounterEvent` packets). Uses the CUPTI Range Profiler (NVIDIA) or rocprofiler (AMD) and is useful for detailed performance analysis. When enabled, the library will:
- Initialize the profiler
- Collect hardware counter data for each kernel launch
- Emit `GpuCounterDescriptor` and `GpuCounterEvent` trace packets

### `gpu.renderstages.<suffix>`

Emits kernel execution events (`GpuRenderStageEvent` packets). This is a lightweight option that captures:
- Kernel timestamps and names
- Grid/block dimensions
- Thread counts and occupancy metrics
- Shared memory and register usage

**Performance Benefit**: When only `gpu.renderstages.<suffix>` is enabled (without `gpu.counters.<suffix>`), profiler initialization and range profiling are skipped, resulting in significantly lower overhead.

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

- `INJECTION_METRICS`: A comma or semicolon-separated list of metric names to collect. If unset, a default set is used. Only applies when the `gpu.counters` data source is enabled.
- `INJECTION_VERBOSE`: Set to any value to enable detailed stdout logging.
- `INJECTION_DATA_SOURCE_NAME_SUFFIX`: Set the suffix for data source names (default: `nv` for NVIDIA, `amd` for AMD).
- `CUDA_HOME`: CUDA installation path (build-time, defaults to `/usr/local/cuda`).

## Build Requirements

- **NVIDIA backend**: CUDA Toolkit installed, or use `--features stubs` for development without hardware.
- **AMD backend**: ROCm installed, or use `--features stubs` for development without hardware.
- **Rust**: Nightly toolchain.
