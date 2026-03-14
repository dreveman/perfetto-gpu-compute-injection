# AGENTS.md

This file provides guidance for AI Agents.

## Build Commands

```bash
# Build NVIDIA backend with stubs (no CUDA toolkit required)
cargo build --release -p perfetto-cuda-injection --features stubs

# Build AMD backend with stubs (no ROCm required)
cargo build --release -p perfetto-hip-injection --features stubs

# Build NVIDIA backend (requires CUDA toolkit)
cargo build --release -p perfetto-cuda-injection

# Build AMD backend (requires ROCm)
cargo build --release -p perfetto-hip-injection
```

Output artifacts:
- `target/release/libperfetto_cuda_injection.so` (NVIDIA)
- `target/release/libperfetto_hip_injection.so` (AMD)

## Testing

```bash
# Run all tests with stubs (no GPU hardware required)
cargo test --workspace --verbose --features stubs

# Run all tests (requires CUDA toolkit and ROCm)
cargo test --workspace --verbose
```

## Linting and Formatting

```bash
# Check Rust formatting
cargo fmt --all -- --check

# Check C/C++ formatting (Google style)
clang-format --dry-run --Werror -style=Google \
  perfetto-cuda-injection/stubs.cpp perfetto-cuda-injection/wrapper.h \
  perfetto-hip-injection/stubs.cpp perfetto-hip-injection/wrapper.h

# Lint with Clippy (CI enforces -D warnings)
cargo clippy --workspace --features stubs -- -D warnings
```

## Architecture

This is a 3-crate Cargo workspace that bridges GPU profiling APIs (NVIDIA CUPTI and AMD rocprofiler) with Perfetto tracing. Each backend injects into GPU applications to capture compute metrics.

### Crate Structure

- **`perfetto-gpu-compute-injection/`** — Shared library crate (rlib), no backend-specific code
  - `src/lib.rs`: Re-exports and crate root
  - `src/config.rs`: `Config` struct, `injection_log!` macro
  - `src/tracing.rs`: `GpuBackend` trait, `register_backend()`, Perfetto data source infrastructure (`gpu.counters` and `gpu.renderstages`)

- **`perfetto-cuda-injection/`** — NVIDIA backend (cdylib)
  - `src/lib.rs`: `CuptiBackend` struct implementing `GpuBackend`, `InitializeInjection()` C entry point
  - `src/callbacks.rs`: CUPTI callback handlers for kernel launches, memcpy, memset, and resource events
  - `src/state.rs`: Global state management with `GLOBAL_STATE` singleton, activity types (`KernelActivity`, `MemcpyActivity`, etc.)
  - `src/metrics.rs`: Default NVIDIA metrics list and parsing
  - `src/cupti_profiler/`: Safe Rust wrapper around CUPTI (profiler, range_profiler, metric_evaluator, activity, subscriber, error, macros)
  - `src/cupti_profiler_sys/`: Low-level FFI bindings auto-generated via bindgen
  - `stubs.cpp`: C++ stub implementations for the `stubs` feature
  - `wrapper.h`: C header for bindgen input

- **`perfetto-hip-injection/`** — AMD backend (cdylib)
  - `src/lib.rs`: `RocprofilerBackend` struct implementing `GpuBackend`, `rocprofiler_configure()` C entry point
  - `src/callbacks.rs`: rocprofiler callback handlers
  - `src/state.rs`: Global state management with `GLOBAL_STATE` singleton
  - `src/metrics.rs`: Default AMD metrics list and parsing
  - `src/rocprofiler_sys/`: Low-level FFI bindings auto-generated via bindgen
  - `stubs.cpp`: C++ stub implementations for the `stubs` feature
  - `wrapper.h`: C header for bindgen input

### Key Design: GpuBackend Trait

The shared crate defines a `GpuBackend` trait that each backend implements. A backend is registered once at startup via `register_backend()`. The trait provides hooks for:
- Data source lifecycle (consumer start/stop)
- Render stage and counter event emission
- Activity buffer flushing
- Teardown

### Key Patterns

1. **Injection Entry**: Each backend exports a C function called when the library is loaded (`InitializeInjection` for CUPTI, `rocprofiler_configure` for rocprofiler)
2. **Callback-Driven**: Intercepts GPU API calls via profiling API callbacks
3. **Global State**: Thread-safe singleton `GLOBAL_STATE` stores per-context profiling data
4. **Panic Safety**: All callbacks use `panic::catch_unwind()` to prevent unwinding into C code
5. **Per-Metric Validation**: Invalid metrics are individually filtered out rather than failing the entire batch

### Data Flow

```
Kernel Launch → Profiling Callback → Range Profiler Session →
Hardware Counter Collection → Metric Evaluation → Perfetto TracePackets
```

### Environment Variables

- `INJECTION_METRICS`: Comma/semicolon-separated metric names. Only used when `gpu.counters` data source is enabled.
- `INJECTION_VERBOSE`: Enable detailed stdout logging
- `INJECTION_DATA_SOURCE_NAME_SUFFIX`: Suffix for Perfetto data source names (default: `nv` for NVIDIA, `amd` for AMD). Data sources are named `gpu.counters.<suffix>` and `gpu.renderstages.<suffix>`.
- `CUDA_HOME`: CUDA installation path (build-time, defaults to `/usr/local/cuda`)

### Data Sources

Each backend registers two independent Perfetto data sources with names in the format `gpu.<type>.<suffix>`:

- **NVIDIA**: `gpu.counters.nv` and `gpu.renderstages.nv`
- **AMD**: `gpu.counters.amd` and `gpu.renderstages.amd`

**Performance Note**: When only `gpu.renderstages.<suffix>` is enabled, profiler initialization and range profiling are skipped for significantly lower overhead.

## Usage

### NVIDIA

```bash
CUDA_INJECTION64_PATH=target/release/libperfetto_cuda_injection.so \
INJECTION_VERBOSE=1 \
/path/to/cuda_app
```

### AMD

```bash
ROCP_TOOL_LIB=target/release/libperfetto_hip_injection.so \
INJECTION_VERBOSE=1 \
/path/to/hip_app
```
