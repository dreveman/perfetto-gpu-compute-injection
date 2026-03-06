# AGENTS.md

This file provides guidance for AI Agents.

## Build Commands

```bash
# Build on Linux with CUDA toolkit installed
cargo build --release
```

The output artifact is `target/release/libperfetto_gpu_compute_injection.so`.

## Testing

```bash
# Run tests on Linux with CUDA toolkit installed
cargo test --workspace --verbose

# Run tests with CUDA stubs (non-Linux or without CUDA toolkit)
cargo test --workspace --verbose --features stubs
```

## Linting and Formatting

```bash
# Check Rust formatting
cargo fmt -- --check

# Check C/C++ formatting (Google style)
clang-format --dry-run --Werror -style=Google cupti-profiler-sys/stubs.cpp cupti-profiler-sys/wrapper.h

# Lint with Clippy (CI enforces -D warnings)
cargo clippy --features stubs -- -D warnings
```

## Architecture

This is a Rust library that bridges NVIDIA CUPTI with Perfetto tracing. It injects into CUDA applications via `CUDA_INJECTION64_PATH` to capture GPU compute metrics.

### Crate Structure

- **Root crate** (`src/`): Main injection library, builds as cdylib (.so)
  - `lib.rs`: Entry point with `InitializeInjection()`, Perfetto trace emission
  - `callbacks.rs`: CUPTI callback handlers for kernel launches and resource events
  - `state.rs`: Global state management with `GLOBAL_STATE` singleton
  - `tracing.rs`: Perfetto data source registration (`gpu.counters`)
  - `metrics.rs`: Default metrics list and parsing
  - `config.rs`: Environment variable configuration

- **cupti-profiler-sys** (`cupti-profiler-sys/`): Low-level FFI bindings to CUPTI
  - `src/bindings.rs`: Auto-generated via bindgen from `wrapper.h`
  - `build.rs`: Build script for bindgen generation and linking
  - `wrapper.h`: C header for bindgen input
  - `stubs.cpp`: C++ stub implementations for the `stubs` feature

- **cupti-profiler** (`cupti-profiler/`): Safe Rust wrapper around CUPTI
  - `range_profiler.rs`: Range profiling session lifecycle
  - `profiler.rs`: ProfilerHost initialization
  - `metric_evaluator.rs`: Metric decoding from binary counter data

### Key Patterns

1. **Injection Entry**: `InitializeInjection()` is the exported C function called when the library is loaded
2. **Callback-Driven**: Intercepts `cuLaunchKernel` via CUPTI driver API callbacks
3. **Global State**: Thread-safe singleton `GLOBAL_STATE` stores per-context profiling data
4. **Panic Safety**: All callbacks use `panic::catch_unwind()` to prevent unwinding into C code

### Data Flow

```
Kernel Launch → CUPTI Callback → Range Profiler Session →
Hardware Counter Collection → Metric Evaluation → Perfetto TracePackets
```

### Environment Variables

- `INJECTION_METRICS`: Comma/semicolon-separated metric names (defaults to 24 standard metrics)
- `INJECTION_VERBOSE`: Enable detailed stdout logging
- `INJECTION_DATA_SOURCE_NAME`: Override Perfetto data source name (defaults to `gpu.counters`)
- `CUDA_HOME`: CUDA installation path (build-time, defaults to `/usr/local/cuda`)

## Usage

```bash
CUDA_INJECTION64_PATH=target/release/libperfetto_gpu_compute_injection.so \
INJECTION_VERBOSE=1 \
/path/to/cuda_app
```
