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

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

static VERBOSE_ENABLED: AtomicBool = AtomicBool::new(false);
static TRACE_STARTUP: OnceLock<Vec<String>> = OnceLock::new();

/// Log a message to stderr with the injection prefix.
/// Only logs if INJECTION_VERBOSE is set.
#[macro_export]
macro_rules! injection_log {
    ($($arg:tt)*) => {
        if $crate::config::is_verbose() {
            eprintln!("==INJECTION== {}", format!($($arg)*));
        }
    };
}

/// Log a fatal error to stderr and exit the process.
/// Always prints regardless of INJECTION_VERBOSE.
#[macro_export]
macro_rules! injection_fatal {
    ($($arg:tt)*) => {{
        eprintln!("==INJECTION== FATAL: {}", format!($($arg)*));
        std::process::exit(1);
    }};
}

/// Check if verbose logging is enabled.
pub fn is_verbose() -> bool {
    VERBOSE_ENABLED.load(Ordering::Relaxed)
}

/// Returns true if `__TRACE_STARTUP` environment variable contains the given data source name.
pub fn trace_startup_has(name: &str) -> bool {
    TRACE_STARTUP
        .get()
        .map(|v| v.iter().any(|s| s == name))
        .unwrap_or(false)
}

/// Configuration for the injection library.
#[derive(Debug, Clone)]
pub struct Config {
    /// Whether verbose logging is enabled.
    pub verbose: bool,
    /// List of metrics to be collected.
    ///
    /// Note: This is only used when the counters data source is enabled.
    /// When only the renderstages data source is enabled, metrics collection
    /// is skipped for better performance. Metrics parsing lives in the
    /// backend-specific crate (e.g. perfetto-cuda-injection).
    pub metrics: Vec<String>,
}

#[allow(clippy::derivable_impls)]
impl Default for Config {
    fn default() -> Self {
        Self {
            verbose: false,
            metrics: Vec::new(),
        }
    }
}

/// Parses a comma or semicolon separated string of metrics.
///
/// If `input` is empty or whitespace-only, returns `defaults` converted to `String`s.
pub fn parse_metrics(input: &str, defaults: &[&str]) -> Vec<String> {
    if input.trim().is_empty() {
        return defaults.iter().map(|s| s.to_string()).collect();
    }
    input
        .split(&[';', ','][..])
        .filter_map(|m| {
            let trimmed = m.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect()
}

/// Whether to match against the mangled or demangled kernel name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ActivityNameFilterNameBase {
    #[default]
    MangledKernelName = 0,
    DemangledKernelName = 1,
}

/// A glob pattern for filtering which kernel dispatches to profile.
#[derive(Debug, Clone, Default)]
pub struct ActivityNameFilter {
    /// Glob pattern to match against the kernel name.
    pub name_glob: String,
    /// Which form of the kernel name to match against.
    pub name_base: ActivityNameFilterNameBase,
}

/// Controls which dispatch ranges to profile (skip N, then profile M).
#[derive(Debug, Clone, Default)]
pub struct ActivityRange {
    /// Number of dispatches to skip before profiling.
    pub skip: u32,
    /// Number of dispatches to profile after skipping.
    pub count: u32,
}

/// Parsed `InstrumentedSamplingConfig` from the Perfetto trace config.
#[derive(Debug, Clone, Default)]
pub struct InstrumentedSamplingConfig {
    /// Glob filters for kernel names. A kernel is profiled only if it
    /// matches at least one filter (or no filters are set).
    pub activity_name_filters: Vec<ActivityNameFilter>,
    /// Include globs matched against NVTX range names. A dispatch is
    /// profiled only if it occurs within an NVTX range matching at least
    /// one of these globs (or no include globs are set).
    pub activity_tx_include_globs: Vec<String>,
    /// Exclude globs matched against NVTX range names. A dispatch is
    /// skipped if it occurs within an NVTX range matching any of these.
    pub activity_tx_exclude_globs: Vec<String>,
    /// Dispatch ranges for sampling (skip N, profile M, repeat).
    pub activity_ranges: Vec<ActivityRange>,
}

/// Simple glob pattern matcher supporting `*` (any sequence) and `?` (single char).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_pi, mut star_ti) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

impl InstrumentedSamplingConfig {
    /// Returns whether a kernel should be profiled based on `activity_name_filters`.
    ///
    /// If no filters are configured, all kernels are profiled.
    /// Otherwise, a kernel is profiled if it matches **any** filter's glob pattern.
    pub fn should_profile_kernel(&self, mangled: &str, demangled: &str) -> bool {
        if self.activity_name_filters.is_empty() {
            return true;
        }
        self.activity_name_filters.iter().any(|filter| {
            let name = match filter.name_base {
                ActivityNameFilterNameBase::MangledKernelName => mangled,
                ActivityNameFilterNameBase::DemangledKernelName => demangled,
            };
            glob_match(&filter.name_glob, name)
        })
    }

    /// Returns whether a kernel should be profiled based on the current NVTX
    /// range stack and the `activity_tx_include_globs` / `activity_tx_exclude_globs`.
    ///
    /// - If both lists are empty, returns `true` (no NVTX filtering).
    /// - If include globs are set, at least one range name in the stack must
    ///   match at least one include glob.
    /// - If exclude globs are set, no range name in the stack may match any
    ///   exclude glob.
    pub fn should_profile_in_nvtx_context(&self, nvtx_stack: &[String]) -> bool {
        if self.activity_tx_include_globs.is_empty() && self.activity_tx_exclude_globs.is_empty() {
            return true;
        }
        // Exclude check: reject if any range matches any exclude glob.
        for range_name in nvtx_stack {
            for glob in &self.activity_tx_exclude_globs {
                if glob_match(glob, range_name) {
                    return false;
                }
            }
        }
        // Include check: require at least one range to match an include glob.
        if self.activity_tx_include_globs.is_empty() {
            return true;
        }
        for range_name in nvtx_stack {
            for glob in &self.activity_tx_include_globs {
                if glob_match(glob, range_name) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns whether a dispatch at the given count should be profiled
    /// based on `activity_ranges` (skip/count sampling).
    ///
    /// If no ranges are configured, returns `true` (profile all).
    /// Otherwise, ranges are evaluated in sequence: each range skips
    /// `skip` dispatches then profiles `count` dispatches. After all
    /// ranges are exhausted, no more dispatches are profiled.
    pub fn should_profile_at_count(&self, dispatch_count: u64) -> bool {
        if self.activity_ranges.is_empty() {
            return true;
        }
        let mut offset: u64 = 0;
        for range in &self.activity_ranges {
            let skip = range.skip as u64;
            let count = range.count as u64;
            if dispatch_count < offset + skip {
                return false; // in skip window
            }
            if dispatch_count < offset + skip + count {
                return true; // in profile window
            }
            offset += skip + count;
        }
        false // all ranges exhausted
    }
}

/// Parsed `GpuCounterConfig` fields from the Perfetto trace config.
///
/// Populated in the counters data source `on_setup` callback.
#[derive(Debug, Clone, Default)]
pub struct CounterConfig {
    /// Whether instrumented (per-kernel) counter sampling is enabled.
    pub instrumented_sampling: bool,
    /// Metric names from the trace config's `counter_names` field.
    /// When non-empty, these override `INJECTION_METRICS` and backend defaults.
    pub counter_names: Vec<String>,
    /// Configuration for instrumented (per-kernel) counter sampling.
    pub instrumented_sampling_config: InstrumentedSamplingConfig,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// - `INJECTION_VERBOSE`: specifies if verbose logging is enabled.
    /// - `__TRACE_STARTUP`: comma-separated list of data sources to wait for at startup.
    ///
    /// Note: `INJECTION_METRICS` is handled by backend-specific crates, not here.
    pub fn from_env() -> Self {
        let verbose = env::var("INJECTION_VERBOSE").is_ok();
        VERBOSE_ENABLED.store(verbose, Ordering::Relaxed);

        // Parse __TRACE_STARTUP into a list of data source names
        let _ = TRACE_STARTUP.get_or_init(|| {
            env::var("__TRACE_STARTUP")
                .map(|v| v.split(',').map(|s| s.to_string()).collect())
                .unwrap_or_default()
        });

        Self {
            verbose,
            metrics: Vec::new(),
        }
    }
}

/// Returns the process ID and process name (read from `/proc/self/comm`).
pub fn get_process_info() -> (i32, String) {
    let pid = unsafe { libc::getpid() };
    let name = std::fs::read_to_string("/proc/self/comm")
        .unwrap_or_else(|_| "unknown".to_string())
        .trim_end_matches('\n')
        .to_owned();
    (pid, name)
}

/// Captures the thread name for `tid` from `/proc/self/task/<tid>/comm` into
/// `thread_names` if the entry is vacant and the name is non-empty.
pub fn capture_thread_name<K: Eq + std::hash::Hash + std::fmt::Display>(
    thread_names: &mut std::collections::HashMap<K, String>,
    tid: K,
) {
    if let std::collections::hash_map::Entry::Vacant(e) = thread_names.entry(tid) {
        let name = std::fs::read_to_string(format!("/proc/self/task/{}/comm", e.key()))
            .unwrap_or_default()
            .trim_end_matches('\n')
            .to_owned();
        if !name.is_empty() {
            e.insert(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics_custom() {
        let input = "metric1; metric2, metric3";
        let metrics = parse_metrics(input, &["default1"]);
        assert_eq!(metrics, vec!["metric1", "metric2", "metric3"]);
    }

    #[test]
    fn test_parse_metrics_with_empty_segments() {
        let input = "metric1;;,metric2";
        let metrics = parse_metrics(input, &["default1"]);
        assert_eq!(metrics, vec!["metric1", "metric2"]);
    }

    #[test]
    fn test_parse_metrics_empty_returns_defaults() {
        let defaults = &["d1", "d2"];
        let metrics = parse_metrics("", defaults);
        assert_eq!(metrics, vec!["d1", "d2"]);
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("foo*", "foobar"));
        assert!(glob_match("*bar", "foobar"));
        assert!(glob_match("*baz*", "foobazqux"));
        assert!(glob_match("f?o", "foo"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("foo", "bar"));
        assert!(!glob_match("foo*", "barfoo"));
        assert!(!glob_match("f?o", "fo"));
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
        assert!(glob_match("*", ""));
        assert!(glob_match("**", "abc"));
    }

    #[test]
    fn test_should_profile_kernel_empty_filters() {
        let isc = InstrumentedSamplingConfig::default();
        assert!(isc.should_profile_kernel("_Z3foov", "foo()"));
    }

    #[test]
    fn test_should_profile_kernel_mangled_filter() {
        let isc = InstrumentedSamplingConfig {
            activity_name_filters: vec![ActivityNameFilter {
                name_glob: "*foo*".to_string(),
                name_base: ActivityNameFilterNameBase::MangledKernelName,
            }],
            ..Default::default()
        };
        assert!(isc.should_profile_kernel("_Z3foov", "foo()"));
        assert!(!isc.should_profile_kernel("_Z3barv", "bar()"));
    }

    #[test]
    fn test_should_profile_kernel_demangled_filter() {
        let isc = InstrumentedSamplingConfig {
            activity_name_filters: vec![ActivityNameFilter {
                name_glob: "*matmul*".to_string(),
                name_base: ActivityNameFilterNameBase::DemangledKernelName,
            }],
            ..Default::default()
        };
        assert!(isc.should_profile_kernel("_Z13matmul_kernelv", "matmul_kernel()"));
        assert!(!isc.should_profile_kernel("_Z13reduce_kernelv", "reduce_kernel()"));
    }

    #[test]
    fn test_should_profile_kernel_multiple_filters_or() {
        let isc = InstrumentedSamplingConfig {
            activity_name_filters: vec![
                ActivityNameFilter {
                    name_glob: "*matmul*".to_string(),
                    name_base: ActivityNameFilterNameBase::DemangledKernelName,
                },
                ActivityNameFilter {
                    name_glob: "*reduce*".to_string(),
                    name_base: ActivityNameFilterNameBase::DemangledKernelName,
                },
            ],
            ..Default::default()
        };
        assert!(isc.should_profile_kernel("x", "matmul_kernel()"));
        assert!(isc.should_profile_kernel("x", "reduce_kernel()"));
        assert!(!isc.should_profile_kernel("x", "softmax_kernel()"));
    }

    #[test]
    fn test_nvtx_context_empty_globs() {
        let isc = InstrumentedSamplingConfig::default();
        assert!(isc.should_profile_in_nvtx_context(&[]));
        assert!(isc.should_profile_in_nvtx_context(&["anything".to_string()]));
    }

    #[test]
    fn test_nvtx_context_include_match() {
        let isc = InstrumentedSamplingConfig {
            activity_tx_include_globs: vec!["training*".to_string()],
            ..Default::default()
        };
        assert!(isc.should_profile_in_nvtx_context(&["training_step".to_string()]));
        assert!(!isc.should_profile_in_nvtx_context(&["inference".to_string()]));
        assert!(!isc.should_profile_in_nvtx_context(&[]));
    }

    #[test]
    fn test_nvtx_context_exclude_match() {
        let isc = InstrumentedSamplingConfig {
            activity_tx_exclude_globs: vec!["warmup*".to_string()],
            ..Default::default()
        };
        assert!(isc.should_profile_in_nvtx_context(&["training_step".to_string()]));
        assert!(!isc.should_profile_in_nvtx_context(&["warmup_phase".to_string()]));
        assert!(isc.should_profile_in_nvtx_context(&[]));
    }

    #[test]
    fn test_nvtx_context_nested_stack() {
        let isc = InstrumentedSamplingConfig {
            activity_tx_include_globs: vec!["training*".to_string()],
            activity_tx_exclude_globs: vec!["*backward*".to_string()],
            ..Default::default()
        };
        // Inner range matches include — profiled.
        assert!(isc
            .should_profile_in_nvtx_context(
                &["epoch_1".to_string(), "training_step".to_string(),]
            ));
        // Inner range matches exclude — skipped even though outer matches include.
        assert!(!isc.should_profile_in_nvtx_context(&[
            "training_step".to_string(),
            "backward_pass".to_string(),
        ]));
        // No include match.
        assert!(!isc.should_profile_in_nvtx_context(&["inference".to_string()]));
    }

    #[test]
    fn test_skip_count_empty_ranges() {
        let isc = InstrumentedSamplingConfig::default();
        for i in 0..10 {
            assert!(isc.should_profile_at_count(i));
        }
    }

    #[test]
    fn test_skip_count_single_range() {
        let isc = InstrumentedSamplingConfig {
            activity_ranges: vec![ActivityRange { skip: 2, count: 3 }],
            ..Default::default()
        };
        // skip 0, skip 1, profile 2, profile 3, profile 4, done
        assert!(!isc.should_profile_at_count(0));
        assert!(!isc.should_profile_at_count(1));
        assert!(isc.should_profile_at_count(2));
        assert!(isc.should_profile_at_count(3));
        assert!(isc.should_profile_at_count(4));
        assert!(!isc.should_profile_at_count(5));
        assert!(!isc.should_profile_at_count(100));
    }

    #[test]
    fn test_skip_count_multiple_ranges() {
        let isc = InstrumentedSamplingConfig {
            activity_ranges: vec![
                ActivityRange { skip: 1, count: 2 },
                ActivityRange { skip: 3, count: 1 },
            ],
            ..Default::default()
        };
        // Range 1: skip 0, profile 1, profile 2
        assert!(!isc.should_profile_at_count(0));
        assert!(isc.should_profile_at_count(1));
        assert!(isc.should_profile_at_count(2));
        // Range 2: skip 3, skip 4, skip 5, profile 6
        assert!(!isc.should_profile_at_count(3));
        assert!(!isc.should_profile_at_count(4));
        assert!(!isc.should_profile_at_count(5));
        assert!(isc.should_profile_at_count(6));
        // All exhausted
        assert!(!isc.should_profile_at_count(7));
    }

    #[test]
    fn test_skip_count_zero_skip() {
        let isc = InstrumentedSamplingConfig {
            activity_ranges: vec![ActivityRange { skip: 0, count: 3 }],
            ..Default::default()
        };
        assert!(isc.should_profile_at_count(0));
        assert!(isc.should_profile_at_count(1));
        assert!(isc.should_profile_at_count(2));
        assert!(!isc.should_profile_at_count(3));
    }

    #[test]
    fn test_skip_count_zero_count() {
        let isc = InstrumentedSamplingConfig {
            activity_ranges: vec![
                ActivityRange { skip: 0, count: 0 },
                ActivityRange { skip: 0, count: 2 },
            ],
            ..Default::default()
        };
        // First range is empty (skip=0, count=0), goes to second.
        assert!(isc.should_profile_at_count(0));
        assert!(isc.should_profile_at_count(1));
        assert!(!isc.should_profile_at_count(2));
    }
}
