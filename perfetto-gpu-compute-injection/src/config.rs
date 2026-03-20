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
}
