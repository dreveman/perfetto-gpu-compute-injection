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

use crate::metrics::{parse_metrics, DEFAULT_METRICS};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};

static VERBOSE_ENABLED: AtomicBool = AtomicBool::new(false);

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

/// Check if verbose logging is enabled.
pub fn is_verbose() -> bool {
    VERBOSE_ENABLED.load(Ordering::Relaxed)
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
    /// is skipped for better performance.
    pub metrics: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            verbose: false,
            metrics: DEFAULT_METRICS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// - `INJECTION_VERBOSE`: specifies if verbose logging is enabled.
    /// - `INJECTION_METRICS`: semicolon or comma separated list of metrics.
    pub fn from_env() -> Self {
        let verbose = env::var("INJECTION_VERBOSE").is_ok();
        VERBOSE_ENABLED.store(verbose, Ordering::Relaxed);
        let metrics_str = env::var("INJECTION_METRICS").unwrap_or_default();
        let metrics = parse_metrics(&metrics_str);

        Self { verbose, metrics }
    }
}
