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

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
#[allow(unused_imports)]
#[allow(clippy::all)]
#[path = "bindings.rs"]
mod raw_bindings;

// Runtime dlsym dispatch: re-exports all types/constants from raw_bindings
// and provides dlsym-based function implementations. Within the dispatch
// module, locally-defined functions take precedence over glob-imported
// extern "C" declarations from raw_bindings.
#[cfg(not(feature = "stubs"))]
mod dispatch;

// Expose the right module as `bindings` so callers using
// `crate::cupti_profiler_sys::bindings::*` get dispatch functions
// transparently.
#[cfg(not(feature = "stubs"))]
pub mod bindings {
    pub use super::dispatch::*;
}
#[cfg(feature = "stubs")]
pub mod bindings {
    pub use super::raw_bindings::*;
}
pub use bindings::*;
