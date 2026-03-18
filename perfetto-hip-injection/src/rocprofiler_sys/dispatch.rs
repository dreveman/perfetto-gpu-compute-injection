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

//! Runtime dlsym dispatch for rocprofiler-sdk functions.
//!
//! Instead of linking against `librocprofiler-sdk.so` (which pulls in
//! `librocprofiler-register.so` whose constructor races to find
//! `rocprofiler_configure` before our symbols are globally visible),
//! we resolve all rocprofiler functions via `dlsym` at runtime.

use std::os::raw::{c_char, c_void};
use std::sync::OnceLock;

use super::types::*;

// ---------------------------------------------------------------------------
// Library handle
// ---------------------------------------------------------------------------

/// RTLD_NOLOAD: only return a handle if the library is already loaded.
const RTLD_NOLOAD: i32 = 0x4;
const RTLD_LAZY: i32 = 0x1;

extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

static LIB_HANDLE: OnceLock<usize> = OnceLock::new();

fn get_lib_handle() -> *mut c_void {
    *LIB_HANDLE.get_or_init(|| {
        let handle = unsafe { dlopen(c"librocprofiler-sdk.so".as_ptr(), RTLD_NOLOAD | RTLD_LAZY) };
        if handle.is_null() {
            eprintln!(
                "==INJECTION== could not find librocprofiler-sdk.so \
                 (expected to be loaded by host runtime)"
            );
        }
        handle as usize
    }) as *mut c_void
}

// ---------------------------------------------------------------------------
// Helper macro
// ---------------------------------------------------------------------------

macro_rules! dispatch_fn {
    (
        $fn_name:ident ( $($arg_name:ident : $arg_ty:ty),* $(,)? ) -> $ret_ty:ty
    ) => {
        #[allow(non_snake_case)]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $fn_name ( $($arg_name: $arg_ty),* ) -> $ret_ty {
            static CACHE: OnceLock<Option<usize>> = OnceLock::new();
            let fp = *CACHE.get_or_init(|| {
                let handle = get_lib_handle();
                if handle.is_null() {
                    return None;
                }
                let sym = dlsym(handle, concat!(stringify!($fn_name), "\0").as_ptr() as *const c_char);
                if sym.is_null() {
                    eprintln!(
                        "==INJECTION== Error: could not resolve symbol: {}",
                        stringify!($fn_name)
                    );
                    if let Ok(hint) = std::env::var("INJECTION_HIP_SYMBOL_LOOKUP_HINT") {
                        for line in hint.lines() {
                            eprintln!("==INJECTION==   {}", line);
                        }
                    }
                    None
                } else {
                    Some(sym as usize)
                }
            });
            match fp {
                Some(addr) => {
                    let f: unsafe extern "C" fn($($arg_ty),*) -> $ret_ty =
                        std::mem::transmute(addr);
                    f($($arg_name),*)
                }
                None => ROCPROFILER_STATUS_ERROR,
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Status error constant (not in bindings, but value 1 per rocprofiler-sdk)
// ---------------------------------------------------------------------------

const ROCPROFILER_STATUS_ERROR: rocprofiler_status_t = 1;

// ---------------------------------------------------------------------------
// Dispatched functions
// ---------------------------------------------------------------------------

dispatch_fn!(rocprofiler_query_available_agents(
    version: rocprofiler_agent_version_t,
    callback: rocprofiler_query_available_agents_cb_t,
    agent_size: usize,
    user_data: *mut c_void,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_iterate_agent_supported_counters(
    agent_id: rocprofiler_agent_id_t,
    cb: rocprofiler_available_counters_cb_t,
    user_data: *mut c_void,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_query_counter_info(
    counter_id: rocprofiler_counter_id_t,
    version: rocprofiler_counter_info_version_id_t,
    info: *mut c_void,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_create_counter_config(
    agent_id: rocprofiler_agent_id_t,
    counters_list: *mut rocprofiler_counter_id_t,
    counters_count: usize,
    config_id: *mut rocprofiler_counter_config_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_create_context(
    context_id: *mut rocprofiler_context_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_configure_callback_tracing_service(
    context_id: rocprofiler_context_id_t,
    kind: rocprofiler_callback_tracing_kind_t,
    operations: *mut rocprofiler_tracing_operation_t,
    operations_count: usize,
    callback: rocprofiler_callback_tracing_cb_t,
    callback_args: *mut c_void,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_start_context(
    context_id: rocprofiler_context_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_create_buffer(
    context: rocprofiler_context_id_t,
    size: usize,
    watermark: usize,
    policy: rocprofiler_buffer_policy_t,
    callback: rocprofiler_buffer_tracing_cb_t,
    callback_data: *mut c_void,
    buffer_id: *mut rocprofiler_buffer_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_configure_buffer_tracing_service(
    context_id: rocprofiler_context_id_t,
    kind: rocprofiler_buffer_tracing_kind_t,
    operations: *mut rocprofiler_tracing_operation_t,
    operations_count: usize,
    buffer_id: rocprofiler_buffer_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_configure_callback_dispatch_counting_service(
    context_id: rocprofiler_context_id_t,
    dispatch_callback: rocprofiler_dispatch_counting_service_cb_t,
    dispatch_callback_args: *mut c_void,
    record_callback: rocprofiler_dispatch_counting_record_cb_t,
    record_callback_args: *mut c_void,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_stop_context(
    context_id: rocprofiler_context_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_flush_buffer(
    buffer_id: rocprofiler_buffer_id_t,
) -> rocprofiler_status_t);

dispatch_fn!(rocprofiler_query_record_counter_id(
    id: u64,
    counter_id: *mut rocprofiler_counter_id_t,
) -> rocprofiler_status_t);
