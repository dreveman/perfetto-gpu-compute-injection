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

// Types and constants: use bindgen-generated output at build time when the
// `bindgen` feature is enabled, otherwise use the checked-in generated
// bindings (bindings.rs).
#[cfg(not(feature = "bindgen"))]
mod types {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code,
        clippy::missing_safety_doc,
        clippy::useless_transmute,
        clippy::fn_to_numeric_cast,
        clippy::ptr_offset_with_cast,
        clippy::unnecessary_cast,
        clippy::too_many_arguments
    )]
    include!("bindings.rs");
}

#[cfg(feature = "bindgen")]
mod types {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code,
        clippy::missing_safety_doc,
        clippy::useless_transmute,
        clippy::fn_to_numeric_cast,
        clippy::ptr_offset_with_cast,
        clippy::unnecessary_cast,
        clippy::too_many_arguments
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings_generated.rs"));
}

pub use types::*;

// Short constant aliases: bindgen prefixes enum constants with the type name.
pub const ROCPROFILER_STATUS_SUCCESS: rocprofiler_status_t =
    rocprofiler_status_t_ROCPROFILER_STATUS_SUCCESS as rocprofiler_status_t;
#[allow(clippy::unnecessary_cast)]
pub const ROCPROFILER_BUFFER_CATEGORY_TRACING: u32 =
    rocprofiler_buffer_category_t_ROCPROFILER_BUFFER_CATEGORY_TRACING as u32;
pub const ROCPROFILER_AGENT_TYPE_GPU: rocprofiler_agent_type_t =
    rocprofiler_agent_type_t_ROCPROFILER_AGENT_TYPE_GPU as rocprofiler_agent_type_t;
pub const ROCPROFILER_AGENT_INFO_VERSION_0: rocprofiler_agent_version_t =
    rocprofiler_agent_version_t_ROCPROFILER_AGENT_INFO_VERSION_0 as rocprofiler_agent_version_t;
pub const ROCPROFILER_CALLBACK_PHASE_ENTER: rocprofiler_callback_phase_t =
    rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_ENTER as rocprofiler_callback_phase_t;
pub const ROCPROFILER_CALLBACK_PHASE_LOAD: rocprofiler_callback_phase_t =
    rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_LOAD as rocprofiler_callback_phase_t;
pub const ROCPROFILER_CALLBACK_PHASE_EXIT: rocprofiler_callback_phase_t =
    rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_EXIT as rocprofiler_callback_phase_t;
pub const ROCPROFILER_CALLBACK_PHASE_UNLOAD: rocprofiler_callback_phase_t =
    rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_UNLOAD as rocprofiler_callback_phase_t;
pub const ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT: rocprofiler_callback_tracing_kind_t =
    rocprofiler_callback_tracing_kind_t_ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
        as rocprofiler_callback_tracing_kind_t;
pub const ROCPROFILER_CODE_OBJECT_LOAD: rocprofiler_tracing_operation_t =
    rocprofiler_code_object_operation_t_ROCPROFILER_CODE_OBJECT_LOAD
        as rocprofiler_tracing_operation_t;
pub const ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER: rocprofiler_tracing_operation_t =
    rocprofiler_code_object_operation_t_ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER
        as rocprofiler_tracing_operation_t;
pub const ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: rocprofiler_buffer_tracing_kind_t =
    rocprofiler_buffer_tracing_kind_t_ROCPROFILER_BUFFER_TRACING_MEMORY_COPY
        as rocprofiler_buffer_tracing_kind_t;
pub const ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: rocprofiler_buffer_tracing_kind_t =
    rocprofiler_buffer_tracing_kind_t_ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
        as rocprofiler_buffer_tracing_kind_t;
pub const ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API: rocprofiler_buffer_tracing_kind_t =
    rocprofiler_buffer_tracing_kind_t_ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API
        as rocprofiler_buffer_tracing_kind_t;
pub const ROCPROFILER_BUFFER_POLICY_LOSSLESS: rocprofiler_buffer_policy_t =
    rocprofiler_buffer_policy_t_ROCPROFILER_BUFFER_POLICY_LOSSLESS as rocprofiler_buffer_policy_t;
pub const ROCPROFILER_COUNTER_INFO_VERSION_0: rocprofiler_counter_info_version_id_t =
    rocprofiler_counter_info_version_id_t_ROCPROFILER_COUNTER_INFO_VERSION_0
        as rocprofiler_counter_info_version_id_t;

// Accessor methods for bindgen structs that use anonymous unions or
// different field names than simple named fields.
impl rocprofiler_record_header_t {
    #[inline]
    pub fn category(&self) -> u32 {
        unsafe { self.__bindgen_anon_1.__bindgen_anon_1.category }
    }
    #[inline]
    pub fn kind(&self) -> u32 {
        unsafe { self.__bindgen_anon_1.__bindgen_anon_1.kind }
    }
}

impl rocprofiler_agent_v0_t {
    #[inline]
    pub fn agent_type(&self) -> rocprofiler_agent_type_t {
        self.type_
    }
}

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

// When using stubs, link against the compiled C++ stubs library.
// When not using stubs, use runtime dlsym dispatch instead.
#[cfg(feature = "stubs")]
extern "C" {
    pub fn rocprofiler_create_context(ctx: *mut rocprofiler_context_id_t) -> rocprofiler_status_t;
    pub fn rocprofiler_start_context(ctx: rocprofiler_context_id_t) -> rocprofiler_status_t;
    pub fn rocprofiler_stop_context(ctx: rocprofiler_context_id_t) -> rocprofiler_status_t;
    pub fn rocprofiler_create_buffer(
        context: rocprofiler_context_id_t,
        size: usize,
        watermark: usize,
        policy: rocprofiler_buffer_policy_t,
        callback: rocprofiler_buffer_tracing_cb_t,
        callback_data: *mut std::os::raw::c_void,
        buffer_id: *mut rocprofiler_buffer_id_t,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_flush_buffer(buffer_id: rocprofiler_buffer_id_t) -> rocprofiler_status_t;
    pub fn rocprofiler_configure_buffer_tracing_service(
        context_id: rocprofiler_context_id_t,
        kind: rocprofiler_buffer_tracing_kind_t,
        operations: *mut rocprofiler_tracing_operation_t,
        operations_count: usize,
        buffer_id: rocprofiler_buffer_id_t,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_configure_callback_tracing_service(
        context_id: rocprofiler_context_id_t,
        kind: rocprofiler_callback_tracing_kind_t,
        operations: *mut rocprofiler_tracing_operation_t,
        operations_count: usize,
        callback: rocprofiler_callback_tracing_cb_t,
        callback_args: *mut std::os::raw::c_void,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_query_available_agents(
        version: rocprofiler_agent_version_t,
        callback: rocprofiler_query_available_agents_cb_t,
        agent_size: usize,
        user_data: *mut std::os::raw::c_void,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_iterate_agent_supported_counters(
        agent_id: rocprofiler_agent_id_t,
        cb: rocprofiler_available_counters_cb_t,
        user_data: *mut std::os::raw::c_void,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_query_counter_info(
        counter_id: rocprofiler_counter_id_t,
        version: rocprofiler_counter_info_version_id_t,
        info: *mut std::os::raw::c_void,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_query_record_counter_id(
        id: u64,
        counter_id: *mut rocprofiler_counter_id_t,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_create_counter_config(
        agent_id: rocprofiler_agent_id_t,
        counters_list: *mut rocprofiler_counter_id_t,
        counters_count: usize,
        config_id: *mut rocprofiler_counter_config_id_t,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_configure_callback_dispatch_counting_service(
        context_id: rocprofiler_context_id_t,
        dispatch_callback: rocprofiler_dispatch_counting_service_cb_t,
        dispatch_callback_args: *mut std::os::raw::c_void,
        record_callback: rocprofiler_dispatch_counting_record_cb_t,
        record_callback_args: *mut std::os::raw::c_void,
    ) -> rocprofiler_status_t;
    pub fn rocprofiler_query_buffer_tracing_kind_operation_name(
        kind: rocprofiler_buffer_tracing_kind_t,
        operation: rocprofiler_tracing_operation_t,
        name: *mut *const std::os::raw::c_char,
        name_len: *mut usize,
    ) -> rocprofiler_status_t;
}

#[cfg(not(feature = "stubs"))]
mod dispatch;
#[cfg(not(feature = "stubs"))]
pub use dispatch::*;
