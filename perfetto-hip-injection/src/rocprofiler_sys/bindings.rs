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

//! Hand-written FFI bindings for rocprofiler-sdk.
//!
//! Types and constants derived from:
//!   rocprofiler-sdk/fwd.h
//!   rocprofiler-sdk/agent.h
//!   rocprofiler-sdk/buffer.h
//!   rocprofiler-sdk/buffer_tracing.h
//!   rocprofiler-sdk/callback_tracing.h
//!   rocprofiler-sdk/context.h
//!   rocprofiler-sdk/counters.h
//!   rocprofiler-sdk/registration.h

// C naming conventions are used throughout to match the rocprofiler-sdk ABI.
#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// Status codes (rocprofiler_status_t)
// ---------------------------------------------------------------------------

pub type rocprofiler_status_t = u32;
pub const ROCPROFILER_STATUS_SUCCESS: rocprofiler_status_t = 0;

// ---------------------------------------------------------------------------
// Primitive type aliases
// ---------------------------------------------------------------------------

pub type rocprofiler_timestamp_t = u64;
pub type rocprofiler_thread_id_t = u64;
pub type rocprofiler_tracing_operation_t = i32;
pub type rocprofiler_kernel_id_t = u64;
pub type rocprofiler_dispatch_id_t = u64;
pub type rocprofiler_callback_tracing_kind_t = u32;
pub type rocprofiler_buffer_tracing_kind_t = u32;
pub type rocprofiler_callback_phase_t = u32;
pub type rocprofiler_buffer_policy_t = u32;
pub type rocprofiler_agent_version_t = u32;

// ---------------------------------------------------------------------------
// Buffer policy
// ---------------------------------------------------------------------------

pub const ROCPROFILER_BUFFER_POLICY_LOSSLESS: rocprofiler_buffer_policy_t = 2;

// ---------------------------------------------------------------------------
// Buffer category
// ---------------------------------------------------------------------------

pub const ROCPROFILER_BUFFER_CATEGORY_TRACING: u32 = 1;

// ---------------------------------------------------------------------------
// Callback tracing kinds (rocprofiler_callback_tracing_kind_t)
// ---------------------------------------------------------------------------

pub const ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT: rocprofiler_callback_tracing_kind_t = 10;

// ---------------------------------------------------------------------------
// Callback phases (rocprofiler_callback_phase_t)
// ---------------------------------------------------------------------------

/// ENTER == LOAD: used for code object load and API enter callbacks.
pub const ROCPROFILER_CALLBACK_PHASE_ENTER: rocprofiler_callback_phase_t = 1;
pub const ROCPROFILER_CALLBACK_PHASE_LOAD: rocprofiler_callback_phase_t = 1;
pub const ROCPROFILER_CALLBACK_PHASE_EXIT: rocprofiler_callback_phase_t = 2;
pub const ROCPROFILER_CALLBACK_PHASE_UNLOAD: rocprofiler_callback_phase_t = 2;

// ---------------------------------------------------------------------------
// Code object operations
// ---------------------------------------------------------------------------

pub const ROCPROFILER_CODE_OBJECT_LOAD: rocprofiler_tracing_operation_t = 1;
pub const ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER: rocprofiler_tracing_operation_t =
    2;

// ---------------------------------------------------------------------------
// Buffer tracing kinds (rocprofiler_buffer_tracing_kind_t)
// ---------------------------------------------------------------------------

pub const ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: rocprofiler_buffer_tracing_kind_t = 10;
pub const ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: rocprofiler_buffer_tracing_kind_t = 11;

// ---------------------------------------------------------------------------
// Agent type
// ---------------------------------------------------------------------------

pub type rocprofiler_agent_type_t = u32;
pub const ROCPROFILER_AGENT_TYPE_GPU: rocprofiler_agent_type_t = 2;

// ---------------------------------------------------------------------------
// Agent version
// ---------------------------------------------------------------------------

pub const ROCPROFILER_AGENT_INFO_VERSION_0: rocprofiler_agent_version_t = 1;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_context_id_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_buffer_id_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_agent_id_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_queue_id_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_counter_id_t {
    pub handle: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_counter_config_id_t {
    pub handle: u64,
}

// ---------------------------------------------------------------------------
// Union / compound types
// ---------------------------------------------------------------------------

/// `rocprofiler_user_data_t` — union of a u64 and a void pointer.
/// Both members occupy 8 bytes on 64-bit platforms.
#[repr(C)]
#[derive(Clone, Copy)]
pub union rocprofiler_user_data_t {
    pub value: u64,
    pub ptr: *mut c_void,
}

/// Internal + external correlation ID pair (16 bytes total).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rocprofiler_correlation_id_t {
    pub internal: u64,
    pub external: rocprofiler_user_data_t,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct rocprofiler_dim3_t {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

// ---------------------------------------------------------------------------
// Counter info types (rocprofiler-sdk/counters.h)
// ---------------------------------------------------------------------------

pub type rocprofiler_counter_info_version_id_t = u32;
pub const ROCPROFILER_COUNTER_INFO_VERSION_0: rocprofiler_counter_info_version_id_t = 1;

#[repr(C)]
pub struct rocprofiler_counter_info_v0_t {
    pub id: rocprofiler_counter_id_t,
    pub name: *const c_char,
    pub description: *const c_char,
    pub block: *const c_char,
    pub expression: *const c_char,
    pub is_constant: u8, // bitfield: bit 0 = is_constant, bit 1 = is_derived
}

/// Counter record delivered by the dispatch counting callback.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rocprofiler_counter_record_t {
    pub id: u64, // rocprofiler_counter_instance_id_t
    pub counter_value: f64,
    pub dispatch_id: u64,
    pub user_data: rocprofiler_user_data_t,
    pub agent_id: rocprofiler_agent_id_t,
}

/// Async correlation ID (internal + external).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rocprofiler_async_correlation_id_t {
    pub internal: u64,
    pub external: rocprofiler_user_data_t,
}

/// Data delivered to dispatch counting service callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rocprofiler_dispatch_counting_service_data_t {
    pub size: u64,
    pub correlation_id: rocprofiler_async_correlation_id_t,
    pub start_timestamp: u64,
    pub end_timestamp: u64,
    pub dispatch_info: rocprofiler_kernel_dispatch_info_t,
}

// ---------------------------------------------------------------------------
// Callback tracing record (delivered synchronously on the calling thread)
// ---------------------------------------------------------------------------

/// Matches the C layout from fwd.h:
///   context_id (8) + thread_id (8) + correlation_id (16) +
///   kind (4) + operation (4) + phase (4) + [padding 4] + payload (*mut c_void, 8)
#[repr(C)]
pub struct rocprofiler_callback_tracing_record_t {
    pub context_id: rocprofiler_context_id_t,
    pub thread_id: rocprofiler_thread_id_t,
    pub correlation_id: rocprofiler_correlation_id_t,
    pub kind: rocprofiler_callback_tracing_kind_t,
    pub operation: rocprofiler_tracing_operation_t,
    pub phase: rocprofiler_callback_phase_t,
    pub payload: *mut c_void,
}

// ---------------------------------------------------------------------------
// Buffer record header (32-bit category + 32-bit kind + pointer payload)
// ---------------------------------------------------------------------------

/// `rocprofiler_record_header_t` from fwd.h.
/// The hash union's first 32 bits are category, second 32 bits are kind.
#[repr(C)]
pub struct rocprofiler_record_header_t {
    pub category: u32,
    pub kind: u32,
    pub payload: *mut c_void,
}

impl rocprofiler_record_header_t {
    #[inline]
    pub fn category(&self) -> u32 {
        self.category
    }
    #[inline]
    pub fn kind(&self) -> u32 {
        self.kind
    }
}

// ---------------------------------------------------------------------------
// Kernel dispatch info (128 bytes, static_assert confirmed in fwd.h)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rocprofiler_kernel_dispatch_info_t {
    pub size: u64,
    pub agent_id: rocprofiler_agent_id_t,
    pub queue_id: rocprofiler_queue_id_t,
    pub kernel_id: rocprofiler_kernel_id_t,
    pub dispatch_id: rocprofiler_dispatch_id_t,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub workgroup_size: rocprofiler_dim3_t,
    pub grid_size: rocprofiler_dim3_t,
    pub reserved_padding: [u8; 56],
}

// ---------------------------------------------------------------------------
// Buffer tracing records
// ---------------------------------------------------------------------------

/// Kernel dispatch buffer record.
#[repr(C)]
pub struct rocprofiler_buffer_tracing_kernel_dispatch_record_t {
    pub size: u64,
    pub kind: rocprofiler_buffer_tracing_kind_t,
    pub operation: rocprofiler_tracing_operation_t,
    pub correlation_id: rocprofiler_correlation_id_t,
    pub thread_id: rocprofiler_thread_id_t,
    pub start_timestamp: rocprofiler_timestamp_t,
    pub end_timestamp: rocprofiler_timestamp_t,
    pub dispatch_info: rocprofiler_kernel_dispatch_info_t,
}

/// Memory copy buffer record.
#[repr(C)]
pub struct rocprofiler_buffer_tracing_memory_copy_record_t {
    pub size: u64,
    pub kind: rocprofiler_buffer_tracing_kind_t,
    pub operation: rocprofiler_tracing_operation_t,
    pub correlation_id: rocprofiler_correlation_id_t,
    pub thread_id: rocprofiler_thread_id_t,
    pub start_timestamp: rocprofiler_timestamp_t,
    pub end_timestamp: rocprofiler_timestamp_t,
    pub dst_agent_id: rocprofiler_agent_id_t,
    pub src_agent_id: rocprofiler_agent_id_t,
    pub bytes: u64,
}

// ---------------------------------------------------------------------------
// Code object kernel symbol register data (payload of code_object_callback)
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t {
    pub size: u64,
    pub kernel_id: rocprofiler_kernel_id_t,
    pub code_object_id: u64,
    pub kernel_name: *const c_char,
    pub kernel_object: u64,
    pub kernarg_segment_size: u32,
    pub kernarg_segment_alignment: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub sgpr_count: u32,
    pub arch_vgpr_count: u32,
    pub accum_vgpr_count: u32,
}

// ---------------------------------------------------------------------------
// Agent descriptor (rocprofiler_agent_v0_t)
// We only need the fields up to logical_node_type_id for Phase 1.
// Using a partial repr with the fields we actually access.
// ---------------------------------------------------------------------------

/// Partial layout of `rocprofiler_agent_v0_t`.
/// Fields beyond `logical_node_type_id` are not listed; access via raw pointer
/// arithmetic is not performed, so this is safe.
#[repr(C)]
pub struct rocprofiler_agent_v0_t {
    pub size: u64,
    pub id: rocprofiler_agent_id_t,
    pub r#type: rocprofiler_agent_type_t,
    pub cpu_cores_count: u32,
    pub simd_count: u32,
    pub mem_banks_count: u32,
    pub caches_count: u32,
    pub io_links_count: u32,
    pub cpu_core_id_base: u32,
    pub simd_id_base: u32,
    pub max_waves_per_simd: u32,
    pub lds_size_in_kb: u32,
    pub gds_size_in_kb: u32,
    pub num_gws: u32,
    pub wave_front_size: u32,
    pub num_xcc: u32,
    pub cu_count: u32,
    pub array_count: u32,
    pub num_shader_banks: u32,
    pub simd_arrays_per_engine: u32,
    pub cu_per_simd_array: u32,
    pub simd_per_cu: u32,
    pub max_slots_scratch_cu: u32,
    pub gfx_target_version: u32,
    pub vendor_id: u16,
    pub device_id: u16,
    pub location_id: u32,
    pub domain: u32,
    pub drm_render_minor: u32,
    pub num_sdma_engines: u32,
    pub num_sdma_xgmi_engines: u32,
    pub num_sdma_queues_per_engine: u32,
    pub num_cp_queues: u32,
    pub max_engine_clk_ccompute: u32,
    pub max_engine_clk_fcompute: u32,
    // HSA_ENGINE_VERSION sdma_fw_version (2x u16 = 4 bytes)
    pub sdma_fw_version_uCodeSDMA: u16,
    pub sdma_fw_version_uCodeRes: u16,
    // HSA_ENGINE_ID fw_version (2x u32 = 8 bytes)
    pub fw_version_uCode: u32,
    pub fw_version_Major: u32,
    // HSA_CAPABILITY capability (u64)
    pub capability: u64,
    pub cu_per_engine: u32,
    pub max_waves_per_cu: u32,
    pub family_id: u32,
    pub workgroup_max_size: u32,
    pub grid_max_size: u32,
    pub _pad_to_align: u32, // padding to align local_mem_size to 8 bytes
    pub local_mem_size: u64,
    pub hive_id: u64,
    pub gpu_id: u64,
    pub workgroup_max_dim: rocprofiler_dim3_t,
    pub grid_max_dim: rocprofiler_dim3_t,
    // These are pointers (8 bytes each)
    pub mem_banks: *const c_void,
    pub caches: *const c_void,
    pub io_links: *const c_void,
    pub name: *const c_char,
    pub vendor_name: *const c_char,
    pub product_name: *const c_char,
    pub model_name: *const c_char,
    pub node_id: u32,
    pub logical_node_id: i32,
    pub logical_node_type_id: i32,
    pub reserved_padding0: i32,
}

impl rocprofiler_agent_v0_t {
    #[inline]
    pub fn agent_type(&self) -> rocprofiler_agent_type_t {
        self.r#type
    }
}

pub type rocprofiler_agent_t = rocprofiler_agent_v0_t;

// ---------------------------------------------------------------------------
// Registration types
// ---------------------------------------------------------------------------

/// Client identifier (the `handle` field is const in C; we use u32 in Rust).
#[repr(C)]
pub struct rocprofiler_client_id_t {
    pub name: *const c_char,
    pub handle: u32,
}

/// The finalize function pointer takes `rocprofiler_client_id_t` BY VALUE.
pub type rocprofiler_client_finalize_t = Option<unsafe extern "C" fn(rocprofiler_client_id_t)>;

pub type rocprofiler_tool_initialize_t =
    Option<unsafe extern "C" fn(rocprofiler_client_finalize_t, *mut c_void) -> i32>;

pub type rocprofiler_tool_finalize_t = Option<unsafe extern "C" fn(*mut c_void)>;

#[repr(C)]
pub struct rocprofiler_tool_configure_result_t {
    pub size: usize,
    pub initialize: rocprofiler_tool_initialize_t,
    pub finalize: rocprofiler_tool_finalize_t,
    pub tool_data: *mut c_void,
}

// ---------------------------------------------------------------------------
// Callback function types
// ---------------------------------------------------------------------------

pub type rocprofiler_buffer_tracing_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_context_id_t,
        rocprofiler_buffer_id_t,
        *mut *mut rocprofiler_record_header_t,
        usize,
        *mut c_void,
        u64,
    ),
>;

pub type rocprofiler_callback_tracing_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_callback_tracing_record_t,
        *mut rocprofiler_user_data_t,
        *mut c_void,
    ),
>;

pub type rocprofiler_query_available_agents_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_agent_version_t,
        *mut *const c_void,
        usize,
        *mut c_void,
    ) -> rocprofiler_status_t,
>;

pub type rocprofiler_available_counters_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_agent_id_t,
        *mut rocprofiler_counter_id_t,
        usize,
        *mut c_void,
    ) -> rocprofiler_status_t,
>;

pub type rocprofiler_dispatch_counting_service_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_dispatch_counting_service_data_t,
        *mut rocprofiler_counter_config_id_t,
        *mut rocprofiler_user_data_t,
        *mut c_void,
    ),
>;

pub type rocprofiler_dispatch_counting_record_cb_t = Option<
    unsafe extern "C" fn(
        rocprofiler_dispatch_counting_service_data_t,
        *mut rocprofiler_counter_record_t,
        usize,
        rocprofiler_user_data_t,
        *mut c_void,
    ),
>;

// ---------------------------------------------------------------------------
// FFI function declarations (rocprofiler-sdk C API)
//
// When the `stubs` feature is enabled, these are provided by the stub library
// linked at build time. Otherwise, the dispatch module resolves them via
// dlsym at runtime and re-exports wrappers with matching signatures.
// ---------------------------------------------------------------------------

#[cfg(feature = "stubs")]
extern "C" {
    pub fn rocprofiler_create_context(
        context_id: *mut rocprofiler_context_id_t,
    ) -> rocprofiler_status_t;

    pub fn rocprofiler_start_context(context_id: rocprofiler_context_id_t) -> rocprofiler_status_t;

    pub fn rocprofiler_stop_context(context_id: rocprofiler_context_id_t) -> rocprofiler_status_t;

    pub fn rocprofiler_create_buffer(
        context: rocprofiler_context_id_t,
        size: usize,
        watermark: usize,
        policy: rocprofiler_buffer_policy_t,
        callback: rocprofiler_buffer_tracing_cb_t,
        callback_data: *mut c_void,
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
        callback_args: *mut c_void,
    ) -> rocprofiler_status_t;

    pub fn rocprofiler_query_available_agents(
        version: rocprofiler_agent_version_t,
        callback: rocprofiler_query_available_agents_cb_t,
        agent_size: usize,
        user_data: *mut c_void,
    ) -> rocprofiler_status_t;

    pub fn rocprofiler_iterate_agent_supported_counters(
        agent_id: rocprofiler_agent_id_t,
        cb: rocprofiler_available_counters_cb_t,
        user_data: *mut c_void,
    ) -> rocprofiler_status_t;

    pub fn rocprofiler_query_counter_info(
        counter_id: rocprofiler_counter_id_t,
        version: rocprofiler_counter_info_version_id_t,
        info: *mut c_void,
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
        dispatch_callback_args: *mut c_void,
        record_callback: rocprofiler_dispatch_counting_record_cb_t,
        record_callback_args: *mut c_void,
    ) -> rocprofiler_status_t;
}
