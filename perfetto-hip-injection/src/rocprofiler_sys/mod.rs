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

// Types and constants: use bindgen-generated output when available,
// otherwise fall back to hand-written bindings.
#[cfg(not(feature = "bindgen"))]
#[path = "bindings.rs"]
mod types;

#[cfg(feature = "bindgen")]
mod types {
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        dead_code
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings_generated.rs"));

    // Short constant aliases: bindgen prefixes enum constants with the type name.
    pub const ROCPROFILER_STATUS_SUCCESS: rocprofiler_status_t =
        rocprofiler_status_t_ROCPROFILER_STATUS_SUCCESS as rocprofiler_status_t;
    pub const ROCPROFILER_BUFFER_CATEGORY_TRACING: u32 =
        rocprofiler_buffer_category_t_ROCPROFILER_BUFFER_CATEGORY_TRACING as u32;
    pub const ROCPROFILER_AGENT_TYPE_GPU: rocprofiler_agent_type_t =
        rocprofiler_agent_type_t_ROCPROFILER_AGENT_TYPE_GPU as rocprofiler_agent_type_t;
    pub const ROCPROFILER_AGENT_INFO_VERSION_0: rocprofiler_agent_version_t =
        rocprofiler_agent_version_t_ROCPROFILER_AGENT_INFO_VERSION_0 as rocprofiler_agent_version_t;
    pub const ROCPROFILER_CALLBACK_PHASE_ENTER: rocprofiler_callback_phase_t =
        rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_ENTER
            as rocprofiler_callback_phase_t;
    pub const ROCPROFILER_CALLBACK_PHASE_LOAD: rocprofiler_callback_phase_t =
        rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_LOAD
            as rocprofiler_callback_phase_t;
    pub const ROCPROFILER_CALLBACK_PHASE_EXIT: rocprofiler_callback_phase_t =
        rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_EXIT
            as rocprofiler_callback_phase_t;
    pub const ROCPROFILER_CALLBACK_PHASE_UNLOAD: rocprofiler_callback_phase_t =
        rocprofiler_callback_phase_t_ROCPROFILER_CALLBACK_PHASE_UNLOAD
            as rocprofiler_callback_phase_t;
    pub const ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT: rocprofiler_callback_tracing_kind_t =
        rocprofiler_callback_tracing_kind_t_ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
            as rocprofiler_callback_tracing_kind_t;
    pub const ROCPROFILER_CODE_OBJECT_LOAD: rocprofiler_tracing_operation_t =
        rocprofiler_code_object_operation_t_ROCPROFILER_CODE_OBJECT_LOAD
            as rocprofiler_tracing_operation_t;
    pub const ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER:
        rocprofiler_tracing_operation_t =
        rocprofiler_code_object_operation_t_ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER
            as rocprofiler_tracing_operation_t;
    pub const ROCPROFILER_BUFFER_TRACING_MEMORY_COPY: rocprofiler_buffer_tracing_kind_t =
        rocprofiler_buffer_tracing_kind_t_ROCPROFILER_BUFFER_TRACING_MEMORY_COPY
            as rocprofiler_buffer_tracing_kind_t;
    pub const ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: rocprofiler_buffer_tracing_kind_t =
        rocprofiler_buffer_tracing_kind_t_ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
            as rocprofiler_buffer_tracing_kind_t;
    pub const ROCPROFILER_BUFFER_POLICY_LOSSLESS: rocprofiler_buffer_policy_t =
        rocprofiler_buffer_policy_t_ROCPROFILER_BUFFER_POLICY_LOSSLESS
            as rocprofiler_buffer_policy_t;
    pub const ROCPROFILER_COUNTER_INFO_VERSION_0: rocprofiler_counter_info_version_id_t =
        rocprofiler_counter_info_version_id_t_ROCPROFILER_COUNTER_INFO_VERSION_0
            as rocprofiler_counter_info_version_id_t;

    // Accessor methods for bindgen structs that use anonymous unions or
    // different field names than the hand-written bindings.
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
}

pub use types::*;

// Runtime dlsym dispatch replaces the extern function declarations when not
// using stubs. The dispatch module's pub functions shadow the extern
// declarations re-exported from `types`.
#[cfg(not(feature = "stubs"))]
mod dispatch;
#[cfg(not(feature = "stubs"))]
pub use dispatch::*;
