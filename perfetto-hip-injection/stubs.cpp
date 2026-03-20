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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Stub implementations for rocprofiler-sdk functions.
// These are self-contained and do not require ROCm headers.

typedef int rocprofiler_status_t;
#define ROCPROFILER_STATUS_SUCCESS 0

typedef struct {
  uint64_t handle;
} rocprofiler_context_id_t;
typedef struct {
  uint64_t handle;
} rocprofiler_buffer_id_t;
typedef struct {
  uint64_t handle;
} rocprofiler_agent_id_t;

typedef int rocprofiler_buffer_policy_t;
typedef int rocprofiler_buffer_tracing_kind_t;
typedef int rocprofiler_callback_tracing_kind_t;
typedef int rocprofiler_tracing_operation_t;
typedef int rocprofiler_agent_version_t;

typedef struct {
  const char* name;
  const uint32_t handle;
} rocprofiler_client_id_t;
typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);
typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t,
                                             void*);
typedef void (*rocprofiler_tool_finalize_t)(void*);
typedef struct {
  size_t size;
  rocprofiler_tool_initialize_t initialize;
  rocprofiler_tool_finalize_t finalize;
  void* tool_data;
} rocprofiler_tool_configure_result_t;
typedef rocprofiler_tool_configure_result_t* (*rocprofiler_configure_func_t)(
    uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t,
                                                rocprofiler_buffer_id_t, void**,
                                                size_t, void*, uint64_t);
typedef void (*rocprofiler_callback_tracing_cb_t)(void*, void*, void*);

typedef rocprofiler_status_t (*rocprofiler_query_available_agents_cb_t)(
    rocprofiler_agent_version_t, const void**, size_t, void*);

extern "C" {

rocprofiler_status_t rocprofiler_create_context(rocprofiler_context_id_t* ctx) {
  if (ctx) ctx->handle = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_start_context(rocprofiler_context_id_t ctx) {
  (void)ctx;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_stop_context(rocprofiler_context_id_t ctx) {
  (void)ctx;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_context_is_valid(rocprofiler_context_id_t ctx,
                                                  int* status) {
  (void)ctx;
  if (status) *status = 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_create_buffer(
    rocprofiler_context_id_t context, size_t size, size_t watermark,
    rocprofiler_buffer_policy_t policy,
    rocprofiler_buffer_tracing_cb_t callback, void* callback_data,
    rocprofiler_buffer_id_t* buffer_id) {
  (void)context;
  (void)size;
  (void)watermark;
  (void)policy;
  (void)callback;
  (void)callback_data;
  if (buffer_id) buffer_id->handle = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_flush_buffer(
    rocprofiler_buffer_id_t buffer_id) {
  (void)buffer_id;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_destroy_buffer(
    rocprofiler_buffer_id_t buffer_id) {
  (void)buffer_id;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_configure_buffer_tracing_service(
    rocprofiler_context_id_t context_id, rocprofiler_buffer_tracing_kind_t kind,
    rocprofiler_tracing_operation_t* operations, size_t operations_count,
    rocprofiler_buffer_id_t buffer_id) {
  (void)context_id;
  (void)kind;
  (void)operations;
  (void)operations_count;
  (void)buffer_id;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_configure_callback_tracing_service(
    rocprofiler_context_id_t context_id,
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t* operations, size_t operations_count,
    rocprofiler_callback_tracing_cb_t callback, void* callback_args) {
  (void)context_id;
  (void)kind;
  (void)operations;
  (void)operations_count;
  (void)callback;
  (void)callback_args;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_query_available_agents(
    rocprofiler_agent_version_t version,
    rocprofiler_query_available_agents_cb_t callback, size_t agent_size,
    void* user_data) {
  (void)version;
  (void)agent_size;
  if (callback) callback(version, nullptr, 0, user_data);
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_is_initialized(int* status) {
  if (status) *status = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_is_finalized(int* status) {
  if (status) *status = 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_force_configure(
    rocprofiler_configure_func_t configure_func) {
  (void)configure_func;
  return ROCPROFILER_STATUS_SUCCESS;
}

const char* rocprofiler_get_status_name(rocprofiler_status_t status) {
  (void)status;
  return "ROCPROFILER_STATUS_SUCCESS";
}

const char* rocprofiler_get_status_string(rocprofiler_status_t status) {
  (void)status;
  return "Success";
}

// ---------------------------------------------------------------------------
// Counter API stubs
// ---------------------------------------------------------------------------

typedef struct {
  uint64_t handle;
} rocprofiler_counter_id_t;
typedef struct {
  uint64_t handle;
} rocprofiler_counter_config_id_t;

typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(
    rocprofiler_agent_id_t, rocprofiler_counter_id_t*, size_t, void*);

typedef void (*rocprofiler_dispatch_counting_service_cb_t)(
    void*, rocprofiler_counter_config_id_t*, void*, void*);

typedef void (*rocprofiler_dispatch_counting_record_cb_t)(void*, void*, size_t,
                                                          uint64_t, void*);

rocprofiler_status_t rocprofiler_iterate_agent_supported_counters(
    rocprofiler_agent_id_t agent_id, rocprofiler_available_counters_cb_t cb,
    void* user_data) {
  (void)agent_id;
  if (cb) cb(agent_id, nullptr, 0, user_data);
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_query_counter_info(
    rocprofiler_counter_id_t counter_id, int version, void* info) {
  (void)counter_id;
  (void)version;
  (void)info;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_query_record_counter_id(
    uint64_t id, rocprofiler_counter_id_t* counter_id) {
  (void)id;
  if (counter_id) counter_id->handle = 0;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_create_counter_config(
    rocprofiler_agent_id_t agent_id, rocprofiler_counter_id_t* counters_list,
    size_t counters_count, rocprofiler_counter_config_id_t* config_id) {
  (void)agent_id;
  (void)counters_list;
  (void)counters_count;
  if (config_id) config_id->handle = 1;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_destroy_counter_config(
    rocprofiler_counter_config_id_t config_id) {
  (void)config_id;
  return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void* dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t record_callback,
    void* record_callback_args) {
  (void)context_id;
  (void)dispatch_callback;
  (void)dispatch_callback_args;
  (void)record_callback;
  (void)record_callback_args;
  return ROCPROFILER_STATUS_SUCCESS;
}

static const char* kUnknownOpName = "unknown";

rocprofiler_status_t rocprofiler_query_buffer_tracing_kind_operation_name(
    rocprofiler_buffer_tracing_kind_t kind,
    rocprofiler_tracing_operation_t operation, const char** name,
    size_t* name_len) {
  (void)kind;
  (void)operation;
  if (name) *name = kUnknownOpName;
  if (name_len) *name_len = 7;
  return ROCPROFILER_STATUS_SUCCESS;
}

}  // extern "C"
