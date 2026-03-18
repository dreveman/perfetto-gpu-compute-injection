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

//! Runtime dlsym dispatch for CUDA driver API and CUPTI functions.
//!
//! Instead of linking against `-lcuda` and `-lcupti` (which bakes
//! the build host's RPATH into the injection .so and prevents finding
//! the libraries at runtime), we resolve all CUDA/CUPTI functions via
//! `dlsym` at runtime.  `libcuda.so` is expected to be already loaded
//! by the host application; `libcupti.so` is loaded on demand.

use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;

// Re-export all types and constants from the raw bindings so that callers
// importing `dispatch::*` also get everything they need.  The dispatch
// functions defined below take precedence over the glob-imported extern "C"
// function declarations from raw_bindings.
#[allow(unused_imports)]
pub use super::raw_bindings::*;

// ---------------------------------------------------------------------------
// dlopen / dlsym FFI
// ---------------------------------------------------------------------------

const RTLD_NOLOAD: c_int = 0x4;
const RTLD_LAZY: c_int = 0x1;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

// ---------------------------------------------------------------------------
// Library handles
// ---------------------------------------------------------------------------

static CUDA_HANDLE: OnceLock<usize> = OnceLock::new();
static CUPTI_HANDLE: OnceLock<usize> = OnceLock::new();

/// Returns a handle to `libcuda.so.1`, which must already be loaded by the
/// host application (we use `RTLD_NOLOAD`).
fn get_cuda_handle() -> *mut c_void {
    *CUDA_HANDLE.get_or_init(|| {
        let handle = unsafe { dlopen(c"libcuda.so.1".as_ptr(), RTLD_NOLOAD | RTLD_LAZY) };
        if handle.is_null() {
            eprintln!(
                "==INJECTION== could not find libcuda.so.1 \
                 (expected to be loaded by host application)"
            );
        }
        handle as usize
    }) as *mut c_void
}

/// Returns a handle to `libcupti.so`, loading it on demand.
fn get_cupti_handle() -> *mut c_void {
    *CUPTI_HANDLE.get_or_init(|| {
        let handle = unsafe { dlopen(c"libcupti.so".as_ptr(), RTLD_LAZY) };
        if !handle.is_null() {
            return handle as usize;
        }
        // Try versioned names.
        for name in [c"libcupti.so.13".as_ptr(), c"libcupti.so.12".as_ptr()] {
            let handle = unsafe { dlopen(name, RTLD_LAZY) };
            if !handle.is_null() {
                return handle as usize;
            }
        }
        eprintln!(
            "==INJECTION== could not load libcupti.so; \
             CUPTI may not be installed or not in the library search path"
        );
        0
    }) as *mut c_void
}

// ---------------------------------------------------------------------------
// Helper macros
// ---------------------------------------------------------------------------

macro_rules! dispatch_fn {
    (
        $lib_handle_fn:ident, $err_val:expr,
        $fn_name:ident ( $($arg_name:ident : $arg_ty:ty),* $(,)? ) -> $ret_ty:ty
    ) => {
        #[allow(non_snake_case)]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe fn $fn_name ( $($arg_name: $arg_ty),* ) -> $ret_ty {
            static CACHE: OnceLock<Option<usize>> = OnceLock::new();
            let fp = *CACHE.get_or_init(|| {
                let handle = $lib_handle_fn();
                if handle.is_null() {
                    return None;
                }
                let sym = dlsym(
                    handle,
                    concat!(stringify!($fn_name), "\0").as_ptr() as *const c_char,
                );
                if sym.is_null() {
                    eprintln!(
                        "==INJECTION== Error: could not resolve symbol: {}",
                        stringify!($fn_name)
                    );
                    if let Ok(hint) = std::env::var("INJECTION_CUDA_SYMBOL_LOOKUP_HINT") {
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
                None => $err_val,
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Error constants
// ---------------------------------------------------------------------------

const CUDA_ERROR_UNKNOWN: CUresult = 999;
const CUPTI_ERROR_UNKNOWN: CUptiResult = 999;

// ---------------------------------------------------------------------------
// CUDA Driver API (from libcuda.so)
// ---------------------------------------------------------------------------

dispatch_fn!(get_cuda_handle, CUDA_ERROR_UNKNOWN,
    cuCtxGetDevice(device: *mut CUdevice) -> CUresult);

dispatch_fn!(get_cuda_handle, CUDA_ERROR_UNKNOWN,
    cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult);

dispatch_fn!(get_cuda_handle, CUDA_ERROR_UNKNOWN,
    cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult);

dispatch_fn!(get_cuda_handle, CUDA_ERROR_UNKNOWN,
    cuFuncGetAttribute(
        pi: *mut c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction,
    ) -> CUresult);

dispatch_fn!(get_cuda_handle, CUDA_ERROR_UNKNOWN,
    cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut c_int,
        func: CUfunction,
        blockSize: c_int,
        dynamicSMemSize: usize,
    ) -> CUresult);

// ---------------------------------------------------------------------------
// CUPTI: result / version / utility
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiGetResultString(
        result: CUptiResult,
        str_: *mut *const c_char,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiGetVersion(version: *mut u32) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiGetLastError() -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiGetContextId(context: CUcontext, contextId: *mut u32) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: subscription & callback management
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiSubscribe(
        subscriber: *mut CUpti_SubscriberHandle,
        callback: CUpti_CallbackFunc,
        userdata: *mut c_void,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiEnableCallback(
        enable: u32,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
        cbid: CUpti_CallbackId,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiEnableDomain(
        enable: u32,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
    ) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: activity API
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityEnable(kind: CUpti_ActivityKind) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityDisable(kind: CUpti_ActivityKind) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityRegisterCallbacks(
        funcBufferRequested: CUpti_BuffersCallbackRequestFunc,
        funcBufferCompleted: CUpti_BuffersCallbackCompleteFunc,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityRegisterTimestampCallback(
        funcTimestamp: CUpti_TimestampCallbackFunc,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityFlushAll(flag: u32) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiActivityGetNextRecord(
        buffer: *mut u8,
        validBufferSizeBytes: usize,
        record: *mut *mut CUpti_Activity,
    ) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: profiler initialization
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerInitialize(
        pParams: *mut CUpti_Profiler_Initialize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerDeInitialize(
        pParams: *mut CUpti_Profiler_DeInitialize_Params,
    ) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: profiler host API
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostInitialize(
        pParams: *mut CUpti_Profiler_Host_Initialize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostDeinitialize(
        pParams: *mut CUpti_Profiler_Host_Deinitialize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostGetMetricProperties(
        pParams: *mut CUpti_Profiler_Host_GetMetricProperties_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostConfigAddMetrics(
        pParams: *mut CUpti_Profiler_Host_ConfigAddMetrics_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostGetConfigImageSize(
        pParams: *mut CUpti_Profiler_Host_GetConfigImageSize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostGetConfigImage(
        pParams: *mut CUpti_Profiler_Host_GetConfigImage_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerHostEvaluateToGpuValues(
        pParams: *mut CUpti_Profiler_Host_EvaluateToGpuValues_Params,
    ) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: counter availability & device info
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiProfilerGetCounterAvailability(
        pParams: *mut CUpti_Profiler_GetCounterAvailability_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiDeviceGetChipName(
        pParams: *mut CUpti_Device_GetChipName_Params,
    ) -> CUptiResult);

// ---------------------------------------------------------------------------
// CUPTI: range profiler
// ---------------------------------------------------------------------------

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerEnable(
        pParams: *mut CUpti_RangeProfiler_Enable_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerDisable(
        pParams: *mut CUpti_RangeProfiler_Disable_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerStart(
        pParams: *mut CUpti_RangeProfiler_Start_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerStop(
        pParams: *mut CUpti_RangeProfiler_Stop_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerSetConfig(
        pParams: *mut CUpti_RangeProfiler_SetConfig_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerGetCounterDataSize(
        pParams: *mut CUpti_RangeProfiler_GetCounterDataSize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerCounterDataImageInitialize(
        pParams: *mut CUpti_RangeProfiler_CounterDataImage_Initialize_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerPushRange(
        pParams: *mut CUpti_RangeProfiler_PushRange_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerPopRange(
        pParams: *mut CUpti_RangeProfiler_PopRange_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerDecodeData(
        pParams: *mut CUpti_RangeProfiler_DecodeData_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerGetCounterDataInfo(
        pParams: *mut CUpti_RangeProfiler_GetCounterDataInfo_Params,
    ) -> CUptiResult);

dispatch_fn!(get_cupti_handle, CUPTI_ERROR_UNKNOWN,
    cuptiRangeProfilerCounterDataGetRangeInfo(
        pParams: *mut CUpti_RangeProfiler_CounterData_GetRangeInfo_Params,
    ) -> CUptiResult);
