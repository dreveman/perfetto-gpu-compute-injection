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

//! NVML FFI types and dispatch.
//!
//! When built with `--features stubs`, links against stub implementations
//! from `stubs.cpp`. Otherwise, loads `libnvidia-ml.so.1` at runtime via
//! `dlsym`, matching the dispatch pattern from `perfetto-cuda-injection`.

use std::os::raw::{c_char, c_int, c_void};

#[cfg(not(feature = "stubs"))]
use std::sync::OnceLock;

/// NVML return type (0 = success).
#[allow(non_camel_case_types)]
pub type nvmlReturn_t = u32;

/// Opaque NVML device handle.
#[allow(non_camel_case_types)]
pub type nvmlDevice_t = *mut c_void;

/// NVML clock type.
#[allow(non_camel_case_types)]
pub type nvmlClockType_t = u32;

/// NVML device architecture type.
#[allow(non_camel_case_types)]
pub type nvmlDeviceArchitecture_t = u32;

/// NVML success return value.
pub const NVML_SUCCESS: nvmlReturn_t = 0;

/// NVML temperature sensor type.
#[allow(non_camel_case_types)]
pub type nvmlTemperatureSensors_t = u32;

/// GPU temperature sensor.
pub const NVML_TEMPERATURE_GPU: nvmlTemperatureSensors_t = 0;

/// NVML utilization information.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct nvmlUtilization_t {
    pub gpu: u32,
    pub memory: u32,
}

/// Graphics clock type.
pub const NVML_CLOCK_GRAPHICS: nvmlClockType_t = 0;

/// SM clock type.
pub const NVML_CLOCK_SM: nvmlClockType_t = 1;

/// Memory clock type.
pub const NVML_CLOCK_MEM: nvmlClockType_t = 2;

// Architecture constants.
pub const NVML_DEVICE_ARCH_KEPLER: nvmlDeviceArchitecture_t = 2;
pub const NVML_DEVICE_ARCH_MAXWELL: nvmlDeviceArchitecture_t = 3;
pub const NVML_DEVICE_ARCH_PASCAL: nvmlDeviceArchitecture_t = 4;
pub const NVML_DEVICE_ARCH_VOLTA: nvmlDeviceArchitecture_t = 5;
pub const NVML_DEVICE_ARCH_TURING: nvmlDeviceArchitecture_t = 6;
pub const NVML_DEVICE_ARCH_AMPERE: nvmlDeviceArchitecture_t = 7;
pub const NVML_DEVICE_ARCH_ADA: nvmlDeviceArchitecture_t = 8;
pub const NVML_DEVICE_ARCH_HOPPER: nvmlDeviceArchitecture_t = 9;
pub const NVML_DEVICE_ARCH_BLACKWELL: nvmlDeviceArchitecture_t = 11;

/// NVML memory information.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct nvmlMemory_t {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// NVML PCI information.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct nvmlPciInfo_t {
    pub bus_id_legacy: [c_char; 16],
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub pci_device_id: u32,
    pub pci_subsystem_id: u32,
    pub bus_id: [c_char; 32],
}

// ---------------------------------------------------------------------------
// Stubs path: link against stubs.cpp
// ---------------------------------------------------------------------------

#[cfg(feature = "stubs")]
extern "C" {
    pub fn nvmlInit_v2() -> nvmlReturn_t;
    pub fn nvmlDeviceGetCount_v2(deviceCount: *mut u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByIndex_v2(index: u32, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetClockInfo(
        device: nvmlDevice_t,
        type_: nvmlClockType_t,
        clock: *mut u32,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetName(device: nvmlDevice_t, name: *mut c_char, length: u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetMemoryInfo(device: nvmlDevice_t, memory: *mut nvmlMemory_t)
        -> nvmlReturn_t;
    pub fn nvmlDeviceGetUUID(device: nvmlDevice_t, uuid: *mut c_char, length: u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPciInfo_v3(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetArchitecture(
        device: nvmlDevice_t,
        arch: *mut nvmlDeviceArchitecture_t,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetCudaComputeCapability(
        device: nvmlDevice_t,
        major: *mut c_int,
        minor: *mut c_int,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetBoardPartNumber(
        device: nvmlDevice_t,
        part_number: *mut c_char,
        length: u32,
    ) -> nvmlReturn_t;
    pub fn nvmlSystemGetDriverVersion(version: *mut c_char, length: u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetVbiosVersion(
        device: nvmlDevice_t,
        version: *mut c_char,
        length: u32,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPowerManagementDefaultLimit(
        device: nvmlDevice_t,
        default_limit: *mut u32,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetNumGpuCores(device: nvmlDevice_t, num_cores: *mut u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetTemperature(
        device: nvmlDevice_t,
        sensor_type: nvmlTemperatureSensors_t,
        temp: *mut u32,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPowerUsage(device: nvmlDevice_t, power: *mut u32) -> nvmlReturn_t;
    pub fn nvmlDeviceGetUtilizationRates(
        device: nvmlDevice_t,
        utilization: *mut nvmlUtilization_t,
    ) -> nvmlReturn_t;
}

// ---------------------------------------------------------------------------
// Runtime dlsym path
// ---------------------------------------------------------------------------

#[cfg(not(feature = "stubs"))]
mod runtime {
    use super::*;

    const RTLD_LAZY: c_int = 0x1;

    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    static NVML_HANDLE: OnceLock<usize> = OnceLock::new();

    /// Returns a handle to `libnvidia-ml.so.1`, loading it on demand.
    /// Also calls `nvmlInit_v2()` to initialize the library.
    fn get_nvml_handle() -> *mut c_void {
        *NVML_HANDLE.get_or_init(|| {
            let handle = unsafe { dlopen(c"libnvidia-ml.so.1".as_ptr(), RTLD_LAZY) };
            if handle.is_null() {
                crate::perfetto_elog!(
                    "could not load libnvidia-ml.so.1; GPU probes will not function"
                );
                return 0;
            }
            // NVML requires initialization before any other calls.
            let init_sym = unsafe { dlsym(handle, c"nvmlInit_v2".as_ptr()) };
            if init_sym.is_null() {
                crate::perfetto_elog!("could not resolve nvmlInit_v2");
                return 0;
            }
            let init_fn: unsafe extern "C" fn() -> nvmlReturn_t =
                unsafe { std::mem::transmute(init_sym) };
            let ret = unsafe { init_fn() };
            if ret != 0 {
                crate::perfetto_elog!("nvmlInit_v2 failed (error {})", ret);
                return 0;
            }
            handle as usize
        }) as *mut c_void
    }

    const NVML_ERROR_UNINITIALIZED: nvmlReturn_t = 1;

    /// Initializes NVML by loading `libnvidia-ml.so.1` and calling `nvmlInit_v2`.
    ///
    /// # Safety
    ///
    /// Must be called before any other NVML functions.
    #[allow(non_snake_case)]
    pub unsafe fn nvmlInit_v2() -> nvmlReturn_t {
        if get_nvml_handle().is_null() {
            NVML_ERROR_UNINITIALIZED
        } else {
            NVML_SUCCESS
        }
    }

    macro_rules! dispatch_fn {
        (
            $fn_name:ident ( $($arg_name:ident : $arg_ty:ty),* $(,)? ) -> $ret_ty:ty
        ) => {
            /// Dynamically dispatched NVML function.
            ///
            /// # Safety
            ///
            /// Caller must uphold the NVML C API contract for this function
            /// (valid handles, non-null out-pointers, etc.).
            #[allow(non_snake_case)]
            pub unsafe fn $fn_name ( $($arg_name: $arg_ty),* ) -> $ret_ty {
                static CACHE: OnceLock<Option<usize>> = OnceLock::new();
                let fp = *CACHE.get_or_init(|| {
                    let handle = get_nvml_handle();
                    if handle.is_null() {
                        return None;
                    }
                    let sym = dlsym(
                        handle,
                        concat!(stringify!($fn_name), "\0").as_ptr() as *const c_char,
                    );
                    if sym.is_null() {
                        crate::perfetto_elog!(
                            "could not resolve symbol: {}",
                            stringify!($fn_name)
                        );
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
                    None => NVML_ERROR_UNINITIALIZED,
                }
            }
        };
    }

    dispatch_fn!(nvmlDeviceGetCount_v2(deviceCount: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetHandleByIndex_v2(index: u32, device: *mut nvmlDevice_t) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetClockInfo(device: nvmlDevice_t, type_: nvmlClockType_t, clock: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetName(device: nvmlDevice_t, name: *mut c_char, length: u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetMemoryInfo(device: nvmlDevice_t, memory: *mut nvmlMemory_t) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetUUID(device: nvmlDevice_t, uuid: *mut c_char, length: u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetPciInfo_v3(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetArchitecture(device: nvmlDevice_t, arch: *mut nvmlDeviceArchitecture_t) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetCudaComputeCapability(device: nvmlDevice_t, major: *mut c_int, minor: *mut c_int) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetBoardPartNumber(device: nvmlDevice_t, part_number: *mut c_char, length: u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlSystemGetDriverVersion(version: *mut c_char, length: u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetVbiosVersion(device: nvmlDevice_t, version: *mut c_char, length: u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetPowerManagementDefaultLimit(device: nvmlDevice_t, default_limit: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetNumGpuCores(device: nvmlDevice_t, num_cores: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetTemperature(device: nvmlDevice_t, sensor_type: nvmlTemperatureSensors_t, temp: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetPowerUsage(device: nvmlDevice_t, power: *mut u32) -> nvmlReturn_t);
    dispatch_fn!(nvmlDeviceGetUtilizationRates(device: nvmlDevice_t, utilization: *mut nvmlUtilization_t) -> nvmlReturn_t);
}

#[cfg(not(feature = "stubs"))]
pub use runtime::*;
