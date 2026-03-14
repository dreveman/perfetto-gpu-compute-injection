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

use crate::cupti_profiler::bindings::*;
use perfetto_gpu_compute_injection::injection_log;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Safe wrapper for low-level CUPTI Profiler initialization.
pub struct Profiler {}

impl Profiler {
    /// Initialize the CUPTI profiler.
    ///
    /// This function calls `cuptiProfilerInitialize` safely.
    pub fn initialize() -> Result<(), CUptiResult> {
        let mut params: CUpti_Profiler_Initialize_Params = unsafe { std::mem::zeroed() };
        params.structSize =
            struct_size_up_to!(CUpti_Profiler_Initialize_Params, pPriv: *const c_void);
        check_cupti!(unsafe { cuptiProfilerInitialize(&mut params) });
        Ok(())
    }
}

/// Manages the host-side CUPTI profiler object.
pub struct ProfilerHost {
    chip_name: String,
    counter_availability_image: Vec<u8>,
    profiler_type: CUpti_ProfilerType,
    pub host_object: *mut CUpti_Profiler_Host_Object,
}

unsafe impl Send for ProfilerHost {}
unsafe impl Sync for ProfilerHost {}

impl ProfilerHost {
    /// Creates a new, uninitialized `ProfilerHost`.
    pub fn new() -> Self {
        Self {
            chip_name: String::new(),
            counter_availability_image: Vec::new(),
            profiler_type: CUpti_ProfilerType_CUPTI_PROFILER_TYPE_RANGE_PROFILER,
            host_object: ptr::null_mut(),
        }
    }

    /// Sets up the profiler host with the given chip name and counter availability.
    ///
    /// Initializes the profiler and creates a host object.
    pub fn setup(
        &mut self,
        chip_name: &str,
        counter_availability_image: Vec<u8>,
        profiler_type: CUpti_ProfilerType,
    ) -> Result<(), CUptiResult> {
        if !self.host_object.is_null() {
            injection_log!("ProfilerHost already initialized");
            return Ok(());
        }
        Profiler::initialize()?;
        self.chip_name = chip_name.to_string();
        self.counter_availability_image = counter_availability_image;
        self.profiler_type = profiler_type;
        let c_chip_name = CString::new(self.chip_name.clone()).unwrap();
        let mut params: CUpti_Profiler_Host_Initialize_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_Profiler_Host_Initialize_Params, pHostObject: *mut CUpti_Profiler_Host_Object);
        params.profilerType = profiler_type;
        params.pChipName = c_chip_name.as_ptr();
        params.pCounterAvailabilityImage = self.counter_availability_image.as_ptr();
        check_cupti!(unsafe { cuptiProfilerHostInitialize(&mut params) });
        self.host_object = params.pHostObject;
        Ok(())
    }

    pub fn teardown(&mut self) -> Result<(), CUptiResult> {
        if self.host_object.is_null() {
            return Ok(());
        }
        let mut params: CUpti_Profiler_Host_Deinitialize_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_Profiler_Host_Deinitialize_Params, pHostObject: *mut CUpti_Profiler_Host_Object);
        params.pHostObject = self.host_object;
        check_cupti!(unsafe { cuptiProfilerHostDeinitialize(&mut params) });
        self.host_object = ptr::null_mut();
        Ok(())
    }

    /// Validates each metric name against the profiler host object and returns
    /// only the metrics that are available on this GPU.
    pub fn filter_valid_metrics(&self, metric_names: &[String]) -> Vec<String> {
        metric_names
            .iter()
            .filter(|name| {
                let c_name = CString::new(name.as_str()).unwrap();
                let mut params: CUpti_Profiler_Host_GetMetricProperties_Params =
                    unsafe { std::mem::zeroed() };
                params.structSize = struct_size_up_to!(
                    CUpti_Profiler_Host_GetMetricProperties_Params,
                    metricCollectionScope: CUpti_MetricCollectionScope
                );
                params.pHostObject = self.host_object;
                params.pMetricName = c_name.as_ptr();
                let result = unsafe { cuptiProfilerHostGetMetricProperties(&mut params) };
                if result != CUptiResult_CUPTI_SUCCESS {
                    injection_log!("metric '{}' not available, skipping", name);
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Creates a configuration image for the specified metrics.
    pub fn create_config_image(&mut self, metric_names: &[String]) -> Result<Vec<u8>, CUptiResult> {
        let c_metric_names: Vec<CString> = metric_names
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let mut c_metric_ptrs: Vec<*const c_char> =
            c_metric_names.iter().map(|s| s.as_ptr()).collect();
        check_cupti!(unsafe {
            let mut params: CUpti_Profiler_Host_ConfigAddMetrics_Params = std::mem::zeroed();
            params.structSize =
                struct_size_up_to!(CUpti_Profiler_Host_ConfigAddMetrics_Params, numMetrics: usize);
            params.pHostObject = self.host_object;
            params.ppMetricNames = c_metric_ptrs.as_mut_ptr();
            params.numMetrics = metric_names.len();
            cuptiProfilerHostConfigAddMetrics(&mut params)
        });
        let mut params_size: CUpti_Profiler_Host_GetConfigImageSize_Params =
            unsafe { std::mem::zeroed() };
        params_size.structSize = struct_size_up_to!(CUpti_Profiler_Host_GetConfigImageSize_Params, configImageSize: usize);
        params_size.pHostObject = self.host_object;
        check_cupti!(unsafe { cuptiProfilerHostGetConfigImageSize(&mut params_size) });
        let mut config_image = vec![0u8; params_size.configImageSize];
        let mut params_img: CUpti_Profiler_Host_GetConfigImage_Params =
            unsafe { std::mem::zeroed() };
        params_img.structSize =
            struct_size_up_to!(CUpti_Profiler_Host_GetConfigImage_Params, pConfigImage: *mut u8);
        params_img.pHostObject = self.host_object;
        params_img.pConfigImage = config_image.as_mut_ptr();
        params_img.configImageSize = config_image.len();
        check_cupti!(unsafe { cuptiProfilerHostGetConfigImage(&mut params_img) });
        Ok(config_image)
    }
}

impl Drop for ProfilerHost {
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}

impl Default for ProfilerHost {
    fn default() -> Self {
        Self::new()
    }
}

/// Retrieves the chip name for a given device index.
pub fn get_chip_name(device_index: usize) -> Result<String, CUptiResult> {
    let mut params: CUpti_Device_GetChipName_Params = unsafe { std::mem::zeroed() };
    params.structSize =
        struct_size_up_to!(CUpti_Device_GetChipName_Params, pChipName: *const c_char);
    params.deviceIndex = device_index;
    check_cupti!(unsafe { cuptiDeviceGetChipName(&mut params) });
    let c_str = unsafe { CStr::from_ptr(params.pChipName) };
    Ok(c_str.to_string_lossy().into_owned())
}

/// Gets the counter availability image for the current context.
/// # Safety
///
/// The `ctx` pointer must be a valid CUDA context.
pub unsafe fn get_counter_availability_image(ctx: CUcontext) -> Result<Vec<u8>, CUptiResult> {
    let mut params: CUpti_Profiler_GetCounterAvailability_Params = unsafe { std::mem::zeroed() };
    params.structSize = struct_size_up_to!(CUpti_Profiler_GetCounterAvailability_Params, pCounterAvailabilityImage: *mut u8);
    params.ctx = ctx;
    check_cupti!(unsafe { cuptiProfilerGetCounterAvailability(&mut params) });
    let mut image = vec![0u8; params.counterAvailabilityImageSize];
    params.pCounterAvailabilityImage = image.as_mut_ptr();
    check_cupti!(unsafe { cuptiProfilerGetCounterAvailability(&mut params) });
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_host_initialization() {
        let host = ProfilerHost::new();
        assert!(
            host.host_object.is_null(),
            "Host object should be null initially"
        );
    }
}
