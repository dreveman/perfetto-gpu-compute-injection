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
use crate::cupti_profiler::profiler::{
    get_chip_name, get_counter_availability_image, ProfilerHost,
};
use perfetto_gpu_compute_injection::injection_log;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

/// Manages on-device range profiling sessions.
pub struct RangeProfiler {
    context: CUcontext,
    range_profiler_object: *mut CUpti_RangeProfiler_Object,
    pub config_image: Vec<u8>,
    pub pass_index: usize,
    pub target_nesting_level: usize,
    pub is_all_pass_submitted: bool,
    validated_metric_names: Vec<String>,
}

unsafe impl Send for RangeProfiler {}
unsafe impl Sync for RangeProfiler {}

impl RangeProfiler {
    /// Creates a new `RangeProfiler` for the given CUDA context.
    pub fn new(ctx: CUcontext) -> Self {
        Self {
            context: ctx,
            range_profiler_object: ptr::null_mut(),
            config_image: Vec::new(),
            pass_index: 0,
            target_nesting_level: 0,
            is_all_pass_submitted: false,
            validated_metric_names: Vec::new(),
        }
    }

    /// Enables the range profiler on the device.
    pub fn enable(&mut self) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_Enable_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_Enable_Params, pRangeProfilerObject: *mut CUpti_RangeProfiler_Object);
        params.ctx = self.context;
        check_cupti!(unsafe { cuptiRangeProfilerEnable(&mut params) });
        self.range_profiler_object = params.pRangeProfilerObject;
        Ok(())
    }

    /// Disables the range profiler.
    pub fn disable(&mut self) -> Result<(), CUptiResult> {
        if self.range_profiler_object.is_null() {
            return Ok(());
        }
        let mut params: CUpti_RangeProfiler_Disable_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_Disable_Params, pRangeProfilerObject: *mut CUpti_RangeProfiler_Object);
        params.pRangeProfilerObject = self.range_profiler_object;
        check_cupti!(unsafe { cuptiRangeProfilerDisable(&mut params) });
        self.range_profiler_object = ptr::null_mut();
        Ok(())
    }

    /// Starts a profiling session.
    pub fn start(&self) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_Start_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_Start_Params, pRangeProfilerObject: *mut CUpti_RangeProfiler_Object);
        params.pRangeProfilerObject = self.range_profiler_object;
        check_cupti!(unsafe { cuptiRangeProfilerStart(&mut params) });
        Ok(())
    }

    /// Stops the profiling session.
    pub fn stop(&mut self) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_Stop_Params = unsafe { std::mem::zeroed() };
        params.structSize =
            struct_size_up_to!(CUpti_RangeProfiler_Stop_Params, isAllPassSubmitted: u8);
        params.pRangeProfilerObject = self.range_profiler_object;
        check_cupti!(unsafe { cuptiRangeProfilerStop(&mut params) });
        self.pass_index = params.passIndex;
        self.target_nesting_level = params.targetNestingLevel;
        self.is_all_pass_submitted = params.isAllPassSubmitted != 0;
        Ok(())
    }

    /// Sets the configuration for the range profiler, including metrics to collect.
    pub fn set_config(
        &mut self,
        metric_names: &[String],
        counter_data_image: &mut Vec<u8>,
        max_num_ranges: usize,
        replay_mode: CUpti_ProfilerReplayMode,
    ) -> Result<(), CUptiResult> {
        let mut host = ProfilerHost::new();
        let mut device: CUdevice = 0;
        unsafe {
            cuCtxGetDevice(&mut device);
        }
        let chip_name = get_chip_name(device as usize)?;
        let counter_avail = unsafe { get_counter_availability_image(self.context)? };
        host.setup(
            &chip_name,
            counter_avail,
            CUpti_ProfilerType_CUPTI_PROFILER_TYPE_RANGE_PROFILER,
        )?;
        let valid_metrics = host.filter_valid_metrics(metric_names);
        if valid_metrics.is_empty() {
            injection_log!("no valid metrics available, skipping counter collection");
            return Ok(());
        }
        self.config_image = host.create_config_image(&valid_metrics)?;
        if counter_data_image.is_empty() {
            self.create_counter_data_image(max_num_ranges, &valid_metrics, counter_data_image)?;
        }
        self.validated_metric_names = valid_metrics;
        let mut params: CUpti_RangeProfiler_SetConfig_Params = unsafe { std::mem::zeroed() };
        params.structSize =
            struct_size_up_to!(CUpti_RangeProfiler_SetConfig_Params, targetNestingLevel: u16);
        params.pRangeProfilerObject = self.range_profiler_object;
        params.pConfig = self.config_image.as_ptr();
        params.configSize = self.config_image.len();
        params.pCounterDataImage = counter_data_image.as_mut_ptr();
        params.counterDataImageSize = counter_data_image.len();
        params.range = CUpti_ProfilerRange_CUPTI_AutoRange;
        params.replayMode = replay_mode;
        params.maxRangesPerPass = max_num_ranges;
        params.numNestingLevels = 1;
        params.minNestingLevel = 1;
        params.passIndex = self.pass_index;
        params.targetNestingLevel = self.target_nesting_level as u16;
        check_cupti!(unsafe { cuptiRangeProfilerSetConfig(&mut params) });
        Ok(())
    }

    pub fn create_counter_data_image(
        &self,
        max_num_ranges: usize,
        metric_names: &[String],
        counter_data_image: &mut Vec<u8>,
    ) -> Result<(), CUptiResult> {
        let c_metric_names: Vec<CString> = metric_names
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let mut c_metric_ptrs: Vec<*const c_char> =
            c_metric_names.iter().map(|s| s.as_ptr()).collect();
        let mut params: CUpti_RangeProfiler_GetCounterDataSize_Params =
            unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_GetCounterDataSize_Params, counterDataSize: usize);
        params.pRangeProfilerObject = self.range_profiler_object;
        params.pMetricNames = c_metric_ptrs.as_mut_ptr();
        params.numMetrics = metric_names.len();
        params.maxNumOfRanges = max_num_ranges;
        params.maxNumRangeTreeNodes = max_num_ranges as u32;
        check_cupti!(unsafe { cuptiRangeProfilerGetCounterDataSize(&mut params) });
        counter_data_image.resize(params.counterDataSize, 0);
        let mut init_params: CUpti_RangeProfiler_CounterDataImage_Initialize_Params =
            unsafe { std::mem::zeroed() };
        init_params.structSize = struct_size_up_to!(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, pCounterData: *mut u8);
        init_params.pRangeProfilerObject = self.range_profiler_object;
        init_params.pCounterData = counter_data_image.as_mut_ptr();
        init_params.counterDataSize = counter_data_image.len();
        check_cupti!(unsafe { cuptiRangeProfilerCounterDataImageInitialize(&mut init_params) });
        Ok(())
    }

    /// Push a named profiling range (for UserRange mode).
    pub fn push_range(&self, name: &CString) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_PushRange_Params = unsafe { std::mem::zeroed() };
        params.structSize =
            struct_size_up_to!(CUpti_RangeProfiler_PushRange_Params, pRangeName: *const c_char);
        params.pRangeProfilerObject = self.range_profiler_object;
        params.pRangeName = name.as_ptr();
        check_cupti!(unsafe { cuptiRangeProfilerPushRange(&mut params) });
        Ok(())
    }

    /// Pop the current profiling range (for UserRange mode).
    pub fn pop_range(&self) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_PopRange_Params = unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_PopRange_Params, pRangeProfilerObject: *mut CUpti_RangeProfiler_Object);
        params.pRangeProfilerObject = self.range_profiler_object;
        check_cupti!(unsafe { cuptiRangeProfilerPopRange(&mut params) });
        Ok(())
    }

    pub fn decode_counter_data(&self) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_DecodeData_Params = unsafe { std::mem::zeroed() };
        params.structSize =
            struct_size_up_to!(CUpti_RangeProfiler_DecodeData_Params, numOfRangeDropped: usize);
        params.pRangeProfilerObject = self.range_profiler_object;
        check_cupti!(unsafe { cuptiRangeProfilerDecodeData(&mut params) });
        Ok(())
    }

    /// Returns the metric names that were validated as available during `set_config`.
    pub fn validated_metric_names(&self) -> &[String] {
        &self.validated_metric_names
    }

    pub fn initialize_counter_data_image(
        &self,
        counter_data_image: &mut Vec<u8>,
    ) -> Result<(), CUptiResult> {
        let mut params: CUpti_RangeProfiler_CounterDataImage_Initialize_Params =
            unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_CounterDataImage_Initialize_Params, pCounterData: *mut u8);
        params.pRangeProfilerObject = self.range_profiler_object;
        params.pCounterData = counter_data_image.as_mut_ptr();
        params.counterDataSize = counter_data_image.len();
        check_cupti!(unsafe { cuptiRangeProfilerCounterDataImageInitialize(&mut params) });
        Ok(())
    }
}

impl Drop for RangeProfiler {
    fn drop(&mut self) {
        unsafe {
            let mut params: CUpti_Profiler_DeInitialize_Params = std::mem::zeroed();
            params.structSize = struct_size_up_to!(CUpti_Profiler_DeInitialize_Params, pPriv: *mut std::ffi::c_void);
            cuptiProfilerDeInitialize(&mut params);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_profiler_initialization() {
        // We can pass a dummy context pointer since we aren't calling CUDA functions
        // that use it in this test (just initialization).
        let dummy_ctx = std::ptr::null_mut();
        let profiler = RangeProfiler::new(dummy_ctx);

        assert!(
            profiler.config_image.is_empty(),
            "Config image should be empty initially"
        );
        assert_eq!(profiler.pass_index, 0, "Pass index should be 0");
        assert_eq!(
            profiler.target_nesting_level, 0,
            "Target nesting level should be 0"
        );
        assert!(
            !profiler.is_all_pass_submitted,
            "All pass submitted should be false"
        );

        // Prevent Drop from running, as it calls FFI functions that are missing
        std::mem::forget(profiler);
    }
}
