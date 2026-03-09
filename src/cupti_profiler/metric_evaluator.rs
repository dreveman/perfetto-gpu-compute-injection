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
    get_chip_name, get_counter_availability_image, Profiler, ProfilerHost,
};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Represents a single metric value.
pub struct MetricValuePair {
    pub metric_name: String,
    pub value: f64,
}

/// Contains profiling results for a specific range.
pub struct RangeInfo {
    pub range_name: String,
    pub metric_and_values: Vec<MetricValuePair>,
}

/// High-level evaluator to extract metrics from counter data.
pub struct MetricEvaluator {
    pub host: ProfilerHost,
}

unsafe impl Send for MetricEvaluator {}
unsafe impl Sync for MetricEvaluator {}

impl MetricEvaluator {
    /// # Safety
    ///
    /// The `ctx` pointer must be a valid CUDA context.
    pub unsafe fn new(ctx: CUcontext) -> Result<Self, CUptiResult> {
        let mut host = ProfilerHost::new();
        Profiler::initialize()?;
        let mut device: CUdevice = 0;
        unsafe {
            let res = cuCtxGetDevice(&mut device);
            if res != 0 {
                return Err(CUptiResult_CUPTI_ERROR_UNKNOWN);
            }
        }
        let chip_name = get_chip_name(device as usize)?;
        let counter_avail = get_counter_availability_image(ctx)?;
        host.setup(
            &chip_name,
            counter_avail,
            CUpti_ProfilerType_CUPTI_PROFILER_TYPE_RANGE_PROFILER,
        )?;
        Ok(Self { host })
    }

    pub fn get_num_of_ranges(&self, counter_data_image: &[u8]) -> Result<usize, CUptiResult> {
        let mut params: CUpti_RangeProfiler_GetCounterDataInfo_Params =
            unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_GetCounterDataInfo_Params, numTotalRanges: usize);
        params.pCounterDataImage = counter_data_image.as_ptr();
        params.counterDataImageSize = counter_data_image.len();
        check_cupti!(unsafe { cuptiRangeProfilerGetCounterDataInfo(&mut params) });
        Ok(params.numTotalRanges)
    }

    pub fn get_range_name(
        &self,
        range_index: usize,
        counter_data_image: &[u8],
    ) -> Result<String, CUptiResult> {
        let mut params: CUpti_RangeProfiler_CounterData_GetRangeInfo_Params =
            unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_RangeProfiler_CounterData_GetRangeInfo_Params, rangeName: *const c_char);
        params.pCounterDataImage = counter_data_image.as_ptr();
        params.counterDataImageSize = counter_data_image.len();
        params.rangeIndex = range_index;
        let delim = CString::new("/").unwrap();
        params.rangeDelimiter = delim.as_ptr();
        check_cupti!(unsafe { cuptiRangeProfilerCounterDataGetRangeInfo(&mut params) });
        let c_str = unsafe { CStr::from_ptr(params.rangeName) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    pub fn evaluate_metrics_for_range(
        &self,
        counter_data_image: &[u8],
        metric_names: &[String],
        range_index: usize,
    ) -> Result<Vec<f64>, CUptiResult> {
        let c_metric_names: Vec<CString> = metric_names
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let mut c_metric_ptrs: Vec<*const c_char> =
            c_metric_names.iter().map(|s| s.as_ptr()).collect();
        let mut metric_values = vec![0.0f64; metric_names.len()];
        let mut params: CUpti_Profiler_Host_EvaluateToGpuValues_Params =
            unsafe { std::mem::zeroed() };
        params.structSize = struct_size_up_to!(CUpti_Profiler_Host_EvaluateToGpuValues_Params, pMetricValues: *mut f64);
        params.pHostObject = self.host.host_object;
        params.pCounterDataImage = counter_data_image.as_ptr();
        params.counterDataImageSize = counter_data_image.len();
        params.ppMetricNames = c_metric_ptrs.as_mut_ptr();
        params.numMetrics = metric_names.len();
        params.rangeIndex = range_index;
        params.pMetricValues = metric_values.as_mut_ptr();
        check_cupti!(unsafe { cuptiProfilerHostEvaluateToGpuValues(&mut params) });
        Ok(metric_values)
    }

    pub fn evaluate_all_ranges(
        &self,
        counter_data_image: &[u8],
        metric_names: &[String],
    ) -> Result<Vec<RangeInfo>, CUptiResult> {
        let num_ranges = self.get_num_of_ranges(counter_data_image)?;
        let mut range_infos = Vec::new();
        for i in 0..num_ranges {
            let range_name = self.get_range_name(i, counter_data_image)?;
            let values = self.evaluate_metrics_for_range(counter_data_image, metric_names, i)?;
            let mut metric_pairs = Vec::new();
            for (j, val) in values.iter().enumerate() {
                metric_pairs.push(MetricValuePair {
                    metric_name: metric_names[j].clone(),
                    value: *val,
                });
            }
            range_infos.push(RangeInfo {
                range_name,
                metric_and_values: metric_pairs,
            });
        }
        Ok(range_infos)
    }
}
