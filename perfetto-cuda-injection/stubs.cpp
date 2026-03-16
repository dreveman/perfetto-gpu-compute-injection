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

#include <stdint.h>
#include <string.h>

// Dummy types for stubs to avoid CUDA header dependency
typedef int CUresult;
typedef int CUptiResult;
typedef int CUdevice;
typedef int CUcontext;
typedef int CUfunction;
typedef int CUdevice_attribute;
typedef int CUfunction_attribute;
typedef int CUpti_ActivityKind;
typedef int CUpti_CallbackDomain;
typedef int CUpti_CallbackId;

typedef struct {
  char bytes[16];
} CUuuid;

#define CUDA_SUCCESS 0
#define CUPTI_SUCCESS 0

typedef struct {
  void* pCounterAvailabilityImage;
  size_t counterAvailabilityImageSize;
} CUpti_Profiler_GetCounterAvailability_Params;

typedef struct {
  size_t configImageSize;
} CUpti_Profiler_Host_GetConfigImageSize_Params;

typedef struct {
  size_t counterDataSize;
} CUpti_RangeProfiler_GetCounterDataSize_Params;

// Opaque pointers for other structs
typedef void CUpti_Profiler_Initialize_Params;
typedef void CUpti_Profiler_DeInitialize_Params;
typedef void CUpti_Profiler_Host_Initialize_Params;
typedef void CUpti_Profiler_Host_Deinitialize_Params;
typedef void CUpti_Profiler_Host_ConfigAddMetrics_Params;
typedef void CUpti_Profiler_Host_GetConfigImage_Params;
typedef void CUpti_Profiler_Host_EvaluateToGpuValues_Params;
typedef void CUpti_Profiler_Host_GetMetricProperties_Params;
typedef void CUpti_Device_GetChipName_Params;
typedef void CUpti_RangeProfiler_Enable_Params;
typedef void CUpti_RangeProfiler_Disable_Params;
typedef void CUpti_RangeProfiler_Start_Params;
typedef void CUpti_RangeProfiler_Stop_Params;
typedef void CUpti_RangeProfiler_SetConfig_Params;
typedef void CUpti_RangeProfiler_CounterDataImage_Initialize_Params;
typedef void CUpti_RangeProfiler_DecodeData_Params;
typedef void CUpti_RangeProfiler_PushRange_Params;
typedef void CUpti_RangeProfiler_PopRange_Params;
typedef void CUpti_RangeProfiler_GetCounterDataInfo_Params;
typedef void CUpti_RangeProfiler_CounterData_GetRangeInfo_Params;
typedef void* CUpti_SubscriberHandle;
typedef void (*CUpti_CallbackFunc)(void* userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void* cbdata);
typedef void (*CUpti_BuffersCallbackRequestFunc)(uint8_t** buffer, size_t* size,
                                                 size_t* maxNumRecords);
typedef void (*CUpti_BuffersCallbackCompleteFunc)(uint8_t* buffer, size_t size,
                                                  size_t validSize);
typedef uint64_t (*CUpti_TimestampCallbackFunc)(void);
typedef void CUpti_Activity;

// Define stubs for CUDA/CUPTI functions used in Rust

extern "C" {

CUresult cuCtxGetDevice(CUdevice* device) {
  (void)device;
  return CUDA_SUCCESS;
}
CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                              CUdevice dev) {
  (void)attrib;
  (void)dev;
  *pi = 0;
  return CUDA_SUCCESS;
}
CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib,
                            CUfunction hfunc) {
  (void)attrib;
  (void)hfunc;
  *pi = 0;
  return CUDA_SUCCESS;
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize) {
  (void)func;
  (void)blockSize;
  (void)dynamicSMemSize;
  *numBlocks = 1;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) {
  (void)dev;
  memset(uuid->bytes, 0, 16);
  return CUDA_SUCCESS;
}

CUptiResult cuptiProfilerInitialize(CUpti_Profiler_Initialize_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerDeInitialize(
    CUpti_Profiler_DeInitialize_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiProfilerHostInitialize(
    CUpti_Profiler_Host_Initialize_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostDeinitialize(
    CUpti_Profiler_Host_Deinitialize_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostConfigAddMetrics(
    CUpti_Profiler_Host_ConfigAddMetrics_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostGetConfigImageSize(
    CUpti_Profiler_Host_GetConfigImageSize_Params* pParams) {
  pParams->configImageSize = 100;  // Mock size
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostGetConfigImage(
    CUpti_Profiler_Host_GetConfigImage_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostGetMetricProperties(
    CUpti_Profiler_Host_GetMetricProperties_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiProfilerHostEvaluateToGpuValues(
    CUpti_Profiler_Host_EvaluateToGpuValues_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiDeviceGetChipName(CUpti_Device_GetChipName_Params* pParams) {
  // Mock chip name
  (void)pParams;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiProfilerGetCounterAvailability(
    CUpti_Profiler_GetCounterAvailability_Params* pParams) {
  if (pParams->pCounterAvailabilityImage == 0) {
    pParams->counterAvailabilityImageSize = 100;
  }
  return CUPTI_SUCCESS;
}

CUptiResult cuptiRangeProfilerEnable(
    CUpti_RangeProfiler_Enable_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerDisable(
    CUpti_RangeProfiler_Disable_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerStart(CUpti_RangeProfiler_Start_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerStop(CUpti_RangeProfiler_Stop_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerSetConfig(
    CUpti_RangeProfiler_SetConfig_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerGetCounterDataSize(
    CUpti_RangeProfiler_GetCounterDataSize_Params* pParams) {
  pParams->counterDataSize = 100;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerCounterDataImageInitialize(
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerDecodeData(
    CUpti_RangeProfiler_DecodeData_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerPushRange(
    CUpti_RangeProfiler_PushRange_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerPopRange(
    CUpti_RangeProfiler_PopRange_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerGetCounterDataInfo(
    CUpti_RangeProfiler_GetCounterDataInfo_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiRangeProfilerCounterDataGetRangeInfo(
    CUpti_RangeProfiler_CounterData_GetRangeInfo_Params* pParams) {
  (void)pParams;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetContextId(CUcontext context, uint32_t* contextId) {
  (void)context;
  *contextId = 1;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiGetLastError() { return CUPTI_SUCCESS; }
CUptiResult cuptiGetVersion(uint32_t* version) {
  *version = 0;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiGetResultString(CUptiResult result, const char** str) {
  (void)result;
  *str = "Success";
  return CUPTI_SUCCESS;
}

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                           CUpti_CallbackFunc callback, void* userdata) {
  (void)subscriber;
  (void)callback;
  (void)userdata;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
  (void)enable;
  (void)subscriber;
  (void)domain;
  (void)cbid;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiEnableDomain(uint32_t enable,
                              CUpti_SubscriberHandle subscriber,
                              CUpti_CallbackDomain domain) {
  (void)enable;
  (void)subscriber;
  (void)domain;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  (void)kind;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  (void)kind;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  (void)funcBufferRequested;
  (void)funcBufferCompleted;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityRegisterTimestampCallback(
    CUpti_TimestampCallbackFunc funcTimestamp) {
  (void)funcTimestamp;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityFlushAll(uint32_t flag) {
  (void)flag;
  return CUPTI_SUCCESS;
}
CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity** record) {
  (void)buffer;
  (void)validBufferSizeBytes;
  (void)record;
  return CUPTI_SUCCESS;
}
}
