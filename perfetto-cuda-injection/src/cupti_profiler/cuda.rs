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
use std::collections::HashMap;
use std::sync::Mutex;

/// Safe wrapper for `cuCtxGetDevice`.
/// # Safety
///
/// The `_ctx` pointer must be valid (even if currently unused).
pub unsafe fn get_device(_ctx: CUcontext) -> Result<CUdevice, u32> {
    let mut device: CUdevice = 0;
    let res = unsafe { cuCtxGetDevice(&mut device) };
    if res != 0 {
        return Err(res);
    }
    Ok(device)
}

/// Safe wrapper for `cuDeviceGetAttribute`.
pub fn get_device_attribute(dev: CUdevice, attr: CUdevice_attribute) -> Result<i32, u32> {
    let mut val = 0;
    let res = unsafe { cuDeviceGetAttribute(&mut val, attr, dev) };
    if res != 0 {
        return Err(res);
    }
    Ok(val)
}

/// Safe wrapper for `cuFuncGetAttribute`.
/// # Safety
///
/// The `func` pointer must be a valid CUDA function handle.
pub unsafe fn get_func_attribute(func: CUfunction, attr: CUfunction_attribute) -> Result<i32, u32> {
    let mut val = 0;
    let res = unsafe { cuFuncGetAttribute(&mut val, attr, func) };
    if res != 0 {
        return Err(res);
    }
    Ok(val)
}

/// Safe wrapper for `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
/// # Safety
///
/// The `func` pointer must be a valid CUDA function handle.
pub unsafe fn occupancy_max_active_blocks_per_multiprocessor(
    func: CUfunction,
    block_size: i32,
    dynamic_smem_size: usize,
) -> Result<i32, u32> {
    let mut num_blocks = 0;
    let res = unsafe {
        cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &mut num_blocks,
            func,
            block_size,
            dynamic_smem_size,
        )
    };
    if res != 0 {
        return Err(res);
    }
    Ok(num_blocks)
}

/// Gets the CUPTI context ID for a CUDA context.
/// # Safety
///
/// The `ctx` pointer must be a valid CUDA context.
pub unsafe fn get_context_id(ctx: CUcontext) -> u32 {
    let mut ctx_id = 0;
    let _ = unsafe { cuptiGetContextId(ctx, &mut ctx_id) };
    ctx_id
}

/// Gets the UUID for a CUDA device as raw bytes.
pub fn get_device_uuid(dev: CUdevice) -> Result<[u8; 16], u32> {
    let mut uuid: CUuuid = CUuuid { bytes: [0; 16] };
    let res = unsafe { cuDeviceGetUuid_v2(&mut uuid, dev) };
    if res != 0 {
        return Err(res);
    }
    // Convert c_char bytes to u8 bytes
    Ok(uuid.bytes.map(|b| b as u8))
}

/// Returns the nvidia-smi index for a CUDA device ordinal.
///
/// Cross-references through NVML: CUDA ordinal → PCI bus ID → NVML handle → nvidia-smi index.
/// Falls back to the CUDA ordinal if NVML is unavailable or any step fails.
/// Results are cached per device ordinal.
pub fn get_nvidia_smi_index(dev: CUdevice) -> u32 {
    static CACHE: Mutex<Option<HashMap<i32, u32>>> = Mutex::new(None);

    let mut cache = CACHE.lock().unwrap();
    let map = cache.get_or_insert_with(HashMap::new);
    if let Some(&idx) = map.get(&dev) {
        return idx;
    }

    let index = get_nvidia_smi_index_uncached(dev);
    map.insert(dev, index);
    index
}

fn get_nvidia_smi_index_uncached(dev: CUdevice) -> u32 {
    injection_log!(
        "gpu_id: looking up nvidia-smi index for CUDA ordinal {}",
        dev
    );

    // Step 1: Get PCI bus ID from CUDA driver
    let mut pci_bus_id = [0i8; 64];
    let res = unsafe { cuDeviceGetPCIBusId(pci_bus_id.as_mut_ptr(), 64, dev) };
    if res != 0 {
        injection_log!(
            "gpu_id: cuDeviceGetPCIBusId failed for device {} (error {}), falling back to CUDA ordinal",
            dev, res
        );
        return dev as u32;
    }
    let pci_bus_id_str = unsafe { std::ffi::CStr::from_ptr(pci_bus_id.as_ptr()) }.to_string_lossy();
    injection_log!(
        "gpu_id: CUDA ordinal {} has PCI bus ID '{}'",
        dev,
        pci_bus_id_str
    );

    // Step 2: Get NVML device handle by PCI bus ID
    let mut nvml_handle: nvmlDevice_t = std::ptr::null_mut();
    let res = unsafe {
        nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.as_ptr() as *const _, &mut nvml_handle)
    };
    if res != 0 {
        injection_log!(
            "gpu_id: nvmlDeviceGetHandleByPciBusId_v2 failed for '{}' (error {}), falling back to CUDA ordinal {}",
            pci_bus_id_str, res, dev
        );
        return dev as u32;
    }
    injection_log!(
        "gpu_id: got NVML handle for PCI bus ID '{}'",
        pci_bus_id_str
    );

    // Step 3: Get nvidia-smi index from NVML handle
    let mut index: u32 = 0;
    let res = unsafe { nvmlDeviceGetIndex(nvml_handle, &mut index) };
    if res != 0 {
        injection_log!(
            "gpu_id: nvmlDeviceGetIndex failed (error {}), falling back to CUDA ordinal {}",
            res,
            dev
        );
        return dev as u32;
    }

    injection_log!(
        "gpu_id: CUDA ordinal {} -> PCI '{}' -> nvidia-smi index {}",
        dev,
        pci_bus_id_str,
        index
    );
    index
}
