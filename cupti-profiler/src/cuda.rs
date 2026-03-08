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

use crate::bindings::*;

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
    Ok(unsafe { std::mem::transmute::<[i8; 16], [u8; 16]>(uuid.bytes) })
}
