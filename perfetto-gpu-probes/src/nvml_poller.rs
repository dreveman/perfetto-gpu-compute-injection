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

//! NVML GPU enumeration and polling.

use crate::nvml;
use crate::perfetto_dlog;
use crate::perfetto_elog;
use crate::poller::{GpuMetadata, InstanceStop, PollableGpu};
use perfetto_sdk::data_source::DataSource;
use std::ffi::CStr;

/// Information about a single NVML GPU.
pub struct GpuInfo {
    pub index: u32,
    pub handle: nvml::nvmlDevice_t,
    pub name: String,
}

impl PollableGpu for GpuInfo {
    fn index(&self) -> u32 {
        self.index
    }
    fn metadata(&self) -> GpuMetadata {
        // UUID
        let mut uuid_buf = [0u8; 96];
        let uuid = unsafe {
            let ret = nvml::nvmlDeviceGetUUID(
                self.handle,
                uuid_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                uuid_buf.len() as u32,
            );
            if ret == nvml::NVML_SUCCESS {
                CStr::from_ptr(uuid_buf.as_ptr() as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            }
        };

        // PCI BDF
        let mut pci_info: nvml::nvmlPciInfo_t = unsafe { std::mem::zeroed() };
        let pci_bdf = unsafe {
            let ret = nvml::nvmlDeviceGetPciInfo_v3(self.handle, &mut pci_info);
            if ret == nvml::NVML_SUCCESS {
                CStr::from_ptr(pci_info.bus_id.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            }
        };

        // Board part number (model)
        let mut part_buf = [0u8; 256];
        let model = unsafe {
            let ret = nvml::nvmlDeviceGetBoardPartNumber(
                self.handle,
                part_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                part_buf.len() as u32,
            );
            if ret == nvml::NVML_SUCCESS {
                CStr::from_ptr(part_buf.as_ptr() as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            }
        };

        // Architecture
        let mut arch_val: nvml::nvmlDeviceArchitecture_t = 0;
        let architecture = unsafe {
            let ret = nvml::nvmlDeviceGetArchitecture(self.handle, &mut arch_val);
            if ret == nvml::NVML_SUCCESS {
                match arch_val {
                    nvml::NVML_DEVICE_ARCH_KEPLER => "Kepler".to_string(),
                    nvml::NVML_DEVICE_ARCH_MAXWELL => "Maxwell".to_string(),
                    nvml::NVML_DEVICE_ARCH_PASCAL => "Pascal".to_string(),
                    nvml::NVML_DEVICE_ARCH_VOLTA => "Volta".to_string(),
                    nvml::NVML_DEVICE_ARCH_TURING => "Turing".to_string(),
                    nvml::NVML_DEVICE_ARCH_AMPERE => "Ampere".to_string(),
                    nvml::NVML_DEVICE_ARCH_ADA => "Ada".to_string(),
                    nvml::NVML_DEVICE_ARCH_HOPPER => "Hopper".to_string(),
                    nvml::NVML_DEVICE_ARCH_BLACKWELL => "Blackwell".to_string(),
                    other => format!("Unknown ({})", other),
                }
            } else {
                String::new()
            }
        };

        // Extra info
        let mut extra_info = Vec::new();

        // Driver version (system-level, no device handle)
        let mut driver_buf = [0u8; 256];
        unsafe {
            let ret = nvml::nvmlSystemGetDriverVersion(
                driver_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                driver_buf.len() as u32,
            );
            if ret == nvml::NVML_SUCCESS {
                let v = CStr::from_ptr(driver_buf.as_ptr() as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned();
                if !v.is_empty() {
                    extra_info.push(("driver_version".to_string(), v));
                }
            }
        }

        // VBIOS version
        let mut vbios_buf = [0u8; 256];
        unsafe {
            let ret = nvml::nvmlDeviceGetVbiosVersion(
                self.handle,
                vbios_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                vbios_buf.len() as u32,
            );
            if ret == nvml::NVML_SUCCESS {
                let v = CStr::from_ptr(vbios_buf.as_ptr() as *const std::os::raw::c_char)
                    .to_string_lossy()
                    .into_owned();
                if !v.is_empty() {
                    extra_info.push(("vbios_version".to_string(), v));
                }
            }
        }

        // Compute capability
        let mut major: std::os::raw::c_int = 0;
        let mut minor: std::os::raw::c_int = 0;
        unsafe {
            let ret = nvml::nvmlDeviceGetCudaComputeCapability(self.handle, &mut major, &mut minor);
            if ret == nvml::NVML_SUCCESS {
                extra_info.push((
                    "compute_capability".to_string(),
                    format!("{}.{}", major, minor),
                ));
            }
        }

        // Total memory
        let mut mem = nvml::nvmlMemory_t {
            total: 0,
            free: 0,
            used: 0,
        };
        unsafe {
            let ret = nvml::nvmlDeviceGetMemoryInfo(self.handle, &mut mem);
            if ret == nvml::NVML_SUCCESS {
                extra_info.push(("total_memory".to_string(), mem.total.to_string()));
            }
        }

        // TDP (power management default limit)
        let mut tdp_mw: u32 = 0;
        unsafe {
            let ret = nvml::nvmlDeviceGetPowerManagementDefaultLimit(self.handle, &mut tdp_mw);
            if ret == nvml::NVML_SUCCESS {
                extra_info.push(("tdp_watts".to_string(), (tdp_mw / 1000).to_string()));
            }
        }

        // Number of GPU cores
        let mut cores: u32 = 0;
        unsafe {
            let ret = nvml::nvmlDeviceGetNumGpuCores(self.handle, &mut cores);
            if ret == nvml::NVML_SUCCESS {
                extra_info.push(("num_cores".to_string(), cores.to_string()));
            }
        }

        GpuMetadata {
            name: self.name.clone(),
            vendor: "NVIDIA".to_string(),
            uuid,
            pci_bdf,
            model,
            architecture,
            extra_info,
        }
    }
    fn read_frequency(&self) -> Option<u32> {
        let mut clock: u32 = 0;
        let ret = unsafe {
            nvml::nvmlDeviceGetClockInfo(self.handle, nvml::NVML_CLOCK_GRAPHICS, &mut clock)
        };
        if ret != nvml::NVML_SUCCESS {
            return None;
        }
        Some(clock)
    }
    fn read_memory_used(&self) -> Option<u64> {
        let mut mem = nvml::nvmlMemory_t {
            total: 0,
            free: 0,
            used: 0,
        };
        let ret = unsafe { nvml::nvmlDeviceGetMemoryInfo(self.handle, &mut mem) };
        if ret != nvml::NVML_SUCCESS {
            return None;
        }
        Some(mem.used)
    }
}

/// Enumerates all GPUs via NVML.
pub fn enumerate_gpus() -> Vec<GpuInfo> {
    let mut count: u32 = 0;
    let ret = unsafe { nvml::nvmlDeviceGetCount_v2(&mut count) };
    if ret != nvml::NVML_SUCCESS {
        perfetto_elog!("nvmlDeviceGetCount_v2 failed (error {})", ret);
        return Vec::new();
    }

    let mut gpus = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut handle: nvml::nvmlDevice_t = std::ptr::null_mut();
        let ret = unsafe { nvml::nvmlDeviceGetHandleByIndex_v2(i, &mut handle) };
        if ret != nvml::NVML_SUCCESS {
            perfetto_elog!(
                "nvmlDeviceGetHandleByIndex_v2({}) failed (error {})",
                i,
                ret
            );
            continue;
        }

        let mut name_buf = [0u8; 256];
        let ret = unsafe {
            nvml::nvmlDeviceGetName(
                handle,
                name_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                name_buf.len() as u32,
            )
        };
        let name = if ret == nvml::NVML_SUCCESS {
            unsafe { CStr::from_ptr(name_buf.as_ptr() as *const std::os::raw::c_char) }
                .to_string_lossy()
                .into_owned()
        } else {
            format!("GPU {}", i)
        };

        gpus.push(GpuInfo {
            index: i,
            handle,
            name,
        });
    }

    perfetto_dlog!("enumerated {} NVML GPU(s)", gpus.len());
    for gpu in &gpus {
        perfetto_dlog!("  GPU {}: {}", gpu.index, gpu.name);
    }

    gpus
}

/// Runs the NVML polling loop for a single instance. Blocks until `stop` is signaled.
pub(crate) fn run_poll_loop(
    data_source: &'static DataSource<'static>,
    inst_id: u32,
    stop: &InstanceStop,
    poll_ms: u64,
) {
    crate::poller::run_poll_loop(enumerate_gpus, "NVML", data_source, inst_id, stop, poll_ms);
}

#[cfg(all(test, feature = "stubs"))]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_gpus() {
        // Initialize NVML (stubs).
        unsafe {
            nvml::nvmlInit_v2();
        }
        let gpus = enumerate_gpus();
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[0].name, "Stub GPU");
        assert_eq!(gpus[1].name, "Stub GPU");
    }

    #[test]
    fn test_gpu_clock_reading() {
        unsafe {
            nvml::nvmlInit_v2();
        }
        let gpus = enumerate_gpus();
        assert!(!gpus.is_empty());
        assert_eq!(gpus[0].read_frequency(), Some(1500));
    }
}
