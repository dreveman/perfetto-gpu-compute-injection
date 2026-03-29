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

//! AMD GPU polling via sysfs.

use crate::amd_sysfs::{self, AmdGpuInfo};
use crate::poller::{GpuMetadata, InstanceStop, PollableGpu};
use perfetto_sdk::data_source::DataSource;

impl PollableGpu for AmdGpuInfo {
    fn index(&self) -> u32 {
        self.index
    }
    fn metadata(&self) -> GpuMetadata {
        let uuid = amd_sysfs::read_gpu_unique_id(self);
        let model = amd_sysfs::read_gpu_device_id(self);

        let mut extra_info = Vec::new();

        let driver = amd_sysfs::read_amdgpu_driver_version();
        if !driver.is_empty() {
            extra_info.push(("driver_version".to_string(), driver));
        }

        let vbios = amd_sysfs::read_gpu_vbios_version(self);
        if !vbios.is_empty() {
            extra_info.push(("vbios_version".to_string(), vbios));
        }

        if let Some(vram) = amd_sysfs::read_gpu_vram_total(self) {
            extra_info.push(("total_memory".to_string(), vram.to_string()));
        }

        let revision = amd_sysfs::read_gpu_revision(self);
        if !revision.is_empty() {
            extra_info.push(("pci_revision".to_string(), revision));
        }

        let link_speed = amd_sysfs::read_gpu_link_speed(self);
        if !link_speed.is_empty() {
            extra_info.push(("pcie_link_speed".to_string(), link_speed));
        }

        let link_width = amd_sysfs::read_gpu_link_width(self);
        if !link_width.is_empty() {
            extra_info.push(("pcie_link_width".to_string(), link_width));
        }

        GpuMetadata {
            name: self.name.clone(),
            vendor: "AMD".to_string(),
            pci_bdf: amd_sysfs::read_gpu_pci_bdf(self),
            uuid,
            model,
            architecture: String::new(),
            extra_info,
        }
    }
    fn read_frequency(&self) -> Option<u32> {
        amd_sysfs::read_gpu_frequency(self)
    }
    fn read_memory_used(&self) -> Option<u64> {
        amd_sysfs::read_gpu_memory_used(self)
    }
}

/// Runs the AMD polling loop for a single instance. Blocks until `stop` is signaled.
pub(crate) fn run_poll_loop(
    data_source: &'static DataSource<'static>,
    inst_id: u32,
    stop: &InstanceStop,
    poll_ms: u64,
) {
    crate::poller::run_poll_loop(
        amd_sysfs::enumerate_amd_gpus,
        "AMD",
        data_source,
        inst_id,
        stop,
        poll_ms,
    );
}
