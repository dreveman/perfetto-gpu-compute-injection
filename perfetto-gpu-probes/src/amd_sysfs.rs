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

//! AMD GPU discovery and sysfs-based counter reading.
//!
//! Enumerates AMD GPUs via `/sys/class/drm/card*` and reads GPU frequency
//! from `pp_dpm_sclk`. No library dependency required.

use crate::perfetto_dlog;
use std::path::{Path, PathBuf};

/// PCI vendor ID for AMD/ATI GPUs.
const AMD_PCI_VENDOR_ID: &str = "0x1002";

/// Sysfs filename for GPU clock DPM states.
const PP_DPM_SCLK: &str = "pp_dpm_sclk";

/// Information about a single AMD GPU discovered via sysfs.
pub struct AmdGpuInfo {
    /// Sequential index among AMD GPUs found.
    pub index: u32,
    /// Path to the device directory, e.g. `/sys/class/drm/card0/device`.
    pub card_path: PathBuf,
    /// Display name, e.g. "AMD GPU 0".
    pub name: String,
}

/// Enumerates AMD GPUs by scanning `/sys/class/drm/card*`.
///
/// Keeps entries whose `device/vendor` contains `0x1002` (AMD).
/// Results are sorted by card number for stable ordering.
pub fn enumerate_amd_gpus() -> Vec<AmdGpuInfo> {
    enumerate_amd_gpus_in(Path::new("/sys/class/drm"))
}

/// Enumerates AMD GPUs under a given sysfs base path (for testing).
fn enumerate_amd_gpus_in(drm_path: &Path) -> Vec<AmdGpuInfo> {
    let entries = match std::fs::read_dir(drm_path) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    // Collect (card_number, device_path) pairs for AMD GPUs.
    let mut cards: Vec<(u32, PathBuf, String)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Match "card0", "card1", etc. but not "card0-DP-1".
        let card_num = match name_str.strip_prefix("card") {
            Some(rest) if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) => {
                rest.parse::<u32>().ok()
            }
            _ => None,
        };
        let Some(card_num) = card_num else {
            continue;
        };

        let device_path = entry.path().join("device");
        let vendor_path = device_path.join("vendor");
        let vendor = match std::fs::read_to_string(&vendor_path) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !vendor.trim().eq_ignore_ascii_case(AMD_PCI_VENDOR_ID) {
            continue;
        }

        // Only include if pp_dpm_sclk exists (indicates a real GPU, not a display-only device).
        let sclk_path = device_path.join(PP_DPM_SCLK);
        if !sclk_path.exists() {
            continue;
        }

        let name = std::fs::read_to_string(device_path.join("product_name"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("AMD GPU {}", card_num));

        cards.push((card_num, device_path, name));
    }

    // Sort by card number for stable ordering.
    cards.sort_by_key(|(num, _, _)| *num);

    let gpus: Vec<AmdGpuInfo> = cards
        .into_iter()
        .enumerate()
        .map(|(idx, (_card_num, device_path, name))| AmdGpuInfo {
            index: idx as u32,
            card_path: device_path,
            name,
        })
        .collect();

    perfetto_dlog!("enumerated {} AMD GPU(s)", gpus.len());
    for gpu in &gpus {
        perfetto_dlog!(
            "  AMD GPU {}: {} ({})",
            gpu.index,
            gpu.name,
            gpu.card_path.display()
        );
    }

    gpus
}

/// Reads the PCI BDF (Bus:Device.Function) string for an AMD GPU.
///
/// Resolves the symlink at `<card_path>` (which points to e.g.
/// `../../../0000:03:00.0`) and extracts the final path component.
pub fn read_gpu_pci_bdf(gpu: &AmdGpuInfo) -> String {
    // The card_path is already the `device` symlink target directory.
    // Try to read the canonical path and extract the PCI BDF from the last component.
    if let Ok(canonical) = std::fs::canonicalize(&gpu.card_path) {
        if let Some(name) = canonical.file_name() {
            return name.to_string_lossy().into_owned();
        }
    }
    String::new()
}

/// Reads the current GPU frequency in MHz from `pp_dpm_sclk`.
///
/// The file format is one DPM state per line:
/// ```text
/// 0: 500Mhz
/// 1: 1000Mhz *
/// ```
/// The active state is marked with a trailing `*`.
pub fn read_gpu_frequency(gpu: &AmdGpuInfo) -> Option<u32> {
    let sclk_path = gpu.card_path.join(PP_DPM_SCLK);
    read_gpu_frequency_from(&sclk_path)
}

/// Parses the active GPU frequency from a `pp_dpm_sclk` file.
fn read_gpu_frequency_from(path: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(path).ok()?;
    parse_pp_dpm_sclk(&content)
}

/// Parses `pp_dpm_sclk` content and returns the active frequency in MHz.
fn parse_pp_dpm_sclk(content: &str) -> Option<u32> {
    for line in content.lines() {
        let line = line.trim();
        if !line.ends_with('*') {
            continue;
        }
        // Format: "N: <freq>Mhz *" or "N: <freq>Mhz*"
        // Find the MHz value by looking for a number before "Mhz" or "MHz".
        let line_lower = line.to_ascii_lowercase();
        let mhz_pos = line_lower.find("mhz")?;
        let before_mhz = &line[..mhz_pos].trim();
        // The frequency is the last token before "Mhz".
        let freq_str = before_mhz
            .rsplit_once(|c: char| !c.is_ascii_digit())
            .map_or(*before_mhz, |(_, num)| num);
        return freq_str.parse::<u32>().ok();
    }
    None
}

/// Reads the current GPU memory usage in bytes from `mem_info_vram_used`.
pub fn read_gpu_memory_used(gpu: &AmdGpuInfo) -> Option<u64> {
    let path = gpu.card_path.join("mem_info_vram_used");
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Reads the PCI device ID from sysfs.
pub fn read_gpu_device_id(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("device"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the GPU unique ID from sysfs.
pub fn read_gpu_unique_id(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("unique_id"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the total VRAM in bytes from sysfs.
pub fn read_gpu_vram_total(gpu: &AmdGpuInfo) -> Option<u64> {
    std::fs::read_to_string(gpu.card_path.join("mem_info_vram_total"))
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Reads the VBIOS version string from sysfs.
pub fn read_gpu_vbios_version(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("vbios_version"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the PCI revision from sysfs.
pub fn read_gpu_revision(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("revision"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the current PCIe link speed from sysfs.
pub fn read_gpu_link_speed(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("current_link_speed"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the current PCIe link width from sysfs.
pub fn read_gpu_link_width(gpu: &AmdGpuInfo) -> String {
    std::fs::read_to_string(gpu.card_path.join("current_link_width"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Reads the amdgpu kernel module version.
pub fn read_amdgpu_driver_version() -> String {
    std::fs::read_to_string("/sys/module/amdgpu/version")
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Creates a mock sysfs card directory with the given vendor and optional pp_dpm_sclk content.
    fn create_mock_card(base: &Path, card_name: &str, vendor: &str, sclk_content: Option<&str>) {
        let card_dir = base.join(card_name).join("device");
        fs::create_dir_all(&card_dir).unwrap();
        fs::write(card_dir.join("vendor"), vendor).unwrap();
        if let Some(content) = sclk_content {
            fs::write(card_dir.join(PP_DPM_SCLK), content).unwrap();
        }
    }

    #[test]
    fn test_enumerate_finds_amd_gpus() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();

        create_mock_card(base, "card0", "0x1002\n", Some("0: 500Mhz\n1: 1000Mhz *\n"));
        create_mock_card(base, "card1", "0x10de\n", Some("0: 500Mhz *\n")); // NVIDIA
        create_mock_card(base, "card2", "0x1002\n", Some("0: 800Mhz *\n"));

        let gpus = enumerate_amd_gpus_in(base);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[0].name, "AMD GPU 0");
        assert_eq!(gpus[1].name, "AMD GPU 2");
    }

    #[test]
    fn test_enumerate_skips_no_sclk() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();

        // AMD vendor but no pp_dpm_sclk (e.g. APU display-only).
        create_mock_card(base, "card0", "0x1002\n", None);

        let gpus = enumerate_amd_gpus_in(base);
        assert!(gpus.is_empty());
    }

    #[test]
    fn test_enumerate_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let gpus = enumerate_amd_gpus_in(tmp.path());
        assert!(gpus.is_empty());
    }

    #[test]
    fn test_enumerate_nonexistent_dir() {
        let gpus = enumerate_amd_gpus_in(Path::new("/nonexistent/path"));
        assert!(gpus.is_empty());
    }

    #[test]
    fn test_enumerate_skips_connector_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();

        create_mock_card(base, "card0", "0x1002\n", Some("0: 500Mhz *\n"));
        // card0-DP-1 should be skipped (not a card directory).
        let connector_dir = base.join("card0-DP-1").join("device");
        fs::create_dir_all(&connector_dir).unwrap();
        fs::write(connector_dir.join("vendor"), "0x1002\n").unwrap();

        let gpus = enumerate_amd_gpus_in(base);
        assert_eq!(gpus.len(), 1);
    }

    #[test]
    fn test_parse_pp_dpm_sclk_basic() {
        let content = "0: 500Mhz\n1: 1000Mhz *\n";
        assert_eq!(parse_pp_dpm_sclk(content), Some(1000));
    }

    #[test]
    fn test_parse_pp_dpm_sclk_first_active() {
        let content = "0: 500Mhz *\n1: 1000Mhz\n";
        assert_eq!(parse_pp_dpm_sclk(content), Some(500));
    }

    #[test]
    fn test_parse_pp_dpm_sclk_no_space_before_star() {
        // Some drivers emit without space: "1: 1000Mhz*"
        let content = "0: 500Mhz\n1: 1000Mhz*\n";
        assert_eq!(parse_pp_dpm_sclk(content), Some(1000));
    }

    #[test]
    fn test_parse_pp_dpm_sclk_uppercase() {
        let content = "0: 500MHz *\n";
        assert_eq!(parse_pp_dpm_sclk(content), Some(500));
    }

    #[test]
    fn test_parse_pp_dpm_sclk_no_active() {
        let content = "0: 500Mhz\n1: 1000Mhz\n";
        assert_eq!(parse_pp_dpm_sclk(content), None);
    }

    #[test]
    fn test_parse_pp_dpm_sclk_empty() {
        assert_eq!(parse_pp_dpm_sclk(""), None);
    }

    #[test]
    fn test_read_gpu_frequency_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let sclk_path = tmp.path().join("pp_dpm_sclk");
        fs::write(&sclk_path, "0: 500Mhz\n1: 1200Mhz *\n2: 1800Mhz\n").unwrap();

        assert_eq!(read_gpu_frequency_from(&sclk_path), Some(1200));
    }

    #[test]
    fn test_read_gpu_frequency_missing_file() {
        assert_eq!(
            read_gpu_frequency_from(Path::new("/nonexistent/pp_dpm_sclk")),
            None
        );
    }
}
