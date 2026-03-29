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

#include <string.h>

typedef unsigned int nvmlReturn_t;
#define NVML_SUCCESS 0
typedef void* nvmlDevice_t;
typedef unsigned int nvmlClockType_t;

typedef struct {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_t;

typedef struct {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

#define NVML_SUCCESS 0

// Stub storage: 2 fake GPUs.
static void* stub_devices[2] = {(void*)0x1, (void*)0x2};

extern "C" {

nvmlReturn_t nvmlInit_v2() { return NVML_SUCCESS; }

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
  *deviceCount = 2;
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                           nvmlDevice_t* device) {
  if (index < 2) {
    *device = stub_devices[index];
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type_,
                                    unsigned int* clock) {
  (void)device;
  (void)type_;
  *clock = 1500;
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name,
                               unsigned int length) {
  (void)device;
  const char* stub_name = "Stub GPU";
  strncpy(name, stub_name, length);
  if (length > 0) {
    name[length - 1] = '\0';
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device,
                                     nvmlMemory_t* memory) {
  (void)device;
  memory->total = 8ULL * 1024 * 1024 * 1024;  // 8 GB
  memory->free = 4ULL * 1024 * 1024 * 1024;   // 4 GB
  memory->used = 4ULL * 1024 * 1024 * 1024;   // 4 GB
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid,
                               unsigned int length) {
  (void)device;
  const char* stub_uuid = "GPU-00000000-0000-0000-0000-000000000000";
  strncpy(uuid, stub_uuid, length);
  if (length > 0) {
    uuid[length - 1] = '\0';
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  (void)device;
  memset(pci, 0, sizeof(*pci));
  strncpy(pci->busId, "0000:01:00.0", sizeof(pci->busId) - 1);
  strncpy(pci->busIdLegacy, "0000:01:00.0", sizeof(pci->busIdLegacy) - 1);
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                       unsigned int* arch) {
  (void)device;
  *arch = 7;  // Ampere
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major,
                                                int* minor) {
  (void)device;
  *major = 8;
  *minor = 0;
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber,
                                          unsigned int length) {
  (void)device;
  const char* stub_part = "900-STUB";
  strncpy(partNumber, stub_part, length);
  if (length > 0) {
    partNumber[length - 1] = '\0';
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
  const char* stub_version = "550.0.0";
  strncpy(version, stub_version, length);
  if (length > 0) {
    version[length - 1] = '\0';
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version,
                                       unsigned int length) {
  (void)device;
  const char* stub_version = "96.00.00.00";
  strncpy(version, stub_version, length);
  if (length > 0) {
    version[length - 1] = '\0';
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                                      unsigned int* limit) {
  (void)device;
  *limit = 300000;  // milliwatts
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device,
                                      unsigned int* numCores) {
  (void)device;
  *numCores = 8192;
  return NVML_SUCCESS;
}

}  // extern "C"
