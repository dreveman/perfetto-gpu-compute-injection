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

// Manually generated bindings matching upstream protos/perfetto/trace/ftrace/gpu_mem.proto.

use perfetto_sdk::pb_msg;

pb_msg!(GpuMemTotalFtraceEvent {
    gpu_id: u32, primitive, 1,
    pid: u32, primitive, 2,
    size: u64, primitive, 3,
});
