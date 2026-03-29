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

// Manually generated bindings for ftrace-related TracePacket fields.

use perfetto_sdk::pb_msg;
use perfetto_sdk::pb_msg_ext;

use super::ftrace::ftrace_event_bundle::*;
use perfetto_sdk::protos::trace::trace_packet::TracePacket;

pb_msg_ext!(TracePacket {
    ftrace_events: FtraceEventBundle, msg, 1,
});

/// Import this to use the ftrace `TracePacket` fields.
pub mod prelude {
    pub use super::TracePacketExt;
}
