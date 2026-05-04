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

//! Inline definition of `TracePacketDefaults` (TracePacket field 59) plus an
//! extension trait that exposes `set_trace_packet_defaults` on `TracePacket`.
//!
//! `TracePacketDefaults` lives in the upstream perfetto schema but the
//! perfetto-sdk Rust bindings hand-pick a subset of `TracePacket` fields
//! (see `perfetto_sdk::protos::trace::trace_packet`) and field 59 is not in
//! that subset yet. This module is a temporary shim using the SDK's own
//! `pb_msg!` / `pb_msg_ext!` macros — the same mechanism perfetto-sdk-protos-gpu
//! uses to layer GPU-specific TracePacket fields on top of the SDK.
//!
//! Drop this module (and switch consumers back to the SDK type) once
//! perfetto-sdk lands TracePacketDefaults upstream — planned for v1.0.
// TODO(perfetto-sdk-1.0): remove this shim once
// `perfetto_sdk::protos::trace::trace_packet::TracePacketDefaults` exists
// and `TracePacket` exposes a `set_trace_packet_defaults` setter directly.

use perfetto_sdk::pb_msg;
use perfetto_sdk::pb_msg_ext;
use perfetto_sdk::protos::trace::trace_packet::TracePacket;
use perfetto_sdk::protos::trace::track_event::track_event::TrackEventDefaults;

pb_msg!(TracePacketDefaults {
    timestamp_clock_id: u32, primitive, 58,
    track_event_defaults: TrackEventDefaults, msg, 11,
});

pb_msg_ext!(TracePacket {
    trace_packet_defaults: TracePacketDefaults, msg, 59,
});

/// Bring the `TracePacket::set_trace_packet_defaults(...)` setter into scope
/// alongside the rest of the consumer's `TracePacket` API.
pub mod prelude {
    pub use super::TracePacketExt;
}
