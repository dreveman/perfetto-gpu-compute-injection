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

/// Enables a CUPTI activity kind.
pub fn activity_enable(kind: CUpti_ActivityKind) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiActivityEnable(kind) });
    Ok(())
}

/// Registers callbacks for CUPTI activity buffering.
/// # Safety
///
/// The function pointers must be valid.
pub unsafe fn activity_register_callbacks(
    func_request: CUpti_BuffersCallbackRequestFunc,
    func_complete: CUpti_BuffersCallbackCompleteFunc,
) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiActivityRegisterCallbacks(func_request, func_complete) });
    Ok(())
}

/// Registers a custom timestamp callback for CUPTI activity records.
///
/// This allows activity records to use timestamps from a custom source
/// (e.g., trace clock) instead of the default CUPTI timestamps.
/// # Safety
///
/// The function pointer must be valid and remain valid for the lifetime of the activity session.
pub unsafe fn activity_register_timestamp_callback(
    func_timestamp: CUpti_TimestampCallbackFunc,
) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiActivityRegisterTimestampCallback(func_timestamp) });
    Ok(())
}

/// Flushes all CUPTI activity buffers.
pub fn activity_flush_all(flag: u32) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiActivityFlushAll(flag) });
    Ok(())
}

/// Retrieves the next activity record from a buffer.
/// # Safety
///
/// The `buffer` and `record` pointers must be valid.
pub unsafe fn activity_get_next_record(
    buffer: *mut u8,
    valid_size: usize,
    record: &mut *mut CUpti_Activity,
) -> Result<(), CUptiResult> {
    let res = unsafe { cuptiActivityGetNextRecord(buffer, valid_size, record) };
    if res != CUptiResult_CUPTI_SUCCESS {
        return Err(res);
    }
    Ok(())
}
