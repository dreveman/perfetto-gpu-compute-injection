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
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;

/// Gets the last CUPTI error.
pub fn get_last_error() -> CUptiResult {
    unsafe { cuptiGetLastError() }
}

/// Gets the string description for a CUPTI result code.
pub fn get_result_string(result: CUptiResult) -> String {
    let mut err_str: *const c_char = ptr::null();
    unsafe {
        cuptiGetResultString(result, &mut err_str);
        if err_str.is_null() {
            return format!("Unknown error {:?}", result);
        }
        CStr::from_ptr(err_str).to_string_lossy().into_owned()
    }
}
