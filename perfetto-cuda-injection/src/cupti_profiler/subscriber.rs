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
use std::{ffi::CStr, ptr};

/// Subscribes to CUPTI callbacks.
/// # Safety
///
/// The callback function and userdata pointer must be valid.
pub unsafe fn subscribe(
    callback: CUpti_CallbackFunc,
    userdata: *mut std::ffi::c_void,
) -> Result<CUpti_SubscriberHandle, CUptiResult> {
    let mut subscriber: CUpti_SubscriberHandle = ptr::null_mut();
    check_cupti!(unsafe { cuptiSubscribe(&mut subscriber, callback, userdata) });
    Ok(subscriber)
}

/// # Safety
///
/// The subscriber handle must be valid.
pub unsafe fn enable_callback(
    enable: u32,
    subscriber: CUpti_SubscriberHandle,
    domain: CUpti_CallbackDomain,
    cbid: CUpti_CallbackId,
) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiEnableCallback(enable, subscriber, domain, cbid) });
    Ok(())
}

/// Enable a specific callback domain.
/// # Safety
///
/// The subscriber handle must be valid.
pub unsafe fn enable_domain(
    enable: u32,
    subscriber: CUpti_SubscriberHandle,
    domain: CUpti_CallbackDomain,
) -> Result<(), CUptiResult> {
    check_cupti!(unsafe { cuptiEnableDomain(enable, subscriber, domain) });
    Ok(())
}

/// Returns the human-readable name for a CUPTI callback ID in the given domain.
///
/// Falls back to `api_call_{cbid}` if the lookup fails.
pub fn get_callback_name(domain: CUpti_CallbackDomain, cbid: u32) -> String {
    let mut name_ptr: *const std::os::raw::c_char = ptr::null();
    let result = unsafe { cuptiGetCallbackName(domain, cbid, &mut name_ptr) };
    if result == CUptiResult_CUPTI_SUCCESS && !name_ptr.is_null() {
        unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned()
    } else {
        format!("api_call_{}", cbid)
    }
}
