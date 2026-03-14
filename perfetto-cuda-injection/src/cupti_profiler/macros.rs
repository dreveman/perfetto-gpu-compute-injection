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

#[macro_export]
macro_rules! check_cupti {
    ($res:expr) => {
        let res = $res;
        if res != $crate::cupti_profiler::bindings::CUptiResult_CUPTI_SUCCESS {
            ::perfetto_gpu_compute_injection::injection_log!("CUPTI Error: {:?}", res);
            return Err(res);
        }
    };
}

#[macro_export]
macro_rules! struct_size_up_to {
    ($ty:ty, $field:tt : $field_ty:ty) => {
        core::mem::offset_of!($ty, $field) + core::mem::size_of::<$field_ty>()
    };
}

#[cfg(test)]
mod tests {
    use crate::cupti_profiler::bindings::{CUptiResult, CUptiResult_CUPTI_SUCCESS};

    #[test]
    fn test_check_cupti_macro() {
        fn dummy_success() -> Result<(), CUptiResult> {
            check_cupti!(CUptiResult_CUPTI_SUCCESS);
            Ok(())
        }

        fn dummy_failure() -> Result<(), CUptiResult> {
            check_cupti!(
                crate::cupti_profiler::bindings::CUptiResult_CUPTI_ERROR_INVALID_PARAMETER
            );
            Ok(())
        }

        assert!(dummy_success().is_ok());
        assert!(dummy_failure().is_err());
    }
}
