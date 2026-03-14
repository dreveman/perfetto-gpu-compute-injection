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

use std::env;

fn main() {
    let use_stubs = env::var("CARGO_FEATURE_STUBS").is_ok();

    #[cfg(feature = "bindgen")]
    {
        let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        let rocm_include = format!("{}/include", rocm_path);
        println!("cargo:rerun-if-changed=wrapper.h");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        let bindings = bindgen::Builder::default()
            .header("wrapper.h")
            .clang_arg(format!("-I{}", rocm_include))
            .clang_arg("-D__HIP_PLATFORM_AMD__")
            .allowlist_type("rocprofiler.*")
            .allowlist_type("hsa_.*")
            .allowlist_var("rocprofiler.*")
            .allowlist_var("hsa_.*")
            // Only generate types and constants; function declarations are
            // provided by the dispatch module (runtime dlsym) or stubs.
            .blocklist_function(".*")
            .generate_comments(false)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate AMD bindings");
        let out_dir = env::var("OUT_DIR").unwrap();
        let out_path = std::path::PathBuf::from(&out_dir).join("bindings_generated.rs");
        bindings
            .write_to_file(&out_path)
            .expect("Couldn't write AMD bindings!");
    }

    if use_stubs {
        #[cfg(feature = "stubs")]
        {
            cc::Build::new()
                .file("stubs.cpp")
                .cpp(true)
                .compile("rocprofiler_stubs");
        }
    } else {
        // Non-stubs: rocprofiler-sdk functions are resolved via dlsym at
        // runtime (see src/rocprofiler_sys/dispatch.rs), so no link directives
        // are needed here.
    }
}
