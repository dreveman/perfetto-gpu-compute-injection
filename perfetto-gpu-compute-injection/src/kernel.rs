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

//! Utilities for simplifying demangled GPU kernel names.

/// Simplify a demangled kernel name by stripping the return type,
/// arguments, and template parameters, leaving just the qualified
/// function name.
///
/// Examples:
///   "void foo::bar<int>(float*, int)" -> "foo::bar"
///   "at::native::vectorized_elementwise_kernel<4, ...>(...)" -> "at::native::vectorized_elementwise_kernel"
///   "simple_kernel" -> "simple_kernel"
pub fn simplify_name(demangled: &str) -> &str {
    let s = demangled.trim();

    // Pass 1: find name_start by locating the last top-level space
    // (not inside angle brackets) before the first top-level `(`.
    let mut depth: u32 = 0;
    let mut last_space: usize = 0;

    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            '(' if depth == 0 => break,
            ' ' if depth == 0 => last_space = i + 1,
            _ => {}
        }
    }

    // Pass 2: from name_start, find the first `<` or `(` — that's
    // where the function name ends.
    let rest = &s[last_space..];
    let name_len = rest.find(['<', '(']).unwrap_or(rest.len());

    &s[last_space..last_space + name_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_name() {
        assert_eq!(simplify_name("simple_kernel"), "simple_kernel");
    }

    #[test]
    fn with_return_type_and_args() {
        assert_eq!(simplify_name("void foo::bar(float*, int)"), "foo::bar");
    }

    #[test]
    fn with_templates_and_args() {
        assert_eq!(
            simplify_name("void foo::bar<int, float>(float*, int)"),
            "foo::bar"
        );
    }

    #[test]
    fn nested_templates() {
        assert_eq!(
            simplify_name("void at::native::vectorized_elementwise_kernel<4, at::detail::Array<char, 2>>(int, at::detail::Array<char, 2>)"),
            "at::native::vectorized_elementwise_kernel"
        );
    }

    #[test]
    fn no_return_type_with_args() {
        assert_eq!(simplify_name("my_kernel(int, float)"), "my_kernel");
    }

    #[test]
    fn no_return_type_with_templates() {
        assert_eq!(simplify_name("my_kernel<int>(float)"), "my_kernel");
    }

    #[test]
    fn template_return_type() {
        assert_eq!(
            simplify_name("std::vector<int> foo::compute(int)"),
            "foo::compute"
        );
    }

    #[test]
    fn empty_string() {
        assert_eq!(simplify_name(""), "");
    }

    #[test]
    fn namespaced_no_args() {
        assert_eq!(simplify_name("foo::bar::baz"), "foo::bar::baz");
    }
}
