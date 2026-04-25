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

//! Utilities for GPU kernel name demangling and simplification.

/// Demangle a C++ mangled symbol name.
///
/// Returns the demangled name if successful, otherwise returns the input unchanged.
pub fn demangle_name(mangled: &str) -> String {
    if let Ok(sym) = cpp_demangle::Symbol::new(mangled) {
        sym.demangle()
            .map(|d| d.to_string())
            .unwrap_or_else(|_| mangled.to_string())
    } else {
        mangled.to_string()
    }
}

/// Simplify a demangled kernel name by stripping the return type,
/// arguments, and template parameters, leaving just the qualified
/// function name. Leading namespaces are progressively dropped until
/// the namespace prefix (including the trailing `::`) is 8 characters
/// or fewer.
///
/// Handles edge cases:
///   - `(anonymous namespace)::` — parens at namespace depth
///   - `operator<<`, `operator()`, `operator>` — operator delimiters
///   - `{lambda(...)#1}` — braces with nested parens
///   - Function pointer template args `void (*)(int)` — parens inside `<>`
///
/// Examples:
///   "void foo::bar<int>(float*, int)" -> "foo::bar"
///   "at::native::vectorized_elementwise_kernel<4, ...>(...)" -> "vectorized_elementwise_kernel"
///   "simple_kernel" -> "simple_kernel"
///   "(anonymous namespace)::my_kernel(int)" -> "my_kernel"
pub fn simplify_name(demangled: &str) -> &str {
    let s = demangled.trim();
    let bytes = s.as_bytes();

    // Pass 1: find name_start by locating the last top-level space
    // before the first top-level argument `(`. Track angle-bracket,
    // paren, and brace depth to handle nested delimiters.
    let mut angle_depth: u32 = 0;
    let mut paren_depth: u32 = 0;
    let mut brace_depth: u32 = 0;
    let mut last_space: usize = 0;
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        // Skip "(anonymous namespace)" — it's a namespace component,
        // not an argument list.
        if c == b'(' && angle_depth == 0 && brace_depth == 0 && paren_depth == 0 {
            if s[i..].starts_with("(anonymous namespace)") {
                i += "(anonymous namespace)".len();
                continue;
            }
            break;
        }
        // Skip operator symbols that contain delimiters.
        if s[i..].starts_with("operator") {
            let op_start = i + "operator".len();
            if let Some(&op_c) = bytes.get(op_start) {
                let skip_len = match op_c {
                    b'(' if bytes.get(op_start + 1) == Some(&b')') => 2,
                    b'<' if bytes.get(op_start + 1) == Some(&b'<') => 2,
                    b'<' => 1,
                    b'>' if bytes.get(op_start + 1) == Some(&b'>') => 2,
                    b'>' => 1,
                    b'!' if bytes.get(op_start + 1) == Some(&b'=') => 2,
                    b'=' if bytes.get(op_start + 1) == Some(&b'=') => 2,
                    b'=' => 1,
                    _ => 0,
                };
                if skip_len > 0 {
                    i = op_start + skip_len;
                    continue;
                }
            }
        }
        match c {
            b'<' => angle_depth += 1,
            b'>' if angle_depth > 0 => angle_depth -= 1,
            b'{' => brace_depth += 1,
            b'}' if brace_depth > 0 => brace_depth -= 1,
            b'(' if angle_depth > 0 || brace_depth > 0 => paren_depth += 1,
            b')' if paren_depth > 0 => paren_depth -= 1,
            b' ' if angle_depth == 0 && paren_depth == 0 && brace_depth == 0 => {
                last_space = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    // Pass 2: from name_start, find the first top-level `<` or `(`
    // (not inside operator names or anonymous namespaces).
    let rest = &s[last_space..];
    let rest_bytes = rest.as_bytes();
    let mut name_end = rest.len();
    let mut j = 0;
    while j < rest_bytes.len() {
        if rest[j..].starts_with("(anonymous namespace)") {
            j += "(anonymous namespace)".len();
            continue;
        }
        if rest[j..].starts_with("operator") {
            let op_start = j + "operator".len();
            if let Some(&op_c) = rest_bytes.get(op_start) {
                let skip_len = match op_c {
                    b'(' if rest_bytes.get(op_start + 1) == Some(&b')') => 2,
                    b'<' if rest_bytes.get(op_start + 1) == Some(&b'<') => 2,
                    b'<' => 1,
                    b'>' if rest_bytes.get(op_start + 1) == Some(&b'>') => 2,
                    b'>' => 1,
                    b'!' if rest_bytes.get(op_start + 1) == Some(&b'=') => 2,
                    b'=' if rest_bytes.get(op_start + 1) == Some(&b'=') => 2,
                    b'=' => 1,
                    _ => 0,
                };
                if skip_len > 0 {
                    name_end = op_start + skip_len;
                    break;
                }
            }
        }
        match rest_bytes[j] {
            b'<' | b'(' => {
                name_end = j;
                break;
            }
            _ => j += 1,
        }
    }
    let qualified = &s[last_space..last_space + name_end];

    // Pass 3: drop leading namespace components until the namespace
    // prefix (including the trailing `::`) is 8 characters or fewer.
    let mut result = qualified;
    while let Some(pos) = result.find("::") {
        if let Some(last_sep) = result.rfind("::") {
            if last_sep + 2 <= 8 {
                break;
            }
        }
        if pos + 2 <= result.len() {
            result = &result[pos + 2..];
        } else {
            break;
        }
    }

    result
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
        // prefix "at::native::" is 12 > 8, drop "at::" -> "native::" is 8 <= 8, keep
        assert_eq!(
            simplify_name("void at::native::vectorized_elementwise_kernel<4, at::detail::Array<char, 2>>(int, at::detail::Array<char, 2>)"),
            "native::vectorized_elementwise_kernel"
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
        // prefix through final "::" is "foo::bar::" (10 > 8), drop "foo::" -> "bar::baz"
        assert_eq!(simplify_name("foo::bar::baz"), "bar::baz");
    }

    #[test]
    fn long_namespace_dropped() {
        // prefix "very_long_ns::" (14 > 8), drop -> "kernel"
        assert_eq!(simplify_name("very_long_ns::kernel"), "kernel");
    }

    #[test]
    fn multiple_namespaces_partially_dropped() {
        // prefix "a::b::c::" (9 > 8), drop "a::" -> "b::c::" (6 <= 8), keep
        assert_eq!(simplify_name("a::b::c::kernel"), "b::c::kernel");
    }

    #[test]
    fn drop_until_short_enough() {
        // prefix "long_ns::mid::" (14 > 8), drop "long_ns::" -> "mid::" (5 <= 8), keep
        assert_eq!(simplify_name("long_ns::mid::kernel"), "mid::kernel");
    }

    #[test]
    fn short_namespace_kept() {
        // prefix "foo::" (5 <= 8), kept
        assert_eq!(simplify_name("foo::bar"), "foo::bar");
    }

    #[test]
    fn anonymous_namespace() {
        assert_eq!(
            simplify_name("(anonymous namespace)::my_kernel(int)"),
            "my_kernel"
        );
    }

    #[test]
    fn anonymous_namespace_nested() {
        assert_eq!(
            simplify_name("void at::native::(anonymous namespace)::batch_norm_kernel(float*)"),
            "batch_norm_kernel"
        );
    }

    #[test]
    fn operator_shift() {
        assert_eq!(
            simplify_name("void foo::operator<<(std::ostream&, int)"),
            "foo::operator<<"
        );
    }

    #[test]
    fn operator_call() {
        assert_eq!(
            simplify_name("void foo::operator()(int, float)"),
            "foo::operator()"
        );
    }

    #[test]
    fn operator_greater() {
        assert_eq!(
            simplify_name("bool foo::operator>(const foo&)"),
            "foo::operator>"
        );
    }

    #[test]
    fn func_ptr_template_arg() {
        assert_eq!(
            simplify_name("void kernel<void (*)(int)>(float*)"),
            "kernel"
        );
    }

    #[test]
    fn demangle_mangled_name() {
        let demangled = demangle_name("_Z3foov");
        assert_eq!(demangled, "foo()");
    }

    #[test]
    fn demangle_plain_name() {
        assert_eq!(demangle_name("simple_kernel"), "simple_kernel");
    }
}
