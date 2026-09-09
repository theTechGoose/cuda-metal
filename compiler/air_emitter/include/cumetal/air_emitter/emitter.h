#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cumetal::air_emitter {

enum class EmitMode {
    kXcrun,
    kExperimentalContainer,
};

struct EmitOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    EmitMode mode = EmitMode::kXcrun;
    bool overwrite = false;
    bool fallback_to_experimental = false;
    bool validate_output = true;
    bool run_xcrun_validate = false;
    std::string kernel_name = "vector_add";
    // Additional AIR modules or Metal sources to compile and statically link
    // into the result. Used by software ISA backends such as VF64.
    std::vector<std::filesystem::path> additional_link_inputs;
    // Metal sources included textually ahead of a .metal input. This is used
    // for private inline support code that must be part of the kernel's own
    // translation unit rather than an MTL visible-function library.
    std::vector<std::filesystem::path> textual_include_inputs;
    // CUDA's default floating-point contract: IEEE comparisons and NaN/Inf
    // preserved ("safe"), with FMA contraction allowed. Apple's compiler
    // defaults to -ffast-math, under which `NaN == NaN` is true and
    // `x != x` is false, so generated MSL is compiled with -fno-fast-math
    // unless the caller asked for --use_fast_math ("fast").
    std::string math_mode = "safe";
    bool fp_contract = true;
};

struct EmitResult {
    bool ok = false;
    EmitMode mode_used = EmitMode::kXcrun;
    std::filesystem::path output;
    std::vector<std::string> logs;
    std::string error;
};

EmitResult emit_metallib(const EmitOptions& options);

// The directory holding CuMetal's own Metal support sources
// (cumetal_fp64_support.metal and cumetal_fp64_inline_support.metal), or an
// empty path when none can be found.
//
// These used to be addressed as CUMETAL_SOURCE_DIR/compiler/metal/support --
// a compile-time path to the machine that BUILT the binary. A release then
// asked xcrun to compile a file inside a source checkout that exists nowhere
// but that machine, so the FP64 support link failed for every user who had
// not built CuMetal themselves from that exact directory. It went unnoticed
// because on a build machine the path is real. Resolution is at runtime now,
// from the binary's own location, with the source tree kept only as the
// last resort that makes an uninstalled build work.
//
// CUMETAL_METAL_SUPPORT_DIR overrides it for a relocated or unusual install.
std::filesystem::path metal_support_dir();

// The support source for a given path: the inline variant is included
// textually into a .metal translation unit, the other is air-linked. Empty
// when the support directory cannot be found or the file is not in it.
std::filesystem::path metal_support_file(const std::string& file_name);

}  // namespace cumetal::air_emitter
