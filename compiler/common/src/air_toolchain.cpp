#include "cumetal/common/air_toolchain.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>

namespace cumetal::common {
namespace {

std::string quote_shell(const std::string& value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

// Read a "!N = !{...}" metadata node that a named node points at.
bool read_named_version(const std::string& ir, const std::string& named, int* major, int* minor) {
    std::smatch match;
    const std::regex ref_re("!" + named + R"( = !\{!(\d+)\})");
    if (!std::regex_search(ir, match, ref_re)) {
        return false;
    }
    const std::string node = match[1].str();
    // air.version:            !N = !{i32 2, i32 7, i32 0}
    // air.language_version:   !N = !{!"Metal", i32 3, i32 2, i32 0}
    const std::regex node_re("!" + node + R"( = !\{(?:!"Metal", )?i32 (\d+), i32 (\d+), i32 \d+\})");
    if (!std::regex_search(ir, match, node_re)) {
        return false;
    }
    *major = std::stoi(match[1].str());
    *minor = std::stoi(match[2].str());
    return true;
}

// Ask the installed compiler what it emits, rather than mapping Xcode releases
// to dialects by hand. Authoritative by construction, and needs no update when
// a new Xcode ships. Any failure leaves the built-in defaults untouched.
AirDialect probe() {
    AirDialect dialect;

    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec) /
                     ("cumetal-air-probe-" + std::to_string(::getpid()));
    if (ec) {
        return dialect;
    }
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return dialect;
    }
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{dir};

    const auto source = dir / "probe.metal";
    {
        std::ofstream out(source);
        if (!out) {
            return dialect;
        }
        out << "#include <metal_stdlib>\n"
               "using namespace metal;\n"
               "kernel void cumetal_air_probe(device float* o [[buffer(0)]],\n"
               "                              uint i [[thread_position_in_grid]]) {\n"
               "    o[i] = 1.0f;\n"
               "}\n";
    }

    const auto ir_path = dir / "probe.ll";
    const std::string command = "xcrun metal -x metal -S -emit-llvm " + quote_shell(source.string()) +
                                " -o " + quote_shell(ir_path.string()) + " 2>/dev/null";
    if (std::system(command.c_str()) != 0) {
        return dialect;
    }

    std::ifstream in(ir_path);
    if (!in) {
        return dialect;
    }
    std::stringstream contents;
    contents << in.rdbuf();
    const std::string ir = contents.str();

    AirDialect probed;
    if (!read_named_version(ir, "air\\.version", &probed.air_major, &probed.air_minor)) {
        return dialect;
    }
    if (!read_named_version(ir, "air\\.language_version", &probed.language_major,
                            &probed.language_minor)) {
        return dialect;
    }

    // metal emits a bare "air64-apple-macosxN"; air-opt wants the AIR version
    // baked into the architecture. Reassemble the form the tools accept.
    std::smatch match;
    static const std::regex triple_re(R"RE(target triple = "air64[^"]*-(apple-[^"]+)")RE");
    if (!std::regex_search(ir, match, triple_re)) {
        return dialect;
    }
    probed.target_triple = "air64_v" + std::to_string(probed.air_major) +
                           std::to_string(probed.air_minor) + "-" + match[1].str();
    probed.probed = true;
    return probed;
}

}  // namespace

std::string AirDialect::air_version_string() const {
    return std::to_string(air_major) + "." + std::to_string(air_minor);
}

std::string AirDialect::language_version_string() const {
    return std::to_string(language_major) + "." + std::to_string(language_minor);
}

// CUMETAL_AIR_DIALECT=<air>/<msl>, e.g. "2.8/4.0". Explicit beats detected.
bool dialect_from_env(AirDialect* out) {
    const char* raw = std::getenv("CUMETAL_AIR_DIALECT");
    if (raw == nullptr) {
        return false;
    }
    std::smatch match;
    const std::string value(raw);
    static const std::regex re(R"RE(^\s*(\d+)\.(\d+)\s*/\s*(\d+)\.(\d+)\s*$)RE");
    if (!std::regex_match(value, match, re)) {
        std::fprintf(stderr,
                     "CuMetal: ignoring malformed CUMETAL_AIR_DIALECT=\"%s\" "
                     "(expected <air_major>.<air_minor>/<msl_major>.<msl_minor>, e.g. 2.8/4.0)\n",
                     raw);
        return false;
    }
    out->air_major = std::stoi(match[1].str());
    out->air_minor = std::stoi(match[2].str());
    out->language_major = std::stoi(match[3].str());
    out->language_minor = std::stoi(match[4].str());
    out->target_triple = "air64_v" + std::to_string(out->air_major) +
                         std::to_string(out->air_minor) + "-apple-macosx" +
                         (out->air_major == 2 && out->air_minor >= 8 ? "26.0.0" : "15.0.0");
    out->probed = true;  // stated by the operator; as authoritative as a probe
    return true;
}

const AirDialect& detected_air_dialect() {
    static const AirDialect cached = [] {
        AirDialect from_env;
        if (dialect_from_env(&from_env)) {
            return from_env;
        }
        const AirDialect result = probe();
        if (!result.probed) {
            // Never guess silently. A wrong dialect does not fail here -- it fails
            // much later as "unrecognized architecture air64_vNN" or, worse, at
            // library load as "language version N.N is not supported on this OS",
            // by which point nothing points back to this decision.
            std::fprintf(stderr,
                         "CuMetal: could not detect the installed Metal toolchain's AIR/MSL "
                         "dialect (is `xcrun metal` available?). Assuming AIR %d.%d / Metal %d.%d. "
                         "If that is wrong, kernels will fail to link or load; set "
                         "CUMETAL_AIR_DIALECT=<air>/<msl> (e.g. 2.8/4.0) to state it.\n",
                         result.air_major, result.air_minor,
                         result.language_major, result.language_minor);
        }
        return result;
    }();
    return cached;
}

}  // namespace cumetal::common
