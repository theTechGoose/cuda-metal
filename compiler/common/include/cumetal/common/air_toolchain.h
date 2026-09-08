#pragma once

#include <string>

namespace cumetal::common {

// The AIR/MSL dialect a Metal toolchain accepts.
//
// CuMetal used to hardcode one dialect -- AIR 2.8, MSL 4.0, air64_v28,
// macosx26.0.0 -- across three files. That is Xcode 26 only. On Xcode 16.4 it
// failed twice, in two different places, for two different reasons:
//
//   air-lld: air version set to 2.8.0 ..., but expecting 2.7
//   newLibraryWithData: This library is using language version 4.0
//                       which is not supported on this OS
//
// The second failure happened at *library load*, long after a clean compile and
// link, so the kernel silently never ran and its output buffer stayed zeroed --
// which reads downstream as wrong numerical results rather than a version
// problem. Hence: never hardcode a dialect, and never let a load failure look
// like arithmetic.
// Last-resort dialect, used only when detection fails AND no override is set.
//
// air-lld matches its OWN version exactly ("air version set to 2.8.0, but
// expecting 2.7"), so no constant is universally safe: whichever we pick is
// wrong on some toolchain. We therefore pick the CONSERVATIVE one -- the oldest
// dialect still in support -- because a machine where the probe fails is far more
// likely to be an older or unusual setup than the current reference toolchain,
// and we announce the guess loudly rather than letting it surface four layers
// later as an unrecognised architecture or a library-load failure.
//
// Set CUMETAL_AIR_DIALECT=<air>/<msl> (e.g. "2.8/4.0") to state it explicitly.
inline constexpr int kFallbackAirMajor = 2;
inline constexpr int kFallbackAirMinor = 7;
inline constexpr int kFallbackLanguageMajor = 3;
inline constexpr int kFallbackLanguageMinor = 2;
inline constexpr const char* kFallbackTargetTriple = "air64_v27-apple-macosx15.0.0";

struct AirDialect {
    int air_major = kFallbackAirMajor;
    int air_minor = kFallbackAirMinor;
    int language_major = kFallbackLanguageMajor;
    int language_minor = kFallbackLanguageMinor;
    // Architecture-qualified triple for air-opt, e.g. "air64_v27-apple-macosx15.0.0".
    std::string target_triple = kFallbackTargetTriple;
    // False when probing was impossible and the fallback above is in use.
    bool probed = false;

    std::string air_version_string() const;       // "2.7"
    std::string language_version_string() const;  // "3.2"

    // AIR 2.8 widened the simdgroup_matrix load/store ABI. The intrinsic has the
    // same name in both dialects, so calling it with the wrong argument types
    // fails as "invalid AIR function" -- which reads like a missing feature but
    // is really an overload mismatch. This is what WMMA lowers to.
    //   <= 2.7:  (ptr, i64 elements_per_row, <2 x i64> origin, i1 transpose)
    //   >= 2.8:  (ptr, <2 x i64> shape, <2 x i64> stride, <2 x i64> origin)
    bool simdgroup_matrix_wide_abi() const {
        return air_major > 2 || (air_major == 2 && air_minor >= 8);
    }
};

// The AIR version stamped on an "experimental container" -- the placeholder the
// emitter produces when no real Metal toolchain was available to lower a kernel.
// It is deliberately NOT the detected dialect: the container is not executable,
// and labelling it with a real version would let it masquerade as one.
inline constexpr const char* kExperimentalContainerAirVersion = "2.6";

// The dialect of the installed toolchain, probed once and cached.
// Thread-safe; falls back to the defaults above when no toolchain is present.
const AirDialect& detected_air_dialect();

}  // namespace cumetal::common
