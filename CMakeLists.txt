cmake_minimum_required(VERSION 3.28)

project(cumetal VERSION 0.6.6 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(CUMETAL_BUILD_TESTS "Build test targets" ON)

# Clang launches ptxas/fatbinary directly while compiling CUDA source. Native
# shims avoid macOS provenance policy killing interpreter scripts when they are
# spawned as compiler subprocesses.
add_executable(cumetal_ptxas_shim
    tools/cuda_toolchain_shims/ptxas.cpp
)
set_target_properties(cumetal_ptxas_shim PROPERTIES
    OUTPUT_NAME ptxas
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_toolchain
)

add_executable(cumetal_fatbinary_shim
    tools/cuda_toolchain_shims/fatbinary.cpp
)
set_target_properties(cumetal_fatbinary_shim PROPERTIES
    OUTPUT_NAME fatbinary
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda_toolchain
)

# Two independent switches that used to be one. Keep them separate:
#
#   CUMETAL_ENABLE_CUDA_REGISTRATION -- the host-side CUDA registration ABI
#       (__cudaRegisterFatBinary / __cudaRegisterFunction / __cudaPopCallConfiguration).
#       Clang emits calls to these when it compiles *the user's own* .cu source, so the
#       source-recompilation path in docs/legal-notice.md depends on them. This is not a
#       binary shim and carries none of its legal exposure, so it is always on by default.
#
#   CUMETAL_ENABLE_BINARY_SHIM -- the libcuda.dylib alias only. That alias is what lets a
#       pre-built binary linked against NVIDIA's libcuda load CuMetal instead, which is the
#       EULA-sensitive drop-in case. It stays off by default in Release builds.
#
# Conflating them meant a Release build silently stubbed out the registration ABI and the
# entire source path went untested in the configuration users actually ship.
option(CUMETAL_ENABLE_CUDA_REGISTRATION
    "Build the host CUDA registration ABI used by clang-compiled .cu source (__cudaRegister*)"
    ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT OFF)
else()
    set(CUMETAL_ENABLE_BINARY_SHIM_DEFAULT ON)
endif()
option(CUMETAL_ENABLE_BINARY_SHIM
    "Enable CUDA binary-shim library alias (build/install libcuda.dylib -> libcumetal.dylib)"
    ${CUMETAL_ENABLE_BINARY_SHIM_DEFAULT}
)

if(CUMETAL_ENABLE_BINARY_SHIM AND NOT CUMETAL_ENABLE_CUDA_REGISTRATION)
    message(FATAL_ERROR
        "CUMETAL_ENABLE_BINARY_SHIM=ON requires CUMETAL_ENABLE_CUDA_REGISTRATION=ON: "
        "the libcuda.dylib alias is useless without the registration ABI behind it.")
endif()

add_library(cumetal_common STATIC
    compiler/common/src/air_toolchain.cpp
    compiler/common/src/metallib.cpp
)

target_include_directories(cumetal_common
    PUBLIC
        compiler/common/include
)

target_compile_options(cumetal_common
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_ptx STATIC
    compiler/ptx/src/parser.cpp
)

target_include_directories(cumetal_ptx
    PUBLIC
        compiler/ptx/include
)

# The AIR/MSL dialect defaults come from the installed Metal toolchain; that
# probe lives in cumetal_common.
target_link_libraries(cumetal_ptx
    PUBLIC
        cumetal_common
)

target_compile_options(cumetal_ptx
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_printf_pass STATIC
    compiler/passes/src/printf_lower.cpp
)

target_include_directories(cumetal_printf_pass
    PUBLIC
        compiler/passes/include
)

target_link_libraries(cumetal_printf_pass
    PUBLIC
        cumetal_ptx
)

target_compile_options(cumetal_printf_pass
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_ir STATIC
    compiler/ir/src/ir.cpp
    compiler/ir/src/nvvm_importer.cpp
    compiler/ir/src/ptx_importer.cpp
)

target_include_directories(cumetal_ir
    PUBLIC
        compiler/ir/include
)

target_link_libraries(cumetal_ir
    PUBLIC
        cumetal_ptx
        cumetal_printf_pass
)

find_package(LLVM CONFIG QUIET
    HINTS
        /opt/homebrew/opt/llvm/lib/cmake/llvm
        /usr/local/opt/llvm/lib/cmake/llvm
)

if(LLVM_FOUND AND LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 18)
    message(STATUS "CuMetal NVVM importer enabled with LLVM ${LLVM_PACKAGE_VERSION}")
    llvm_map_components_to_libnames(CUMETAL_LLVM_LIBS
        Analysis
        Core
        IRReader
        Passes
        ScalarOpts
        Support
        TransformUtils
    )
    target_include_directories(cumetal_ir SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(cumetal_ir PUBLIC ${CUMETAL_LLVM_LIBS})
    target_compile_definitions(cumetal_ir PUBLIC CUMETAL_HAVE_LLVM=1)
else()
    message(WARNING "LLVM 18+ not found; genuine CUDA/NVVM import will be unavailable")
    target_compile_definitions(cumetal_ir PUBLIC CUMETAL_HAVE_LLVM=0)
endif()

target_compile_options(cumetal_ir
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_metal_codegen STATIC
    compiler/metal/src/msl_ast.cpp
    compiler/metal/src/lower_to_msl.cpp
)

target_include_directories(cumetal_metal_codegen
    PUBLIC
        compiler/metal/include
)

target_link_libraries(cumetal_metal_codegen
    PUBLIC
        cumetal_ir
)

target_compile_options(cumetal_metal_codegen
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_passes STATIC
    compiler/passes/src/addrspace.cpp
    compiler/passes/src/intrinsic_lower.cpp
    compiler/passes/src/metadata.cpp
    compiler/passes/src/phase1_pipeline.cpp
    compiler/passes/src/threadgroup_tiling.cpp
)

target_include_directories(cumetal_passes
    PUBLIC
        compiler/passes/include
)

target_link_libraries(cumetal_passes
    PUBLIC
        cumetal_ptx
        cumetal_printf_pass
)

target_compile_options(cumetal_passes
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_ptx_codegen STATIC
    compiler/ptx/src/lower_to_llvm.cpp
    compiler/ptx/src/lower_to_metal.cpp
)

target_include_directories(cumetal_ptx_codegen
    PUBLIC
        compiler/ptx/include
)

target_link_libraries(cumetal_ptx_codegen
    PUBLIC
        cumetal_metal_codegen
        cumetal_passes
)

target_compile_options(cumetal_ptx_codegen
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CUMETAL_ENABLE_CUDA_REGISTRATION)
    set(CUMETAL_REGISTRATION_SOURCE runtime/registration/registration.cpp)
else()
    set(CUMETAL_REGISTRATION_SOURCE runtime/registration/registration_stub.cpp)
endif()

find_path(CUMETAL_LZ4_INCLUDE_DIR NAMES lz4.h REQUIRED)
find_library(CUMETAL_LZ4_LIBRARY NAMES lz4 REQUIRED)
find_path(CUMETAL_ZSTD_INCLUDE_DIR NAMES zstd.h REQUIRED)
find_library(CUMETAL_ZSTD_LIBRARY NAMES zstd REQUIRED)

add_library(cumetal_runtime SHARED
    runtime/cache/module_cache.cpp
    runtime/common/fatbin_elf.cpp
    runtime/common/fatbin_ptx.cpp
    runtime/common/metal_math_mode.cpp
    runtime/driver/cuda_driver.cpp
    runtime/error/library_conflict.cpp
    runtime/registration/native_registration.cpp
    ${CUMETAL_REGISTRATION_SOURCE}
    runtime/rt/allocation_table.cpp
    runtime/rt/library_kernel_source.cpp
    runtime/rt/cublas.cpp
    runtime/rt/cufft.cpp
    runtime/rt/cufft_metal.cpp
    runtime/rt/curand.cpp
    runtime/rt/cusparse.cpp
    runtime/rt/cusolver.cpp
    runtime/rt/cublaslt.cpp
    runtime/rt/cudnn.cpp
    runtime/rt/nvml.cpp
    runtime/rt/nccl.cpp
    runtime/rt/nvrtc.cpp
    runtime/rt/nvrtc_options.cpp
    runtime/rt/cuda_runtime.cpp
    runtime/rt/hash_shim.cpp
    runtime/metal_backend/metalcpp_impl.cpp
    runtime/metal_backend/metal_backend.mm
)

file(SHA256
    "${CMAKE_SOURCE_DIR}/third_party/VF64-metal/Sources/VF64Metal/Shaders/Interop/VF64Support.metal"
    CUMETAL_VF64_SUPPORT_SHA256
)

set_target_properties(cumetal_runtime
    PROPERTIES
        OUTPUT_NAME cumetal
)

set_source_files_properties(
    runtime/metal_backend/metal_backend.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(cumetal_runtime
    PUBLIC
        compiler/air_emitter/include
        compiler/common/include
        compiler/ptx/include
        runtime/api
        runtime/cache
        runtime/common
        runtime/error
        runtime/registration
        runtime/rt
        runtime/metal_backend
    PRIVATE
        ${CUMETAL_LZ4_INCLUDE_DIR}
        ${CUMETAL_ZSTD_INCLUDE_DIR}
)

target_include_directories(cumetal_runtime SYSTEM PRIVATE
    third_party/VkFFT
    third_party/metal-cpp
)

target_link_libraries(cumetal_runtime
    PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework QuartzCore"
        ${CUMETAL_LZ4_LIBRARY}
        ${CUMETAL_ZSTD_LIBRARY}
)

target_compile_definitions(cumetal_runtime
    PRIVATE
        CUMETAL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        CUMETAL_VF64_SUPPORT_SHA256="${CUMETAL_VF64_SUPPORT_SHA256}"
        VKFFT_BACKEND=5
)

target_compile_options(cumetal_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(APPLE)
    add_custom_command(
        TARGET cumetal_runtime
        POST_BUILD
        # Files produced inside a quarantined/provenance-marked checkout can
        # inherit com.apple.provenance. Re-sign after removing it so dyld does
        # not reject an otherwise valid local development dylib.
        COMMAND ${CMAKE_COMMAND}
                -DCUMETAL_BINARY=$<TARGET_FILE:cumetal_runtime>
                -P
                ${CMAKE_SOURCE_DIR}/cmake/prepare_macos_binary.cmake
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcublas.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcurand.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcufft.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcusparse.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcusolver.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcublasLt.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libcudnn.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libnvidia-ml.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libnccl.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                $<TARGET_FILE_NAME:cumetal_runtime>
                $<TARGET_FILE_DIR:cumetal_runtime>/libnvrtc.dylib
    )

    if(CUMETAL_ENABLE_BINARY_SHIM)
        add_custom_command(
            TARGET cumetal_runtime
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE_NAME:cumetal_runtime>
                    $<TARGET_FILE_DIR:cumetal_runtime>/libcuda.dylib
        )
    endif()
endif()

add_library(cumetal_air_validate STATIC
    compiler/air_validate/src/validator.cpp
)

target_include_directories(cumetal_air_validate
    PUBLIC
        compiler/air_validate/include
)

target_link_libraries(cumetal_air_validate
    PUBLIC
        cumetal_common
)

target_compile_options(cumetal_air_validate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_validate
    compiler/air_validate/src/main.cpp
)

target_link_libraries(air_validate
    PRIVATE
        cumetal_air_validate
)

add_library(cumetal_air_emitter STATIC
    compiler/air_emitter/src/emitter.cpp
)

target_include_directories(cumetal_air_emitter
    PUBLIC
        compiler/air_emitter/include
)

target_link_libraries(cumetal_air_emitter
    PUBLIC
        cumetal_air_validate
)

target_compile_options(cumetal_air_emitter
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Only the last-resort branch of metal_support_dir() uses this, so that an
# uninstalled build still finds its own support sources. An installed binary
# resolves them from its own location instead -- see metal_support_dir().
target_compile_definitions(cumetal_air_emitter
    PRIVATE
        CUMETAL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_executable(cumetal-air-emitter
    compiler/air_emitter/src/main.cpp
)

target_link_libraries(cumetal-air-emitter
    PRIVATE
        cumetal_air_emitter
)

target_link_libraries(cumetal_runtime
    PRIVATE
        cumetal_air_emitter
        cumetal_ptx_codegen
)

add_executable(cumetalc
    compiler/cumetalc/main.cpp
)

target_link_libraries(cumetalc
    PRIVATE
        cumetal_air_emitter
        cumetal_ptx_codegen
)

target_compile_options(cumetalc
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_definitions(cumetalc
    PRIVATE
        CUMETAL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        CUMETAL_VERSION_STRING="${PROJECT_VERSION}"
)

add_executable(cumetal
    tools/cumetal/main.cpp
)

target_compile_options(cumetal
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_definitions(cumetal
    PRIVATE
        CUMETAL_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        CUMETAL_VERSION_STRING="${PROJECT_VERSION}"
)

add_executable(cumetal-ptx2llvm
    compiler/ptx/src/main.cpp
)

target_link_libraries(cumetal-ptx2llvm
    PRIVATE
        cumetal_common
        cumetal_ptx_codegen
)

target_compile_options(cumetal-ptx2llvm
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_inspect
    tools/air_inspect/main.cpp
)

target_link_libraries(air_inspect
    PRIVATE
        cumetal_common
)

target_compile_options(air_inspect
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(ptx_diff
    tools/ptx_diff/main.cpp
)

target_link_libraries(ptx_diff
    PRIVATE
        cumetal_ptx_codegen
)

target_compile_options(ptx_diff
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(cumetal_bench
    tools/cumetal_bench/main.cpp
)

target_link_libraries(cumetal_bench
    PRIVATE
        cumetal_runtime
)

target_compile_options(cumetal_bench
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(cumetal_fft_bench
    tools/cumetal_fft_bench/main.cpp
)

target_link_libraries(cumetal_fft_bench
    PRIVATE
        cumetal_runtime
)

target_compile_options(cumetal_fft_bench
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(APPLE)
    foreach(cumetal_tool_target IN ITEMS
            cumetal_ptxas_shim
            cumetal_fatbinary_shim
            air_validate
            cumetal-air-emitter
            cumetalc
            cumetal-ptx2llvm
            air_inspect
            ptx_diff
            cumetal_bench)
        add_custom_command(
            TARGET ${cumetal_tool_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                    -DCUMETAL_BINARY=$<TARGET_FILE:${cumetal_tool_target}>
                    -P
                    ${CMAKE_SOURCE_DIR}/cmake/prepare_macos_binary.cmake
        )
    endforeach()
endif()

if(CUMETAL_BUILD_TESTS)
    enable_testing()
    include(cmake/CuMetalToolchain.cmake)
    cumetal_discover_metal_toolchain(CUMETAL_TEST_METAL_TOOLCHAIN)
    if(CUMETAL_TEST_METAL_TOOLCHAIN)
        message(STATUS
            "CTest Metal toolchain: ${CUMETAL_TEST_METAL_TOOLCHAIN}")
    endif()
    add_subdirectory(tests/air_abi)
    add_subdirectory(tests/conformance)
    add_subdirectory(tests/functional)
    add_subdirectory(tests/fp64_ulp)
    add_subdirectory(tests/air_dialect)
    add_subdirectory(tests/cuda_projects)
    add_subdirectory(tests/ptx_sweep)
    add_subdirectory(tests/unit)

    # Phase 5 performance benchmark test (skipped if xcrun is unavailable).
    # The gate measures host launch overhead as well as GPU execution and is a
    # release criterion; an unoptimized Debug runtime is not a comparable
    # performance configuration. Keep it out of Debug correctness inventories.
    # Uses --max-ratio=2.0 to enforce the spec §5.7 / §10.6 gate: CuMetal GPU time must
    # be within 2x of native Metal GPU time for all memory-bound kernels.
    if(APPLE AND CMAKE_BUILD_TYPE STREQUAL "Release")
        add_test(
            NAME bench_phase5_all_kernels
            COMMAND bash
                ${CMAKE_SOURCE_DIR}/scripts/run_bench_phase5.sh
                $<TARGET_FILE:cumetal_bench>
                ${CMAKE_SOURCE_DIR}/scripts/generate_bench_metallib.sh
                ${CMAKE_CURRENT_BINARY_DIR}/bench_phase5
        )
        set_tests_properties(
            bench_phase5_all_kernels
            PROPERTIES
                SKIP_RETURN_CODE 77
                LABELS "benchmark;gpu;performance"
        )
    endif()

    cumetal_apply_metal_toolchain_to_tests(
        "${CUMETAL_TEST_METAL_TOOLCHAIN}"
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests/air_abi"
        "${CMAKE_SOURCE_DIR}/tests/conformance"
        "${CMAKE_SOURCE_DIR}/tests/functional"
        "${CMAKE_SOURCE_DIR}/tests/fp64_ulp"
        "${CMAKE_SOURCE_DIR}/tests/cuda_projects"
        "${CMAKE_SOURCE_DIR}/tests/ptx_sweep"
        "${CMAKE_SOURCE_DIR}/tests/unit"
    )
endif()

install(TARGETS
    cumetal_runtime
    air_inspect
    air_validate
    cumetal-air-emitter
    cumetal
    cumetalc
    cumetal-ptx2llvm
    ptx_diff
    cumetal_bench
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# `cumetalc foo.cu -o foo` execs these from PATH while clang compiles device code, so an
# installed cumetalc is useless without them. They live in libexec (not bin) because they
# shadow real NVIDIA tool names and must never land on a user's PATH by accident.
install(TARGETS
    cumetal_ptxas_shim
    cumetal_fatbinary_shim
    RUNTIME DESTINATION libexec/cumetal/cuda_toolchain
)

if(APPLE)
    install(CODE [[
        file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/lib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcublas.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcurand.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcufft.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcusparse.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcusolver.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcublasLt.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libcudnn.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libnvidia-ml.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libnccl.dylib")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E create_symlink
                    "libcumetal.dylib"
                    "${CMAKE_INSTALL_PREFIX}/lib/libnvrtc.dylib")
    ]])

    if(CUMETAL_ENABLE_BINARY_SHIM)
        install(CODE [[
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E create_symlink
                        "libcumetal.dylib"
                        "${CMAKE_INSTALL_PREFIX}/lib/libcuda.dylib")
        ]])
    endif()
endif()

# CuMetal's own Metal support sources. cumetalc and libcumetal compile these
# with xcrun at need (the FP64 helpers the legacy path air-links), so a release
# that omits them has a compiler that cannot finish an FP64-touching kernel on
# any machine but the one that built it.
# The PhysX patch series, so a release can build PhysX's GPU runtime without
# also fetching the source archive just for these. They are diffs against
# BSD-3-Clause PhysX and carry its source as context, so NOTICE.md and the
# verbatim upstream PHYSX-LICENSE.md ship with them -- that is the attribution
# the license asks of a source redistribution, and the reason the whole
# directory installs rather than the .patch files alone.
install(DIRECTORY scripts/physx-patches/
    DESTINATION share/cumetal/physx-patches
    USE_SOURCE_PERMISSIONS
)

# Laid out as a miniature source tree rather than two loose files: the support
# sources reach VF64-metal's shaders with a relative include
# (../../../third_party/VF64-metal/...), and mirroring compiler/ and
# third_party/ under one root keeps those includes resolving with no edit to
# the sources themselves.
install(DIRECTORY compiler/metal/support/
    DESTINATION libexec/cumetal/metal-support/compiler/metal/support
    FILES_MATCHING PATTERN "*.metal"
)

install(DIRECTORY third_party/VF64-metal/Sources/VF64Metal/Shaders/
    DESTINATION libexec/cumetal/metal-support/third_party/VF64-metal/Sources/VF64Metal/Shaders
    FILES_MATCHING PATTERN "*.metal"
)

# The public headers are the whole of runtime/api -- shipped as a tree rather than
# an enumeration. The enumeration this replaces had drifted: it listed 79 of the
# 105 headers, and none of the extensionless `cuda/std/*` ones. Among the missing
# were vector_types.h and vector_functions.h, which PhysX's host code includes
# (gpucommon/include/cutil_math.h), so a release could not build PhysX's GPU
# runtime against its own include/ -- reported from the 0.6.0 tarball.
# Anything added under runtime/api is public by construction; keep it that way.
install(DIRECTORY runtime/api/
    DESTINATION include
    PATTERN "CMakeLists.txt" EXCLUDE
)
