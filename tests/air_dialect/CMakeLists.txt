add_test(
    NAME air_dialect_codegen_matrix
    COMMAND bash
        ${CMAKE_CURRENT_SOURCE_DIR}/run_dialect_codegen.sh
        $<TARGET_FILE:cumetal-ptx2llvm>
        ${CMAKE_CURRENT_BINARY_DIR}/air_dialect_codegen
)
set_tests_properties(air_dialect_codegen_matrix PROPERTIES
    LABELS "air-compiler;hosted"
    TIMEOUT 120
)
