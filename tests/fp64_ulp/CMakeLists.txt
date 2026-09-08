# Drift gates: how far each FP64 policy lands from hardware binary64, in ULPs.
foreach(fp64_case IN ITEMS "ieee64;0" "wide48;131072" "fast48;65536")
    list(GET fp64_case 0 fp64_mode)
    list(GET fp64_case 1 fp64_budget)
    add_test(
        NAME fp64_ulp_${fp64_mode}
        COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_fp64_ulp.sh
            $<TARGET_FILE:cumetalc>
            ${CMAKE_CURRENT_SOURCE_DIR}/fp64_ulp.cu
            ${CMAKE_CURRENT_BINARY_DIR}/fp64_ulp_${fp64_mode}
            ${fp64_mode}
            ${fp64_budget}
    )
    set_tests_properties(fp64_ulp_${fp64_mode} PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "gpu;fp64;accuracy"
        TIMEOUT 600
    )
endforeach()
