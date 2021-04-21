configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GRAPHBLAS_TEST_DEPENDS
        FileCheck count not
        graphblas-opt

        )

add_lit_testsuite(check-graphblas "Running the graphblas regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GRAPHBLAS_TEST_DEPENDS}
        )
set_target_properties(check-graphblas PROPERTIES FOLDER "Tests")

add_lit_testsuites(GRAPHBLAS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GRAPHBLAS_TEST_DEPENDS})
