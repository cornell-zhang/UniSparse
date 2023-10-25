configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(UNISPARSE_TEST_DEPENDS
        FileCheck count not
        unisparse-opt
        unisparse-translate
        )

add_lit_testsuite(check-unisparse "Running the unisparse regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${UNISPARSE_TEST_DEPENDS}
        )
set_target_properties(check-unisparse PROPERTIES FOLDER "Tests")

add_lit_testsuites(UNISPARSE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${UNISPARSE_TEST_DEPENDS})
