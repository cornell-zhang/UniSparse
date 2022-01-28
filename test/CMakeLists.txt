configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SPARLAY_TEST_DEPENDS
        FileCheck count not
        sparlay-opt
        sparlay-translate
        )

add_lit_testsuite(check-sparlay "Running the sparlay regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SPARLAY_TEST_DEPENDS}
        )
set_target_properties(check-sparlay PROPERTIES FOLDER "Tests")

add_lit_testsuites(SPARLAY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SPARLAY_TEST_DEPENDS})
