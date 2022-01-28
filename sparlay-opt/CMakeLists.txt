get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSparlay
        )
add_llvm_executable(sparlay-opt sparlay-opt.cpp)

llvm_update_compile_flags(sparlay-opt)
target_link_libraries(sparlay-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(sparlay-opt)
