set(LLVM_OPTIONAL_SOURCES
  SparlayUtils.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

find_package(OpenMP REQUIRED)

add_mlir_library(mlir_sparlay_runner_utils
  SHARED
  SparlayUtils.cpp

  EXCLUDE_FROM_LIBMLIR
  )
set_property(TARGET mlir_sparlay_runner_utils PROPERTY CXX_STANDARD 11)
target_compile_definitions(mlir_sparlay_runner_utils PRIVATE mlir_sparlay_runner_utils_EXPORTS)
target_link_libraries(mlir_sparlay_runner_utils PRIVATE OpenMP::OpenMP_CXX)
