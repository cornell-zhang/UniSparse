set(LLVM_OPTIONAL_SOURCES
  UniSparseUtils.cpp
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

find_package(OpenMP REQUIRED)
# find_package(CUDA REQUIRED)

include(CheckLanguage)
# check_language(CUDA)
# if (CMAKE_CUDA_COMPILER)
#   enable_language(CUDA)
# else()
#   message(SEND_ERROR
#     "Building the mlir cuda runner requires a working CUDA install")
# endif()

# We need the libcuda.so library.
# find_library(CUDA_LIBRARIES cuda REQUIRED)
# find_library(CUDA_cusparse_LIBRARY cuda REQUIRED)

add_mlir_library(mlir_unisparse_runner_utils
  SHARED
  UniSparseUtils.cpp

  EXCLUDE_FROM_LIBMLIR
)

set_property(TARGET mlir_unisparse_runner_utils PROPERTY CXX_STANDARD 14)
target_compile_definitions(mlir_unisparse_runner_utils PRIVATE mlir_unisparse_runner_utils_EXPORTS)
# target_include_directories(mlir_unisparse_runner_utils PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# target_link_libraries(mlir_unisparse_runner_utils PRIVATE OpenMP::OpenMP_CXX PRIVATE ${CUDA_LIBRARIES} PRIVATE ${CUDA_cusparse_LIBRARY})
