get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRUniSparseTranslation
  EmitTapaHLS.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Translation

  LINK_LIBS PUBLIC
  ${dialect_libs}
  MLIRUniSparse
  )
