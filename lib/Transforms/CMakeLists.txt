add_mlir_dialect_library(MLIRUniSparseTransforms
  
  LowerFormatConversionPass.cpp
  LowerStructPass.cpp
  DeadCodeEliminationPass.cpp
  UniSparseCodegen.cpp
  LowerStructConvert.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms
  ${PROJECT_SOURCE_DIR}/include/IR

  DEPENDS
  MLIRUniSparseOpsIncGen
  MLIRUniSparseAttrIncGen
  MLIRUniSparseTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransformUtils
  MLIRVectorToLLVM
  MLIRUniSparse
  )
