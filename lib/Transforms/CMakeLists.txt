add_mlir_dialect_library(MLIRSparlayTransforms
  
  LowerFormatConversionPass.cpp
  LowerStructPass.cpp
  DeadCodeEliminationPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms

  DEPENDS
  MLIRSparlayOpsIncGen
  MLIRSparlayTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRStandard
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToLLVM
  )