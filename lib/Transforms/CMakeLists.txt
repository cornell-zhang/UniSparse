add_mlir_dialect_library(MLIRSparlayTransforms
  
  LowerFormatConversionPass.cpp
  LowerStructPass.cpp
  DeadCodeEliminationPass.cpp
  SparlayCodegen.cpp
  LowerStructConvert.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms
  ${PROJECT_SOURCE_DIR}/include/IR

  DEPENDS
  MLIRSparlayOpsIncGen
  MLIRSparselayAttrIncGen
  MLIRSparlayTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransformUtils
  MLIRVectorToLLVM
  MLIRSparlay
  )
