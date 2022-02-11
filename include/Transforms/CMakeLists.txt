set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(MLIRSparlayTransformsIncGen)

add_mlir_doc(Passes TransformsPasses Transforms/ -gen-pass-doc)
