add_mlir_dialect(UniSparseOps unisparse)
add_mlir_doc(UniSparseDialect UniSparseDialect IR/ -gen-dialect-doc)
add_mlir_doc(UniSparseOps UniSparseOps IR/ -gen-op-doc)


set(LLVM_TARGET_DEFINITIONS UniSparseAttr.td)
mlir_tablegen(UniSparseAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(UniSparseAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRUniSparseAttrIncGen)