cmake -G Ninja -B build \
-DMLIR_DIR=$LLVM_ROOT/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=$LLVM_ROOT/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
-DCNPY_LIB=/work/shared/common/datasets/UniSparse_dataset/cnpy/build \
-DCNPY_INCLUDE_LIB=/work/shared/common/datasets/UniSparse_dataset/cnpy