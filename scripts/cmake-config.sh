cmake -G Ninja -B build \
-DMLIR_DIR=/work/shared/common/llvm-project-15.0.0-gpu/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=/work/shared/common/llvm-project-15.0.0-gpu/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 