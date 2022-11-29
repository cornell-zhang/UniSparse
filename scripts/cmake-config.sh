cmake -G Ninja -B build \
-DMLIR_DIR=/work/shared/common/llvm-project-15.0.0-gpu/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=/work/shared/common/llvm-project-15.0.0-gpu/build/bin/llvm-lit \
-DEXTERNAL_INCLUDE_DIRS=/work/shared/users/ugrad/zd226/Sparse_Layout_Dialect/eigen-3.4.0 \
-DMLIR_LIB_DIR=/work/shared/common/llvm-project-15.0.0-gpu/mlir/lib \
-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ 

