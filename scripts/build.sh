export LLVM_ROOT=/install/llvm-project

cmake -B build \
-DMLIR_DIR=$LLVM_ROOT/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=$LLVM_ROOT/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 

cmake --build build/

export SPLHOME=/install/UniSparse
export LD_LIBRARY_PATH=/install/UniSparse/build/lib:$LD_LIBRARY_PATH
export PATH=/install/UniSparse/build/bin:$PATH

