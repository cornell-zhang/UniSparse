cmake -G Ninja -B build -Wno-dev \
-DMLIR_DIR=/home/dingzj/sparse/tmp/LLVM/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=/home/dingzj/sparse/tmp/LLVM/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 

# ninja
# ninja check-sparlay
# ninja mlir-doc
