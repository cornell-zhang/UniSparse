export LLVM_ROOT=/work/shared/common/llvm-project-15.0.0
export PATH=$LLVM_ROOT/build/bin:$PATH
export LD_LIBRARY_PATH=$LLVM_ROOT/build/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/eigen-3.4.0:$HOME/llvm-project/mlir/lib:$HOME/llvm-project/mlir/include/mlir:$CPATH
export CPATH=$LLVM_ROOT/openmp/build/runtime/src:$CPATH
export LD_LIBRARY_PATH=$LLVM_ROOT/openmp/build/runtime/src:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/install/taco/build/lib:$LD_LIBRARY_PATH

mkdir -p build && cd build
cmake .. \
-DMLIR_DIR=$LLVM_ROOT/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=$LLVM_ROOT/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++  && \
cmake --build .

export SPLHOME=/work/shared/users/phd/jl3952/workspace/MLIR_dialect/unisparse
export LD_LIBRARY_PATH=$SPLHOME/build/lib:$LD_LIBRARY_PATH
export PATH=$SPLHOME/build/bin:$PATH

cd ..

