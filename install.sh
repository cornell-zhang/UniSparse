## Copyright UniSparse authors. All Rights Reserved.

## Install Eigen library
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -P $EIGEN_PATH
cd $EIGEN_PATH && tar -xzf eigen-3.4.0.tar.gz

## Install LLVM/MLIR
git clone --depth 1 --branch llvmorg-15.0.0 https://github.com/llvm/llvm-project.git
mkdir -p $LLVM_PATH/llvm_project/build && cd $LLVM_PATH/llvm_project/build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
    -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON -DLLVM_BUILD_UTILS=ON \
    -DLLVM_USE_LINKER=lld -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build . -j $(nproc)

## OpenMp
cd $LLVM_PATH/llvm_project/openmp && \
    mkdir build && cd build && \
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .. 
make -j $(nproc)
export CPATH=$LLVM_PATH/llvm_project/openmp/build/runtime/src:$CPATH
export LD_LIBRARY_PATH=$LLVM_PATH/llvm_project/openmp/build/runtime/src:$LD_LIBRARY_PATH

## Install UniSparse
mkdir -p $UniSparse_PATH/UniSparse/build && cd $UniSparse_PATH/UniSparse/build
cmake .. \
   -DMLIR_DIR=$LLVM_PATH/llvm_project/build/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_PATH/llvm_project/build/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 
cmake --build . -j $(nproc)
export LD_LIBRARY_PATH=$UniSparse_PATH/UniSparse/build/lib:$LD_LIBRARY_PATH
export PATH=$UniSparse_PATH/UniSparse/build/bin:$PATH


