cmake -G Ninja .. \
-DMLIR_DIR=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-gpu/build/lib/cmake/mlir \
-DLLVM_EXTERNAL_LIT=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-gpu/build/bin/llvm-lit \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 

# ninja
# ninja check-sparlay
# ninja mlir-doc
