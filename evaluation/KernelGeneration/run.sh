DATASET_PATH0=/install/datasets/row_major
DATASET_PATH1=/install/datasets/replicate
DATASETS=( 
            "p2p-Gnutella31_row_major"
            "wiki-Vote_row_major"
            "email-Eu-core_row_major" 
            )
PFIX0=".mtx"
PFIX1="_1.mtx"

unisparse-opt ./UniSparse/unisparse_csr_spmm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csr_spmm_F64.o
clang++ unisparse_csr_spmm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csr_spmm_F64
unisparse-opt ./UniSparse/unisparse_dcsc_spmm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_dcsc_spmm_F64.o
clang++ unisparse_dcsc_spmm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_dcsc_spmm_F64
unisparse-opt ./UniSparse/unisparse_csr_csr_csr_spgemm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csr_csr_csr_spgemm_F64.o
clang++ unisparse_csr_csr_csr_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csr_csr_csr_spgemm_F64
unisparse-opt ./UniSparse/unisparse_csc_csc_csc_spgemm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csc_csc_csc_spgemm_F64.o
clang++ unisparse_csc_csc_csc_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csc_csc_csc_spgemm_F64

mlir-opt ./sparse_tensor_dialect/sparse_tensor_csr_spmm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sparse_tensor_csr_spmm_F64.o
clang++ sparse_tensor_csr_spmm_F64.o -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sparse_tensor_csr_spmm_F64
mlir-opt ./sparse_tensor_dialect/sparse_tensor_dcsc_spmm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sparse_tensor_dcsc_spmm_F64.o
clang++ sparse_tensor_dcsc_spmm_F64.o -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sparse_tensor_dcsc_spmm_F64
mlir-opt ./sparse_tensor_dialect/sparse_tensor_csr_csr_csr_spgemm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sparse_tensor_csr_csr_csr_spgemm_F64.o
clang++ sparse_tensor_csr_csr_csr_spgemm_F64.o -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sparse_tensor_csr_csr_csr_spgemm_F64
mlir-opt ./sparse_tensor_dialect/sparse_tensor_csc_csc_csc_spgemm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sparse_tensor_csc_csc_csc_spgemm_F64.o
clang++ sparse_tensor_csc_csc_csc_spgemm_F64.o -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sparse_tensor_csc_csc_csc_spgemm_F64

g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csr_spmm.cpp -o taco_csr_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_dcsc_spmm.cpp -o taco_dcsc_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csr_csr_csr_spgemm.cpp -o taco_csr_csr_csr_spgemm -ltaco
# g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csc_csc_csc_spgemm.cpp -o taco_csc_csc_csc_spgemm -ltaco


for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH0/$dataset$PFIX0

    echo CSR_SpMM UniSparse
    ./unisparse_csr_spmm_F64
    echo CSR_SpMM SparseTensor
    ./sparse_tensor_csr_spmm_F64
    echo CSR_SpMM TACO
    ./taco_csr_spmm $DATASET_PATH0/$dataset$PFIX0

    echo DCSC_SpMM UniSparse
    ./unisparse_dcsc_spmm_F64
    echo DCSC_SpMM SparseTensor
    ./sparse_tensor_dcsc_spmm_F64
    echo DCSC_SpMM TACO
    ./taco_dcsc_spmm $DATASET_PATH0/$dataset$PFIX0

    echo CSR_CSR_CSR_SpGEMM UniSparse
    ./unisparse_csr_csr_csr_spgemm_F64
    echo CSR_CSR_CSR_SpGEMM SparseTensor
    ./sparse_tensor_csr_csr_csr_spgemm_F64
    echo CSR_CSR_CSR_SpGEMM TACO
    ./taco_csr_csr_csr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo CSC_CSC_CSC_SpGEMM UniSparse
    ./unisparse_csc_csc_csc_spgemm_F64
    echo CSC_CSC_CSC_SpGEMM SparseTensor
    ./sparse_tensor_csc_csc_csc_spgemm_F64

done

