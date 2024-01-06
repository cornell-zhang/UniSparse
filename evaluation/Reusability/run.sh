DATASET_PATH0=/install/datasets/row_major
DATASET_PATH1=/install/datasets/replicate
DATASETS=( 
            "wiki-Vote_row_major.mtx"
            "email-Eu-core_row_major.mtx"
            "crystm02_row_major.mtx"
            "nemeth21_row_major.mtx"
            )

unisparse-opt ./unisparse_coo_to_bcsc.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce -o coo_bcsc.mlir | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_coo_bcsc.o
clang++ unisparse_coo_bcsc.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_coo_bcsc
unisparse-opt ./unisparse_csc_spmm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csc_spmm_F64.o
clang++ unisparse_csc_spmm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csc_spmm_F64
unisparse-opt ./unisparse_csr_csc_csc_spgemm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csr_csc_csc_spgemm_F64.o
clang++ unisparse_csr_csc_csc_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csr_csc_csc_spgemm_F64

for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH0/$dataset

    echo COO_BCSC UniSparse
    ./unisparse_coo_bcsc

    echo CSC_SpMM UniSparse
    ./unisparse_csc_spmm_F64

    echo CSR_CSC_CSC_SpGEMM UniSparse
    ./unisparse_csr_csc_csc_spgemm_F64


done

