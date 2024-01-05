DATASET_PATH0=/install/datasets
DATASETS=( 
            "p2p-Gnutella31_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "email-Eu-core_row_major.mtx" 
            )

g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csr_spmm.cpp -o taco_csr_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_dcsc_spmm.cpp -o taco_dcsc_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csr_csr_csr_spgemm.cpp -o taco_csr_csr_csr_spgemm -ltaco
# g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_csc_csc_csc_spgemm.cpp -o taco_csc_csc_csc_spgemm -ltaco

unisparse-opt ./UniSparse/unisparse_csr_csr_csr_spgemm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csr_csr_csr_spgemm_F64.o
clang++ unisparse_csr_csr_csr_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csr_csr_csr_spgemm_f64
unisparse-opt ./UniSparse/unisparse_csc_csc_csc_spgemm_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o unisparse_csc_csc_csc_spgemm_F64.o
clang++ unisparse_csc_csc_csc_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o unisparse_csc_csc_csc_spgemm_f64

for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH0/$dataset

    echo CSR_SpMM TACO
    ./taco_csr_spmm $DATASET_PATH0/$dataset

    echo DCSC_SpMM TACO
    ./taco_dcsc_spmm $DATASET_PATH0/$dataset

    echo CSR_CSR_CSR_SpGEMM UniSparse
    ./unisparse_csr_csr_csr_spgemm_f64
    echo CSR_CSR_CSR_SpGEMM TACO
    ./taco_csr_csr_csr_spgemm $DATASET_PATH0/$dataset $DATASET_PATH1/$dataset$PFIX1

    ./unisparse_csc_csc_csc_spgemm_f64


done



DATASET_PATH0=/work/shared/common/datasets/UniSparse_dataset
DATASET_PATH1=/work/shared/users/staff/zz546/Sparse_Layout_Dialect/test/Data

PFIX0=".mtx"
PFIX1="_1.mtx"


for dataset in "${DATASETS[@]}"
do
    cp $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./csr_csr_csr_spgemm $dataset
    ./csr_csr_csr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1


done
