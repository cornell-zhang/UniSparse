DATASET_PATH=/install/datasets/row_major

DATASETS=(  
            "email-Eu-core_row_major.mtx"
            "ML_Laplace_row_major.mtx"
            "Transport_row_major.mtx"
            "TSOPF_RS_b2383_row_major.mtx"
            )

mlir-opt ./UniSparse/unisparse_csr_csc.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_csc.o
clang++ csr_csc.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csc
mlir-opt ./UniSparse/unisparse_coo_dia.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o coo_dia.o
clang++ coo_dia.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o coo_dia
mlir-opt ./UniSparse/unisparse_coo_ell.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o coo_ell.o
clang++ coo_ell.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o coo_ell
mlir-opt ./UniSparse/unisparse_dcsc_bcsr.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o dcsc_bcsr.o
clang++ dcsc_bcsr.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_bcsr
mlir-opt ./UniSparse/unisparse_csb_dia_v.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csb_dia_v.o
clang++ csb_dia_v.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csb_dia_v
mlir-opt ./UniSparse/unisparse_coo_c2sr.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o coo_c2sr.o
clang++ coo_c2sr.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o coo_c2sr

mlir-opt ./sparse_tensor_dialect/sparse_tensor_csr_to_csc.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sparse_tensor_csr_csc.o
clang++ sparse_tensor_csr_csc.o -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sparse_tensor_csr_csc

g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/install/taco/include -L/install/taco/build/lib ./taco/taco_format_conversion.cpp -o taco_format_conversion -ltaco


for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    echo CSR_CSC UniSparse 
    ./csr_csc
    echo CSR_CSC SparseTensor 
    ./sparse_tensor_csr_csc
    echo CSR_CSC TACO 
    ./taco_format_conversion $DATASET_PATH/$dataset 2
    
    echo COO_DIA UniSparse 
    ./coo_dia
    echo COO_DIA TACO 
    ./taco_format_conversion $DATASET_PATH/$dataset 1

    echo COO_ELL UniSparse 
    ./coo_ell
    echo COO_ELL TACO 
    ./taco_format_conversion $DATASET_PATH/$dataset 7

    echo DCSC_BCSR UniSparse 
    ./dcsc_bcsr

    echo CSB_DIA_V UniSparse 
    ./csb_dia_v

    echo COO_C2SR UniSparse 
    ./coo_c2sr
done