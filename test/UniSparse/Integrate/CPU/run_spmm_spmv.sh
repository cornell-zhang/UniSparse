rm csr_spmm csc_spmm dcsr_spmm dcsc_spmm csr_spmv csc_spmv dcsr_spmv dcsc_spmv


unisparse-opt ./unisparse_csr_spmm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmm
unisparse-opt ./unisparse_csc_spmm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_spmm
unisparse-opt ./unisparse_dcsr_spmm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_spmm
unisparse-opt ./unisparse_dcsc_spmm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_spmm

unisparse-opt ./unisparse_csr_spmv.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmv
unisparse-opt ./unisparse_csc_spmv.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_spmv
unisparse-opt ./unisparse_dcsr_spmv.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_spmv
unisparse-opt ./unisparse_dcsc_spmv.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_spmv


DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATASETS=( "web-Google_row_major.mtx" 
            "mario002_row_major.mtx"
            "amazon0312_row_major.mtx"
            "m133-b3_row_major.mtx"
            "scircuit_row_major.mtx"
            "p2p-Gnutella31_row_major.mtx"
            "offshore_row_major.mtx"
            "cage12_row_major.mtx"
            "2cubes_sphere_row_major.mtx"
            "filter3D_row_major.mtx"
            "ca-CondMat_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "poisson3Da_row_major.mtx"
            "CollegeMsg_row_major.mtx"
            "email-Eu-core_row_major.mtx" )

for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    ./csr_spmm
    ./csc_spmm
    ./dcsr_spmm
    ./dcsc_spmm
    ./csr_spmv
    ./csc_spmv
    ./dcsr_spmv
    ./dcsc_spmv

done



