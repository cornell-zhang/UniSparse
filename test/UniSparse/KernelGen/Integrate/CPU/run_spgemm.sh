rm csr_csc_csc_spgemm csr_csc_csr_spgemm csr_csc_csc_spgemm csc_csc_csc_spgemm dcsr_dcsr_dcsr_spgemm dcsr_dcsc_dcsr_spgemm dcsr_dcsc_dcsc_spgemm dcsc_dcsc_dcsc_spgemm


unisparse-opt ./unisparse_csr_csr_csr_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csr_csr_spgemm
unisparse-opt ./unisparse_csr_csc_csr_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csc_csr_spgemm
unisparse-opt ./unisparse_csr_csc_csc_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csc_csc_spgemm
unisparse-opt ./unisparse_csc_csc_csc_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_csc_csc_spgemm

unisparse-opt ./unisparse_dcsr_dcsr_dcsr_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsr_dcsr_spgemm
unisparse-opt ./unisparse_dcsr_dcsc_dcsr_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsc_dcsr_spgemm
unisparse-opt ./unisparse_dcsr_dcsc_dcsc_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsc_dcsc_spgemm
unisparse-opt ./unisparse_dcsc_dcsc_dcsc_spgemm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_dcsc_dcsc_spgemm


DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATASETS1=( "2cubes_sphere_row_major.mtx"
            "filter3D_row_major.mtx"
            "ca-CondMat_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "poisson3Da_row_major.mtx"
            "CollegeMsg_row_major.mtx"
            "email-Eu-core_row_major.mtx" )

DATASETS2=( "web-Google_row_major.mtx" 
            "mario002_row_major.mtx"
            "amazon0312_row_major.mtx"
            "m133-b3_row_major.mtx"
            "scircuit_row_major.mtx"
            "p2p-Gnutella31_row_major.mtx"
            "offshore_row_major.mtx"
            "cage12_row_major.mtx" )

for dataset in "${DATASETS1[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    ./csr_csr_csr_spgemm
    ./csr_csc_csr_spgemm
    ./csr_csc_csc_spgemm
    ./csc_csc_csc_spgemm
    ./dcsr_dcsr_dcsr_spgemm
    ./dcsr_dcsc_dcsr_spgemm
    ./dcsr_dcsc_dcsc_spgemm
    ./dcsc_dcsc_dcsc_spgemm

done

for dataset in "${DATASETS2[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    ./csr_csr_csr_spgemm
    ./csc_csc_csc_spgemm
    ./dcsr_dcsr_dcsr_spgemm
    ./dcsc_dcsc_dcsc_spgemm

done


