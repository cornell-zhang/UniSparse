rm csr_spmm csc_spmm dcsr_spmm dcsc_spmm csr_spmv csc_spmv dcsr_spmv dcsc_spmv

mlir-opt ./sparse_tensor_csr_spmm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmm
mlir-opt ./sparse_tensor_csc_spmm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_spmm
mlir-opt ./sparse_tensor_dcsr_spmm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_spmm
mlir-opt ./sparse_tensor_dcsc_spmm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o
clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_spmm

mlir-opt ./sparse_tensor_csr_spmv.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmv
mlir-opt ./sparse_tensor_csc_spmv.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_spmv
mlir-opt ./sparse_tensor_dcsr_spmv.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_spmv
mlir-opt ./sparse_tensor_dcsc_spmv.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
clang++ spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_spmv


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
