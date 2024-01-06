rm csr_csc_csc_spgemm csr_csc_csr_spgemm csr_csc_csc_spgemm csc_csc_csc_spgemm dcsr_dcsr_dcsr_spgemm dcsr_dcsc_dcsr_spgemm dcsr_dcsc_dcsc_spgemm dcsc_dcsc_dcsc_spgemm


mlir-opt ./sparse_tensor_csr_csr_csr_spgemm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_csr_csr_spgemm.o
clang++ csr_csr_csr_spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csr_csr_spgemm
# mlir-opt ./sparse_tensor_csr_csc_csr_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csc_csr_spgemm
# mlir-opt ./sparse_tensor_csr_csc_csc_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csc_csc_spgemm
mlir-opt ./sparse_tensor_csc_csc_csc_spgemm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csc_csc_csc_spgemm.o
clang++ csc_csc_csc_spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csc_csc_csc_spgemm

# mlir-opt ./sparse_tensor_dcsr_dcsr_dcsr_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsr_dcsr_spgemm
# mlir-opt ./sparse_tensor_dcsr_dcsc_dcsr_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsc_dcsr_spgemm
# mlir-opt ./sparse_tensor_dcsr_dcsc_dcsc_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_dcsc_dcsc_spgemm
# mlir-opt ./sparse_tensor_dcsc_dcsc_dcsc_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spgemm.o
# clang++ spgemm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsc_dcsc_dcsc_spgemm


DATASET_PATH=/work/shared/common/datasets/UniSparse_dataset

DATASETS=( "web-Google_row_major.mtx" 
            "mario002_row_major.mtx"
            "amazon0312_row_major.mtx"
            # "m133-b3_row_major.mtx"
            "scircuit_row_major.mtx"
            "p2p-Gnutella31_row_major.mtx"
            # "offshore_row_major.mtx"
            "cage12_row_major.mtx" 
            "2cubes_sphere_row_major.mtx"
            "filter3D_row_major.mtx"
            "ca-CondMat_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "poisson3Da_row_major.mtx"
            # "CollegeMsg_row_major.mtx"
            "email-Eu-core_row_major.mtx" )

for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    ./csr_csr_csr_spgemm
    # ./csr_csc_csr_spgemm
    # ./csr_csc_csc_spgemm
    ./csc_csc_csc_spgemm
    # ./dcsr_dcsr_dcsr_spgemm
    # ./dcsr_dcsc_dcsr_spgemm
    # ./dcsr_dcsc_dcsc_spgemm
    # ./dcsc_dcsc_dcsc_spgemm

done

# for dataset in "${DATASETS2[@]}"
# do
#     echo $dataset
#     export TENSOR0=$DATASET_PATH/$dataset

#     ./csr_csr_csr_spgemm
#     ./csc_csc_csc_spgemm
#     ./dcsr_dcsr_dcsr_spgemm
#     ./dcsc_dcsc_dcsc_spgemm

# done
