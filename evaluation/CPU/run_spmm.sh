make mkl_spmm

export OMP_NUM_THREADS=32

DATASET_PATH=/work/shared/users/phd/jl3952/workspace/MLIR_dialect/sparlay/evaluation/dataset

DATASETS=(  "abb313.mtx" 
            "amazon0312.mtx" 
            "p2p-Gnutella31.mtx" 
            "wiki-Vote.mtx")

BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/mkl_spmm $dataset
    ${BUILD_DIR}/mkl_spmm $DATASET_PATH/$dataset
done