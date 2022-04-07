make mkl_spmv

export OMP_NUM_THREADS=32

DATASET_PATH=/work/shared/common/datasets/SuiteSparse_MLIR/

# DATASETS=(  "ss1_col_major.mtx" 
#             "stomach_col_major.mtx"
#             "scircuit_col_major.mtx" 
#             "Hamrle3_col_major.mtx"
#             "Transport_col_major.mtx")

DATASETS=(  "ss1_row_major.mtx" 
            "stomach_row_major.mtx"
            "scircuit_row_major.mtx" 
            "Hamrle3_row_major.mtx"
            "Transport_row_major.mtx")

BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/mkl_spmv $dataset
    ${BUILD_DIR}/mkl_spmv $DATASET_PATH/$dataset
done
