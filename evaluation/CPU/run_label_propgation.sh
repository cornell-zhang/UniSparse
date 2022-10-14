make mkl_spgemm_lp CXXFLAGS=-Dvar=1000
make mkl_spmm CXXFLAGS=-Dvar=1000

export OMP_NUM_THREADS=48

DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel/lp

PREFIX=arxiv
SPMCSR=_adj_matrix_row_major.mtx
DATASETS=(  "_label_matrix_1.mtx" 
            "_label_matrix_2.mtx"
            "_label_matrix_3.mtx" 
            "_label_matrix_4.mtx")


BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/mkl_spgemm_lp $PREFIX$dataset
    ${BUILD_DIR}/mkl_spgemm_lp $DATASET_PATH/$PREFIX$SPMCSR $DATASET_PATH/$PREFIX$dataset
done

echo ${BUILD_DIR}/mkl_spmm $PREFIX$SPMCSR
${BUILD_DIR}/mkl_spmm $DATASET_PATH/$PREFIX$SPMCSR 

