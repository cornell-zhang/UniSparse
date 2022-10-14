make cusparse_spgemm_lp CUDA_CFLAGS=-DVAR=1000
make cusparse_spmm CUDA_CFLAGS=-DVAR=1000

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel/lp

PREFIX=arxiv
SPMCSR=_adj_matrix_csr_float32.npz
DATASETS=(  "_label_matrix_1_csr_float32.npz" 
            "_label_matrix_2_csr_float32.npz"
            "_label_matrix_3_csr_float32.npz" 
            "_label_matrix_4_csr_float32.npz")


BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/cusparse_spgemm_lp $PREFIX$dataset
    ${BUILD_DIR}/cusparse_spgemm_lp $DATASET_PATH/$PREFIX$SPMCSR $DATASET_PATH/$PREFIX$dataset
done

echo ${BUILD_DIR}/cusparse_spmm $PREFIX$SPMCSR
${BUILD_DIR}/cusparse_spmm $DATASET_PATH/$PREFIX$SPMCSR 

