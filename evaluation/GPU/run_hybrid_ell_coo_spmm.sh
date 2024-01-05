make cusparse_hybrid_ell_coo_spmm
CUSPARSELT_DIR=/work/shared/common/Libraries/libcusparse_lt-linux-x86_64-0.3.0.3-archive
CUTENSOR_DIR=/work/shared/common/Libraries/libcutensor-linux-x86_64-1.6.1.5-archive
export LD_LIBRARY_PATH=/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUSPARSELT_DIR}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUTENSOR_DIR}/lib/11.0:${LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATSETS1=(
          # nemeth21_row_major.mtx_COO.npz
        #   email-Eu-core_row_major.mtx_COO.npz
        #   c8_mat11_row_major.mtx_COO.npz
        #   heart1_row_major.mtx_COO.npz
        #   bibd_18_9_row_major.mtx_COO.npz
        #   cari_row_major.mtx_COO.npz
          roadNet-PA_row_major.mtx_COO.npz
          )
DATSETS2=(
          # nemeth21_row_major.mtx_ELL.npz
        #   email-Eu-core_row_major.mtx_ELL.npz
        #   c8_mat11_row_major.mtx_ELL.npz
        #   heart1_row_major.mtx_ELL.npz
        #   bibd_18_9_row_major.mtx_ELL.npz
        #   cari_row_major.mtx_ELL.npz
          roadNet-PA_row_major.mtx_ELL.npz
          )

BUILD_DIR=./build
for ((i=0;i<${#DATSETS1[@]};i++)); do
    echo ${BUILD_DIR}/cusparse_hybrid_ell_coo_spmm ${DATSETS1[i]} ${DATSETS2[i]}
    ${BUILD_DIR}/cusparse_hybrid_ell_coo_spmm $DATASET_PATH/${DATSETS1[i]} $DATASET_PATH/${DATSETS2[i]}
done

# for dataset1, dataset2 in ${DATSETS1[@]}, ${DATSETS2[@]}
# do
#     echo ${BUILD_DIR}/cusparse_hybrid_spmm $dataset1 $dataset2
#     ${BUILD_DIR}/cusparse_hybrid_spmm $DATASET_PATH/$dataset1 $DATASET_PATH/$dataset2
# done
