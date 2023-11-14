make cusparse_hybrid_bsr_coo_spmm
CUSPARSELT_DIR=/work/shared/common/Libraries/libcusparse_lt-linux-x86_64-0.3.0.3-archive
CUTENSOR_DIR=/work/shared/common/Libraries/libcutensor-linux-x86_64-1.6.1.5-archive
export LD_LIBRARY_PATH=/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUSPARSELT_DIR}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUTENSOR_DIR}/lib/11.0:${LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATSETS1=(
        ogbl-ddi.mtx_bCOO.npz
        # nemeth21_row_major.mtx_bCOO.npz
        #     email-Eu-core_row_major.mtx_bCOO.npz
        #   heart1_row_major.mtx_bCOO.npz
        #   cari_row_major.mtx_bCOO.npz
        #   c8_mat11_row_major.mtx_bCOO.npz
        #   bibd_18_9_row_major.mtx_bCOO.npz
          # roadNet-PA_row_major.mtx_bCOO.npz
          )
DATSETS2=(
        ogbl-ddi.mtx_bBSR.npz
        # nemeth21_row_major.mtx_bBSR.npz
        #     email-Eu-core_row_major.mtx_bBSR.npz
        #   heart1_row_major.mtx_bBSR.npz
        #   cari_row_major.mtx_bBSR.npz
        #   c8_mat11_row_major.mtx_bBSR.npz
        #   bibd_18_9_row_major.mtx_bBSR.npz
          # roadNet-PA_row_major.mtx_bBSR.npz
          )

BUILD_DIR=./build
for ((i=0;i<${#DATSETS1[@]};i++)); do
    echo ${BUILD_DIR}/cusparse_hybrid_bsr_coo_spmm ${DATSETS1[i]} ${DATSETS2[i]}
    ${BUILD_DIR}/cusparse_hybrid_bsr_coo_spmm $DATASET_PATH/${DATSETS1[i]} $DATASET_PATH/${DATSETS2[i]}
done

# for dataset1, dataset2 in ${DATSETS1[@]}, ${DATSETS2[@]}
# do
#     echo ${BUILD_DIR}/cusparse_hybrid_spmm $dataset1 $dataset2
#     ${BUILD_DIR}/cusparse_hybrid_spmm $DATASET_PATH/$dataset1 $DATASET_PATH/$dataset2
# done
