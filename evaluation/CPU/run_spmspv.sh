make mkl_spmspv

export OMP_NUM_THREADS=48

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel
SP_VEC0=_0.01
SP_VEC1=_0.1
SP_VEC2=_0.5
SMCSRPOST=_row_major.mtx
SMCSCPOST=_col_major.mtx
SVPOST=.mtx

DATSETS=(2cubes_sphere
         CollegeMsg
         amazon0312
         ca-CondMat
         cage12
         email-Eu-core
         filter3D
         m133-b3
         mario002
         offshore
         p2p-Gnutella31
         poisson3Da
         scircuit
         web-Google
         wiki-Vote)

BUILD_DIR=./build

for dataset in ${DATSETS[@]}
do
    echo ${BUILD_DIR}/mkl_spmspv $dataset$SP_VEC0$SVPOST
    ${BUILD_DIR}/mkl_spmspv $DATASET_PATH/$dataset$SMCSRPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC0$SVPOST $DATASET_PATH/$dataset$SMCSCPOST

    echo ${BUILD_DIR}/mkl_spmspv $dataset$SP_VEC1$SVPOST
    ${BUILD_DIR}/mkl_spmspv $DATASET_PATH/$dataset$SMCSRPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC1$SVPOST $DATASET_PATH/$dataset$SMCSCPOST

    echo ${BUILD_DIR}/mkl_spmspv $dataset$SP_VEC2$SVPOST
    ${BUILD_DIR}/mkl_spmspv $DATASET_PATH/$dataset$SMCSRPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC2$SVPOST $DATASET_PATH/$dataset$SMCSCPOST
done
