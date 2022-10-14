make cusparse_spmspv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel
SP_VEC0=0.01_
SP_VEC1=0.1_
SP_VEC2=0.5_
SMPOST=csr_float32.npz
SVPOST=csr_float.npz

DATSETS=(2cubes_sphere_
         CollegeMsg_
         amazon0312_
         ca-CondMat_
         cage12_
         email-Eu-core_
         filter3D_
         m133-b3_
         mario002_
         offshore_
         p2p-Gnutella31_
         poisson3Da_
         scircuit_
         web-Google_
         wiki-Vote_)

BUILD_DIR=./build

for dataset in ${DATSETS[@]}
do
    echo ${BUILD_DIR}/cusparse_spmspv $dataset$SP_VEC0$SVPOST
    ${BUILD_DIR}/cusparse_spmspv $DATASET_PATH/$dataset$SMPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC0$SVPOST

    echo ${BUILD_DIR}/cusparse_spmspv $dataset$SP_VEC1$SVPOST
    ${BUILD_DIR}/cusparse_spmspv $DATASET_PATH/$dataset$SMPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC1$SVPOST

    echo ${BUILD_DIR}/cusparse_spmspv $dataset$SP_VEC2$SVPOST
    ${BUILD_DIR}/cusparse_spmspv $DATASET_PATH/$dataset$SMPOST $DATASET_PATH/sparse_vec/$dataset$SP_VEC2$SVPOST
done
