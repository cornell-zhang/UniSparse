make cusparse_spmv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATSETS=(2cubes_sphere_csr_float32.npz
         CollegeMsg_csr_float32.npz
         amazon0312_csr_float32.npz
         ca-CondMat_csr_float32.npz
         cage12_csr_float32.npz
         email-Eu-core_csr_float32.npz
         filter3D_csr_float32.npz
         m133-b3_csr_float32.npz
         mario002_csr_float32.npz
         offshore_csr_float32.npz
         p2p-Gnutella31_csr_float32.npz
         poisson3Da_csr_float32.npz
         scircuit_csr_float32.npz
         web-Google_csr_float32.npz
         wiki-Vote_csr_float32.npz)

BUILD_DIR=./build

for dataset in ${DATSETS[@]}
do
    echo ${BUILD_DIR}/cusparse_spmv $dataset
    ${BUILD_DIR}/cusparse_spmv $DATASET_PATH/$dataset
done
