make mkl_spmm

export OMP_NUM_THREADS=48

DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATASETS=(  "2cubes_sphere_row_major.mtx"
            "cage12_row_major.mtx"
            "email-Eu-core_row_major.mtx"
            "mario002_row_major.mtx"
            "poisson3Da_row_major.mtx"
            "ca-CondMat_row_major.mtx"
            "CollegeMsg_row_major.mtx"
            "filter3D_row_major.mtx"
            "offshore_row_major.mtx"
            "scircuit_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "amazon0312_row_major.mtx"
            "web-Google_row_major.mtx"
            "p2p-Gnutella31_row_major.mtx"
            "m133-b3_row_major.mtx")

BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/mkl_spmm $dataset
    ${BUILD_DIR}/mkl_spmm $DATASET_PATH/$dataset
done