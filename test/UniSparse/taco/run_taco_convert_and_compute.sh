rm csr_spmm csc_spmm dcsr_spmm dcsc_spmm csr_spmv csc_spmv dcsr_spmv dcsc_spmv
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_spmm.cpp -o csr_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csc_spmm.cpp -o csc_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_spmm.cpp -o dcsr_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsc_spmm.cpp -o dcsc_spmm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_spmv.cpp -o csr_spmv -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csc_spmv.cpp -o csc_spmv -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_spmv.cpp -o dcsr_spmv -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsc_spmv.cpp -o dcsc_spmv -ltaco

DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATASETS=( "web-Google_row_major.mtx" 
            "mario002_row_major.mtx"
            "amazon0312_row_major.mtx"
            "m133-b3_row_major.mtx"
            "scircuit_row_major.mtx"
            "p2p-Gnutella31_row_major.mtx"
            "offshore_row_major.mtx"
            "cage12_row_major.mtx"
            "cubes_sphere_row_major.mtx"
            "filter3D_row_major.mtx"
            "ca-CondMat_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "poisson3Da_row_major.mtx"
            "CollegeMsg_row_major.mtx"
            "email-Eu-core_row_major.mtx" )

for dataset in "${DATASETS[@]}"
do
    echo ./csr_spmm $dataset
    ./csr_spmm $DATASET_PATH/$dataset

    echo ./csc_spmm $dataset
    ./csc_spmm $DATASET_PATH/$dataset

    echo ./dcsr_spmm $dataset
    ./dcsr_spmm $DATASET_PATH/$dataset

    echo ./dcsc_spmm $dataset
    ./dcsc_spmm $DATASET_PATH/$dataset

    echo ./csr_spmv $dataset
    ./csr_spmv $DATASET_PATH/$dataset

    echo ./csc_spmv $dataset
    ./csc_spmv $DATASET_PATH/$dataset

    echo ./dcsr_spmv $dataset
    ./dcsr_spmv $DATASET_PATH/$dataset

    echo ./dcsc_spmv $dataset
    ./dcsc_spmv $DATASET_PATH/$dataset
done
