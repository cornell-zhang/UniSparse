rm csr_csr_csr_spgemm csr_csc_csr_spgemm csr_csc_csc_spgemm csc_csc_csc_spgemm dcsr_dcsr_dcsr_spgemm dcsr_dcsc_dcsr_spgemm dcsr_dcsc_dcsc_spgemm dcsc_dcsc_dcsc_spgemm
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_csr_csr_spgemm.cpp -o csr_csr_csr_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_csc_csr_spgemm.cpp -o csr_csc_csr_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_csc_csc_spgemm.cpp -o csr_csc_csc_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csc_csc_csc_spgemm.cpp -o csc_csc_csc_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_dcsr_dcsr_spgemm.cpp -o dcsr_dcsr_dcsr_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_dcsc_dcsr_spgemm.cpp -o dcsr_dcsc_dcsr_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_dcsc_dcsc_spgemm.cpp -o dcsr_dcsc_dcsc_spgemm -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsc_dcsc_dcsc_spgemm.cpp -o dcsc_dcsc_dcsc_spgemm -ltaco

DATASET_PATH0=/work/shared/common/datasets/versatile_sparse_xcel
DATASET_PATH1=/work/shared/users/staff/zz546/Sparse_Layout_Dialect/test/Data

PFIX0=".mtx"
PFIX1="_1.mtx"

DATASETS=( "web-Google_row_major" 
            "mario002_row_major"
            "amazon0312_row_major"
            "m133-b3_row_major"
            "scircuit_row_major"
            "p2p-Gnutella31_row_major"
            "offshore_row_major"
            "cage12_row_major"
            "cubes_sphere_row_major"
            "filter3D_row_major"
            "ca-CondMat_row_major"
            "wiki-Vote_row_major"
            "poisson3Da_row_major"
            "CollegeMsg_row_major"
            "email-Eu-core_row_major" )


for dataset in "${DATASETS[@]}"
do
    cp $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./csr_csr_csr_spgemm $dataset
    ./csr_csr_csr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./csr_csc_csr_spgemm $dataset
    ./csr_csc_csr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./csr_csc_csc_spgemm $dataset
    ./csr_csc_csc_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./csc_csc_csc_spgemm $dataset
    ./csc_csc_csc_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./dcsr_dcsr_dcsr_spgemm $dataset
    ./dcsr_dcsr_dcsr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./dcsr_dcsc_dcsr_spgemm $dataset
    ./dcsr_dcsc_dcsr_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./dcsr_dcsc_dcsc_spgemm $dataset
    ./dcsr_dcsc_dcsc_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

    echo ./dcsc_dcsc_dcsc_spgemm $dataset
    ./dcsc_dcsc_dcsc_spgemm $DATASET_PATH0/$dataset$PFIX0 $DATASET_PATH1/$dataset$PFIX1

done
