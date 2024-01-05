rm csr_to_csc csr_to_dcsc dcsr_to_csc dcsr_to_dcsc
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_to_csc.cpp -o csr_to_csc -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib csr_to_dcsc.cpp -o csr_to_dcsc -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_to_csc.cpp -o dcsr_to_csc -ltaco
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib dcsr_to_dcsc.cpp -o dcsr_to_dcsc -ltaco

DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

DATASETS=(  "web-Google_row_major.mtx" 
            "amazon0312_row_major.mtx" 
            "offshore_row_major.mtx"
            "wiki-Vote_row_major.mtx"
            "email-Eu-core_row_major.mtx" )

for dataset in "${DATASETS[@]}"
do
    echo ./csr_to_csc $dataset
    ./csr_to_csc $DATASET_PATH/$dataset

    echo ./csr_to_dcsc $dataset
    ./csr_to_dcsc $DATASET_PATH/$dataset

    echo ./dcsr_to_csc $dataset
    ./dcsr_to_csc $DATASET_PATH/$dataset

    echo ./dcsr_to_dcsc $dataset
    ./dcsr_to_dcsc $DATASET_PATH/$dataset

done
