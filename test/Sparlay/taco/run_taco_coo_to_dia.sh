rm coo_to_dia
g++ -std=c++11 -O3 -DNDEBUG -DTACO -I/work/shared/users/staff/zz546/taco/include -L/work/shared/users/staff/zz546/taco/build/lib coo_dia.cpp -o coo_to_dia -ltaco

DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel
#DATASET_PATH=/work/shared/users/ugrad/zd226/data
#DATASETS=(
#    "web-Google_row_major.mtx"
#    "amazon0312_row_major.mtx"
#    "offshore_row_major.mtx"
#    "wiki-Vote_row_major.mtx"
#    "email-Eu-core_row_major.mtx"
#)

DATASETS=(  "wiki-Vote_row_major.mtx"
            "email-Eu-core_row_major.mtx"
            "nemeth21_row_major.mtx"
            "crystm02_row_major.mtx"
            "cant_row_major.mtx"
            "ML_Laplace_row_major.mtx"
            "Transport_row_major.mtx"
            "TSOPF_RS_b2383_row_major.mtx"
            "chem_master1_row_major.mtx"
            "majorbasis_row_major.mtx"
            "shyy161_row_major.mtx"
            "Baumann_row_major.mtx" )

for dataset in "${DATASETS[@]}"
do
    echo ./coo_to_dia $dataset
    ./coo_to_dia $DATASET_PATH/$dataset 1

done