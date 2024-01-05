DATASET_PATH=/work/shared/common/datasets/versatile_sparse_xcel

#DATASETS=(  "wiki-Vote_row_major.mtx"
#            "email-Eu-core_row_major.mtx"
#            "nemeth21_row_major.mtx"
#            "crystm02_row_major.mtx"
#            "cant_row_major.mtx"
#            "ML_Laplace_row_major.mtx"
#            "Transport_row_major.mtx"
#            "TSOPF_RS_b2383_row_major.mtx" )

DATASETS=(  "chem_master1_row_major.mtx"
            "majorbasis_row_major.mtx"
            "shyy161_row_major.mtx"
            "Baumann_row_major.mtx" )


for dataset in "${DATASETS[@]}"
do
    echo $dataset
    export TENSOR0=$DATASET_PATH/$dataset

    ./dcsc_to_bcsr

done
