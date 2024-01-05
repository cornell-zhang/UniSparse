VERSAL_PATH=/work/shared/common/datasets/versatile_sparse_xcel
GBLAS_PATH=/work/shared/common/project_build/graphblas/data/sparse_matrix_graph

#DATASETS=(  "roadNet-PA_row_major.mtx"
#            "wikipedia-20051105_row_major.mtx" )

#DATASETSS=( "mouse_gene_45K_29M_csr_float32_row_major.mtx"
#            "gplus_108K_13M_csr_float32_row_major.mtx"
#            "pokec_1633K_31M_csr_float32_row_major.mtx"
#            "hollywood_1M_113M_row_major.mtx"
#            "ogbl_ppa_576K_42M_csr_float32_row_major.mtx"
#            "live_journal_5M_69M_csr_float32_row_major.mtx" )

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
    echo $dataset
    export TENSOR0=$VERSAL_PATH/$dataset

    ./coo_to_c2sr

done

for dataset in "${DATASETSS[@]}"
do
    echo $dataset
    export TENSOR0=$GBLAS_PATH/$dataset

    ./coo_to_c2sr

done
