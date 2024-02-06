make mkl_pagerank

export OMP_NUM_THREADS=48

DATASET_PATH=/work/shared/common/datasets/UniSparse_dataset

DATASETS=(  
            "email-Eu-core" 
            # "wiki-Vote_row_major"
            "amazon0312_row_major"
            "p2p-Gnutella31_row_major"
            # "cit-Patents"
            # "web-Stanford"
            # "cit-HepTh"
            "soc-Slashdot0811"
            # "soc-Epinions1"
            "email-Enron"
            "ca-CondMat"
            # "as-735"
            "ca-HepTh"
            "loc-Brightkite"
            # # "test_col_norm.mtx"
            # # "gplus_108K_13M_csr_float32_row_major_col_norm.mtx"
            # # "pokec_1633K_31M_csr_float32_row_major_col_norm.mtx"
            # # "live_journal_row_major_col_norm.mtx"
            # "wiki-Vote_row_major_col_norm.mtx"
            # # "web-Google_row_major_col_norm.mtx"
            # "amazon0312_row_major_col_norm.mtx"
            # "p2p-Gnutella31_row_major_col_norm.mtx"
 
            # "ML_Geer_row_major_col_norm.mtx" 
            # "ss_row_major_col_norm.mtx"
            # "ML_Laplace_row_major_col_norm.mtx"
            # "Transport_row_major_col_norm.mtx"
            # "rajat31_row_major_col_norm.mtx"
            # "TSOPF_RS_b2383_row_major_col_norm.mtx"
            # "memchip_row_major_col_norm.mtx"
            # "vas_stokes_1M_row_major_col_norm.mtx"
        )

SUFFIX="_col_norm.mtx"
BUILD_DIR=./build

for dataset in "${DATASETS[@]}"
do
    echo ${BUILD_DIR}/mkl_pagerank $dataset$SUFFIX
    ${BUILD_DIR}/mkl_pagerank $DATASET_PATH/$dataset$SUFFIX
done
