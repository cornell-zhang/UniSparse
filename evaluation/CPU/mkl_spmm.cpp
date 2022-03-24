#include <string>
#include <iostream>
#include <chrono>
#include <random>

#include "mtx_read.h"
#include "mkl_spblas.h"


float test_spmm(sparse_matrix_t* AdjMatrix, struct matrix_descr descrAdjMatrix,
               int cols_A, int rows_A, int cols_B, int num_runs) {
    float* MatrixB = (float*)malloc(sizeof(float) * cols_A * cols_B);
    for (int i = 0; i < cols_A * cols_B; i++) {
        MatrixB[i] = 1.0;
    }
    float* OutMatrix = (float*)malloc(sizeof(float) * rows_A * cols_B);
    for (int i = 0; i < rows_A * cols_B; i++) {
        OutMatrix[i] = 0.0;
    }

    float alpha = 1.0;
    float beta = 0;
    // mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
    //                 alpha,
    //                 *AdjMatrix,
    //                 descrAdjMatrix,
    //                 SPARSE_LAYOUT_ROW_MAJOR,
    //                 MatrixB,
    //                 feat_len,
    //                 feat_len,
    //                 beta,
    //                 OutMatrix,
    //                 feat_len);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        *AdjMatrix,
                        descrAdjMatrix,
                        SPARSE_LAYOUT_COLUMN_MAJOR,
                        MatrixB,
                        cols_B,
                        cols_B,
                        beta,
                        OutMatrix,
                        cols_B);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "average time of " << num_runs << " runs: "
    //             << float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000 / num_runs
    //             << " seconds" << std::endl;
    
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    return average_time_in_sec;
}


int main(int argc, char* argv[]) {
    char *file_name = argv[1];

    parse_COO<float> input(file_name);

    // int num_rows = npy_shape.data<int>()[0];
    // int num_cols = npy_shape.data<int>()[2];
    // int num_dst_vertices = num_rows;
    // int num_src_vertices = num_cols;

    // float* csrVal = npy_data.data<float>();
    // MKL_INT* csrRowPtr = npy_indptr.data<MKL_INT>();
    // MKL_INT* csrColInd = npy_indices.data<MKL_INT>();

    sparse_matrix_t AdjMatrix;
    mkl_sparse_s_create_coo(&AdjMatrix,
                            SPARSE_INDEX_BASE_ONE,
                            input.num_rows,
                            input.num_cols,
                            input.num_nnz,
                            input.cooRowInd,
                            input.cooColInd,
                            input.cooValue);
    mkl_sparse_optimize(AdjMatrix);

    struct matrix_descr descrAdjMatrix;
    descrAdjMatrix.type = SPARSE_MATRIX_TYPE_GENERAL;

    // std::vector<int> feat_len_values{32, 64, 128, 256, 512};
    int num_runs = 10;
    
    // for (int feat_len : feat_len_values) {
        // std::cout << "\nfeat_len is: " << feat_len << std::endl;
    float average_time_in_sec = test_spmm(&AdjMatrix, descrAdjMatrix, input.num_cols, input.num_rows, input.num_cols, num_runs);
    // }
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
    float throughput = input.num_nnz / average_time_in_sec / 1000 / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    return 0;
}
