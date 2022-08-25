#include <string>
#include <iostream>
#include <chrono>
#include <random>

#include "mtx_read.h"
#include "mkl_spblas.h"
//#define VAR var

typedef double scalar_t;
double test_spmm(sparse_matrix_t* AdjMatrix, struct matrix_descr descrAdjMatrix,
               int cols_A, int rows_A, int cols_B, int num_runs) {
    scalar_t* MatrixB = (scalar_t*)malloc(sizeof(scalar_t) * cols_A * cols_B);
    for (int i = 0; i < cols_A * cols_B; i++) {
        MatrixB[i] = 1.0;
    }
    scalar_t* OutMatrix = (scalar_t*)malloc(sizeof(scalar_t) * rows_A * cols_B);
    for (int i = 0; i < rows_A * cols_B; i++) {
        OutMatrix[i] = 0.0;
    }

    scalar_t alpha = 1.0;
    scalar_t beta = 0;
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
        mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
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
    double total_time = double(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000;
    printf("total time: %fs\n", total_time);
    double average_time_in_sec = total_time / num_runs;
    return average_time_in_sec;
}


int main(int argc, char* argv[]) {
    char *file_name = argv[1];

    parse_CSR<scalar_t> input(file_name);

    // int num_rows = npy_shape.data<int>()[0];
    // int num_cols = npy_shape.data<int>()[2];
    // int num_dst_vertices = num_rows;
    // int num_src_vertices = num_cols;

    // float* csrVal = npy_data.data<float>();
    // MKL_INT* csrRowPtr = npy_indptr.data<MKL_INT>();
    // MKL_INT* csrColInd = npy_indices.data<MKL_INT>();

    sparse_matrix_t AdjMatrix;
    mkl_sparse_d_create_csr(&AdjMatrix,
                            SPARSE_INDEX_BASE_ZERO,
                            input.num_rows,
                            input.num_cols,
                            input.csrRowPtr,
                            input.csrRowPtr + 1,
                            input.csrColInd,
                            input.csrValue);
    mkl_sparse_optimize(AdjMatrix);

    struct matrix_descr descrAdjMatrix;
    descrAdjMatrix.type = SPARSE_MATRIX_TYPE_GENERAL;

    // std::vector<int> feat_len_values{32, 64, 128, 256, 512};
    int num_runs = var;
    int num_cols_b = 40;
    
    // for (int feat_len : feat_len_values) {
        // std::cout << "\nfeat_len is: " << feat_len << std::endl;
    double average_time_in_sec = test_spmm(&AdjMatrix, descrAdjMatrix, input.num_cols, input.num_rows, num_cols_b, num_runs);
    // }
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
    double throughput = double(input.num_nnz) * double(num_cols_b * 2) / average_time_in_sec / 1000 / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    return 0;
}
