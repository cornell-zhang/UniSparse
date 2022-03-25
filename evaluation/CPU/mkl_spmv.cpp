#include <string>
#include <iostream>
#include <chrono>
#include <random>

#include "mtx_read.h"
#include "mkl_spblas.h"


float test_spmv(sparse_matrix_t* AdjMatrix, struct matrix_descr descrAdjMatrix,
                int num_src_vertices, int num_dst_vertices, int num_runs) {
    double* Vector = (double*)malloc(sizeof(double) * num_src_vertices);
    for (int i = 0; i < num_src_vertices; i++) {
        Vector[i] = 1.0;
    }
    double* Out = (double*)malloc(sizeof(double) * num_dst_vertices);
    for (int i = 0; i < num_dst_vertices; i++) {
        Out[i] = 0.0;
    }

    double alpha = 1.0;
    double beta = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha,
                        *AdjMatrix,
                        descrAdjMatrix,
                        Vector,
                        beta,
                        Out);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float average_time_in_sec = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())
        / 1000000 / num_runs;
    return average_time_in_sec;
}


int main(int argc, char* argv[]) {
    char *file_name = argv[1];

    parse_COO<double> input(file_name);

    int num_dst_vertices = input.num_rows;
    int num_src_vertices = input.num_cols;

    sparse_matrix_t AdjMatrix;
    mkl_sparse_d_create_coo(&AdjMatrix,
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

    int num_runs = 10;
    float average_time_in_sec = test_spmv(&AdjMatrix, descrAdjMatrix, num_src_vertices, num_dst_vertices, num_runs);
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
    float throughput = input.num_nnz / average_time_in_sec / 1000 / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    return 0;
}
