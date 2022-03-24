#include <string>
#include <iostream>
#include <chrono>
#include <random>

#include "mtx_read.h"
#include "mkl_spblas.h"


float test_spmv(sparse_matrix_t* AdjMatrix, struct matrix_descr descrAdjMatrix,
                int num_src_vertices, int num_dst_vertices, int num_runs) {
    float* Vector = (float*)malloc(sizeof(float) * num_src_vertices);
    for (int i = 0; i < num_src_vertices; i++) {
        Vector[i] = 1.0;
    }
    float* Out = (float*)malloc(sizeof(float) * num_dst_vertices);
    for (int i = 0; i < num_dst_vertices; i++) {
        Out[i] = 0.0;
    }

    float alpha = 1.0;
    float beta = 0;
    // mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
    //                 alpha,
    //                 *AdjMatrix,
    //                 descrAdjMatrix,
    //                 Vector,
    //                 beta,
    //                 Out);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
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

    parse_COO<float> input(file_name);

    // std::string dataset = argv[1];

    // cnpy::npz_t npz = cnpy::npz_load(dataset);
    // cnpy::NpyArray npy_shape = npz["shape"];
    // cnpy::NpyArray npy_data = npz["data"];
    // cnpy::NpyArray npy_indptr = npz["indptr"];
    // cnpy::NpyArray npy_indices = npz["indices"];

    // int num_nnz = input.num_nnz;
    // int num_rows = input.num_rows;
    // int num_cols = input.num_cols;
    int num_dst_vertices = input.num_rows;
    int num_src_vertices = input.num_cols;

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

    int num_runs = 10;
    float average_time_in_sec = test_spmv(&AdjMatrix, descrAdjMatrix, num_src_vertices, num_dst_vertices, num_runs);
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
    float throughput = input.num_nnz / average_time_in_sec / 1000 / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    return 0;
}
