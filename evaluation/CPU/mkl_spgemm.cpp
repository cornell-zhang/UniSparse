#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include "mtx_read.h"
#include "mkl_spblas.h"
#include "mkl.h"

typedef double scalar_t;


int test_ops(MKL_INT rows, MKL_INT cols, MKL_INT *row_ptr, MKL_INT *col_idx) {
    int nnz = 0;
    for(int i = 0; i < rows; i++) {
//        std::cout << "Finish initialize" << std::endl;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            int idx = col_idx[j];
            nnz = nnz + 2* (row_ptr[idx+1] - row_ptr[idx]); 
        }
    }
    return nnz;
}

int main(int argc, char* argv[]) {
    char *file_name = argv[1];

/*
     parse_CSC<double> input(file_name);

     int num_dst_vertices = input.num_cols;
     int num_src_vertices = input.num_rows;

     sparse_matrix_t AdjMatrix;
     mkl_sparse_d_create_csc(&AdjMatrix,
                             SPARSE_INDEX_BASE_ONE,
                             input.num_rows,
                             input.num_cols,
                             input.cscColPtr,
                             input.cscColPtr + 1,
                             input.cscRowInd,
                             input.cscValue);
*/
    parse_CSR<scalar_t> input0(file_name);
    sparse_matrix_t matA;
    mkl_sparse_d_create_csr(&matA,
                            SPARSE_INDEX_BASE_ZERO,
                            input0.num_rows,
                            input0.num_cols,
                            input0.csrRowPtr,
                            input0.csrRowPtr + 1,
                            input0.csrColInd,
                            input0.csrValue);
    mkl_sparse_optimize(matA);

    parse_CSR<scalar_t> input1(file_name);
    sparse_matrix_t matB;
    mkl_sparse_d_create_csr(&matB,
                            SPARSE_INDEX_BASE_ZERO,
                            input1.num_rows,
                            input1.num_cols,
                            input1.csrRowPtr,
                            input1.csrRowPtr + 1,
                            input1.csrColInd,
                            input1.csrValue);
    mkl_sparse_optimize(matB);

    int num_runs = var;
  
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {  
        sparse_matrix_t matC = NULL;  
        mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, matA, matB, &matC);
        sparse_index_base_t indexing;
        MKL_INT rows;
        MKL_INT cols;
        MKL_INT *pointerB_C;
        MKL_INT *pointerE_C;
        MKL_INT *columns_C;
        scalar_t *values_C;
        mkl_sparse_d_export_csr(matC, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &columns_C, &values_C);
        mkl_free_buffers();
        mkl_free(matC);
        mkl_free(pointerB_C);
//        mkl_free(pointerE_C);
        mkl_free(columns_C);
        mkl_free(values_C);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float total_time = float(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000;
    printf("total time: %fs\n", total_time);
    float average_time_in_sec = total_time / num_runs;
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
 //   int ops = test_ops(input0.num_rows, input0.num_cols, input0.csrRowPtr, input0.csrColInd);
 //   std::cout << "The tested ops is " << ops << std::endl;
 //   std::cout << "output nnz is " << pointerB_C[rows] << std::endl;

//    for(int i = 0; i < rows+1; i++) {
//        std::cout << "intput row_pointer[" << i << "] is " << input0.csrRowPtr[i] << std::endl;
//        std::cout << "input col_index[" << i << "] is " << input0.csrColInd[i] << std::endl;
//        std::cout << "row_pointer[" << i << "] is " << pointerB_C[i] << std::endl;
//        std::cout << "Column_idxp[" << i << "] is " << columns_C[i] << std::endl;
//    }

    return 0;
}
