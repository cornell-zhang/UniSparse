#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <unordered_map>
#include <cnpy.h>
#include <cusparseLt.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

int benchmark_spmv_hybrid_ell_coo(std::string dataset1, std::string dataset2) {
    // load coo matrix
    cnpy::npz_t npz = cnpy::npz_load(dataset1);
    cnpy::NpyArray npy_shape = npz["shape"];
    cnpy::NpyArray npy_data = npz["data"];
    cnpy::NpyArray npy_row = npz["row"];
    cnpy::NpyArray npy_col = npz["col"];
    int nnz = npy_data.shape[0];
    int num_rows = npy_shape.data<int>()[0];
    int num_cols = npy_shape.data<int>()[2];
    int num_cols_b = 1000;
    float* cooVal = npy_data.data<float>();
    int* cooRow = npy_row.data<int>();
    int* cooCol = npy_col.data<int>();
    float coo_alpha = 1.0;
    float coo_beta = 0.0;

    // load ell matrix
    cnpy::npz_t npz_ell = cnpy::npz_load(dataset2);
    cnpy::NpyArray npy_num_rows = npz_ell["num_rows"];
    cnpy::NpyArray npy_num_cols = npz_ell["num_cols"];
    cnpy::NpyArray npy_ell_blocksize = npz_ell["ell_blocksize"];
    cnpy::NpyArray npy_ell_cols = npz_ell["ell_cols"];
    cnpy::NpyArray npy_num_blocks = npz_ell["num_blocks"];
    cnpy::NpyArray npy_col_ind = npz_ell["col_ind"];
    cnpy::NpyArray npy_values = npz_ell["values"];
    cnpy::NpyArray npy_ell_nnz = npz_ell["nnz"];
    int ell_num_rows = npy_num_rows.data<int>()[0];
    int ell_num_cols = npy_num_cols.data<int>()[0];
    int ell_blocksize = npy_ell_blocksize.data<int>()[0];
    int ell_cols = npy_ell_cols.data<int>()[0];
    int ell_num_blocks = npy_num_blocks.data<int>()[0];
    int ell_nnz = npy_ell_nnz.data<int>()[0];
    int* ell_col_ind = npy_col_ind.data<int>();
    float* ell_values = npy_values.data<float>();
    float ell_alpha = 1.0;
    float ell_beta = 0.0;
    std::cout << "ell_num_rows = " << ell_num_rows << std::endl;
    std::cout << "ell_num_cols = " << ell_num_cols << std::endl;
    std::cout << "ell_blocksize = " << ell_blocksize << std::endl;
    std::cout << "ell_cols = " << ell_cols << std::endl;
    std::cout << "ell_num_blocks = " << ell_num_blocks << std::endl;
    std::cout << "ell_nnz = " << ell_nnz << std::endl;
    // std::cout << "ell_num_blocks = " << ell_num_blocks << std::endl;
    // std::cout << "ell_num_blocks = " << ell_num_blocks << std::endl;

    // std::cout << "nnz:" << nnz << std::endl;
    // std::cout << "num_rows:" << num_rows << std::endl;
    // std::cout << "num_cols:" << num_cols << std::endl;

    //--------------------------------------------------------------------------
    // COO mem copy
    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;

   // device malloc
    float* cu_cooVal=0;
    cudaStat1 = cudaMalloc((void**)&cu_cooVal, nnz * sizeof(float));
    int* cu_cooRow=0;
    cudaStat2 = cudaMalloc((void**)&cu_cooRow, nnz * sizeof(int));
    int* cu_cooCol=0;
    cudaStat3 = cudaMalloc((void**)&cu_cooCol, nnz * sizeof(int));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat1 = cudaMemcpy(cu_cooVal, cooVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cu_cooRow, cooRow, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cu_cooCol, cooCol, nnz * sizeof(int), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }
    //--------------------------------------------------------------------------

    float* InMat = (float*)malloc(sizeof(float) * num_cols * num_cols_b);
    for (int i = 0; i < num_cols * num_cols_b; i++) {
        InMat[i] = 1.0;
    }
    float* OutMat = (float*)malloc(sizeof(float) * num_rows * num_cols_b);
    for (int i = 0; i < num_rows * num_cols_b; i++) {
        OutMat[i] = 0.0;
    }

    //--------------------------------------------------------------------------
    // dense matrix and output dense matrix mem copy
    float* cu_InMat_COO=0;
    cudaStat4 = cudaMalloc((void**)&cu_InMat_COO, num_cols * num_cols_b * sizeof(float));
    float* cu_OutMat_COO=0;
    cudaStat5 = cudaMalloc((void**)&cu_OutMat_COO, num_rows * num_cols_b * sizeof(float));
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat4 = cudaMemcpy(cu_InMat_COO, InMat, num_cols * num_cols_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(cu_OutMat_COO, OutMat, num_rows * num_cols_b * sizeof(float), cudaMemcpyHostToDevice);
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // ELL mem copy
    int* cu_ell_col_ind;
    float* cu_ell_values;
    float* cu_InMat_ELL=0;
    float* cu_OutMat_ELL=0;
    CHECK_CUDA( cudaMalloc((void**) &cu_ell_col_ind, ell_num_blocks * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &cu_ell_values,
                                    ell_cols * ell_num_rows * sizeof(float)) )
    CHECK_CUDA( cudaMemcpy(cu_ell_col_ind, ell_col_ind,
                           ell_num_blocks * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(cu_ell_values, ell_values,
                           ell_cols * ell_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    
    CHECK_CUDA( cudaMalloc((void**)&cu_InMat_ELL, ell_num_cols * num_cols_b * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**)&cu_OutMat_ELL, ell_num_rows * num_cols_b * sizeof(float)) )
    CHECK_CUDA( cudaMemcpy(cu_InMat_ELL, InMat, ell_num_cols * num_cols_b * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(cu_OutMat_ELL, OutMat, ell_num_rows * num_cols_b * sizeof(float), cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // COO kernel
    // initialize cusparse library
    cusparseHandle_t handle_COO = NULL;
    cusparseSpMatDescr_t matA_COO;
    cusparseDnMatDescr_t matX_COO, matY_COO;
    void* dBuffer_COO = NULL;
    size_t bufferSize_COO = 0;
    
    CHECK_CUSPARSE( cusparseCreate(&handle_COO) )
    // Create sparse matrix
    CHECK_CUSPARSE( cusparseCreateCoo(&matA_COO, num_rows, num_cols, nnz,
                                      cu_cooRow, cu_cooCol, cu_cooVal,
                                      CUSPARSE_INDEX_32I, 
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F) )
    
    // Create dense vector input
    CHECK_CUSPARSE( cusparseCreateDnMat(&matX_COO, num_cols, num_cols_b, num_cols, cu_InMat_COO, CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense vector output
    CHECK_CUSPARSE( cusparseCreateDnMat(&matY_COO, num_rows, num_cols_b, num_rows, cu_OutMat_COO, CUDA_R_32F, CUSPARSE_ORDER_COL) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_COO) )

    CHECK_CUDA( cudaMalloc(&dBuffer_COO, bufferSize_COO) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMM(handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE,
			         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_COO) )
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // ELL kernel
    cusparseHandle_t     handle_ELL = NULL;
    cusparseSpMatDescr_t matA_ELL;
    cusparseDnMatDescr_t matX_ELL, matY_ELL;
    void* dBuffer_ELL = NULL;
    size_t bufferSize_ELL = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle_ELL) )
    // Create sparse matrix in blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                      &matA_ELL,
                                      ell_num_rows, ell_num_cols, ell_blocksize,
                                      ell_cols, cu_ell_col_ind, cu_ell_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_16F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matX_ELL, ell_num_cols, num_cols_b, ell_num_cols, cu_InMat_ELL,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matY_ELL, ell_num_rows, num_cols_b, ell_num_rows, cu_OutMat_ELL,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle_ELL,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &ell_alpha, matA_ELL, matX_ELL, &ell_beta, matY_ELL, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_ELL) )
    CHECK_CUDA( cudaMalloc(&dBuffer_ELL, bufferSize_ELL) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle_ELL,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &ell_alpha, matA_ELL, matX_ELL, &ell_beta, matY_ELL, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_ELL) )
    
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // element-wise add
    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));
    float* cu_OutMat=0;
    float alpha = 1.0;
    float gamma = 1.0;
    std::vector<int> modeIn1{'a','b'};
    std::vector<int> modeIn2{'a','b'};
    std::vector<int> modeOut{'a','b'};
    int nmodeIn1 = modeIn1.size();
    int nmodeIn2 = modeIn2.size();
    int nmodeOut = modeOut.size();
    std::unordered_map<int, int64_t> extent;
    extent['a'] = num_rows;
    extent['b'] = num_cols_b;
    std::vector<int64_t> extentIn1;
    for (auto mode : modeIn1)
        extentIn1.push_back(extent[mode]);
    std::vector<int64_t> extentIn2;
    for (auto mode : modeIn2)
        extentIn2.push_back(extent[mode]);
    std::vector<int64_t> extentOut;
    for (auto mode : modeOut)
        extentOut.push_back(extent[mode]);
    cutensorTensorDescriptor_t descIn1;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descIn1,
                 nmodeIn1,
                 extentIn1.data(),
                 NULL /* stride */,
                 CUDA_R_32F, CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descIn2;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descIn2,
                 nmodeIn2,
                 extentIn2.data(),
                 NULL /* stride */,
                 CUDA_R_32F, CUTENSOR_OP_IDENTITY));
    cutensorTensorDescriptor_t descOut;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descOut,
                 nmodeOut,
                 extentOut.data(),
                 NULL /* stride */,
                 CUDA_R_32F, CUTENSOR_OP_IDENTITY));
    CHECK_CUDA( cudaMalloc((void**)&cu_OutMat, num_rows * num_cols_b * sizeof(float)) )
    CHECK_CUDA( cudaMemcpy(cu_OutMat, OutMat, num_rows * num_cols_b * sizeof(float), cudaMemcpyHostToDevice) )
    cutensorElementwiseBinary(&handle,
                (void*)&alpha, cu_OutMat_COO, &descIn1, modeIn1.data(),
                (void*)&gamma, cu_OutMat_ELL, &descIn2, modeIn2.data(),
                               cu_OutMat, &descOut, modeOut.data(),
                CUTENSOR_OP_ADD, CUDA_R_32F, 0 /* stream */);

    //--------------------------------------------------------------------------
    // time execution
    int num_runs = 1000;
    float elapsed_time_ms = 0.0;
    cudaEvent_t start, stop;
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        CHECK_CUSPARSE( cusparseSpMM(handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_COO) )
        CHECK_CUSPARSE( cusparseSpMM(handle_ELL,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &ell_alpha, matA_ELL, matX_ELL, &ell_beta, matY_ELL, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_ELL) )
        cutensorElementwiseBinary(&handle,
                (void*)&alpha, cu_OutMat_COO, &descIn1, modeIn1.data(),
                (void*)&gamma, cu_OutMat_ELL, &descIn2, modeIn2.data(),
                               cu_OutMat, &descOut, modeOut.data(),
                CUTENSOR_OP_ADD, CUDA_R_32F, 0 /* stream */);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "Total time = " << elapsed_time_ms / 1000 << "s" << std::endl;
    std::cout << "average_time = " << elapsed_time_ms / num_runs << " ms" << std::endl;
    std::cout << "COO nnz is " << nnz << " and num_cols is " << num_cols << std::endl;
    std::cout << "ELL nnz is " << ell_nnz << std::endl;
    double throughput = double(nnz+ell_cols * ell_num_rows) * double(2 * num_cols_b * num_runs) / double(elapsed_time_ms) / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // COO dealloc
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_COO) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matX_COO) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matY_COO) )
    CHECK_CUSPARSE( cusparseDestroy(handle_COO) )

    // free memory
    cudaFree(dBuffer_COO);
    cudaFree(cu_cooVal);
    cudaFree(cu_cooCol);
    cudaFree(cu_cooRow);
    cudaFree(cu_InMat_COO);
    cudaFree(cu_OutMat_COO);
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // ELL dealloc
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_ELL) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matX_ELL) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matY_ELL) )
    CHECK_CUSPARSE( cusparseDestroy(handle_ELL) )
    //--------------------------------------------------------------------------
    // device result check
    // CHECK_CUDA( cudaMemcpy(OutMat, cu_OutMat_ELL, ell_num_rows * num_cols_b * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         float c_value  = static_cast<float>(OutMat[i + j * ldc]);
    //         float c_result = static_cast<float>(hC_result[i + j * ldc]);
    //         if (c_value != c_result) {
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     std::printf("spmm_blockedell_example test PASSED\n");
    // else
    //     std::printf("spmm_blockedell_example test FAILED: wrong result\n");
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer_ELL) )
    CHECK_CUDA( cudaFree(cu_ell_col_ind) )
    CHECK_CUDA( cudaFree(cu_ell_values) )
    CHECK_CUDA( cudaFree(cu_InMat_ELL) )
    CHECK_CUDA( cudaFree(cu_OutMat_ELL) )
    //--------------------------------------------------------------------------

    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(0);
    std::string dataset1 = argv[1];
    std::string dataset2 = argv[2];
    benchmark_spmv_hybrid_ell_coo(dataset1, dataset2);
}
