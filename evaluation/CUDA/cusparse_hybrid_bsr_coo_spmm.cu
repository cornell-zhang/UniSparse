#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <cnpy.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <unordered_map>

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

int benchmark_spmv_bsr(std::string dataset1, std::string dataset2) {
    // load coo matrix
    cnpy::npz_t npz_coo = cnpy::npz_load(dataset1);
    cnpy::NpyArray coo_npy_shape = npz_coo["shape"];
    cnpy::NpyArray coo_npy_data = npz_coo["data"];
    cnpy::NpyArray coo_npy_row = npz_coo["row"];
    cnpy::NpyArray coo_npy_col = npz_coo["col"];
    int coo_nnz = coo_npy_data.shape[0];
    int coo_num_rows = coo_npy_shape.data<int>()[0];
    int coo_num_cols = coo_npy_shape.data<int>()[2];
    float* cooVal = coo_npy_data.data<float>();
    int* cooRow = coo_npy_row.data<int>();
    int* cooCol = coo_npy_col.data<int>();
    float coo_alpha = 1.0;
    float coo_beta = 0.0;

    // load csr matrix
    cnpy::npz_t npz = cnpy::npz_load(dataset2);
    cnpy::NpyArray npy_shape = npz["shape"];
    cnpy::NpyArray npy_data = npz["data"];
    cnpy::NpyArray npy_indptr = npz["indptr"];
    cnpy::NpyArray npy_indices = npz["indices"];
    int nnz = npy_data.shape[0];
    int num_rows = npy_shape.data<int>()[0];
    int num_cols = npy_shape.data<int>()[2];
    float* csrVal = npy_data.data<float>();
    int* csrRowPtr = npy_indptr.data<int>();
    int* csrColInd = npy_indices.data<int>();
    float bsr_alpha = 1.0;
    float bsr_beta = 0.0;
    int blockDim = 32;
    int mb = (num_rows + blockDim-1)/blockDim;
    int kb = (num_cols + blockDim-1)/blockDim;
    const int m = mb*blockDim;
    const int k = kb*blockDim;
    const int ldb = k; // leading dimension of B
    const int ldc = m; // leading dimension of C
    int num_cols_b = 1000;

    std::cout << "bsr_nnz:" << nnz << std::endl;
    std::cout << "bsr_num_rows:" << num_rows << std::endl;
    std::cout << "bsr_num_cols:" << num_cols << std::endl;
    std::cout << "coo_nnz:" << coo_nnz << std::endl;
    std::cout << "ldb:" << ldb << std::endl;
    std::cout << "ldc:" << ldc << std::endl;
    std::cout << "coo_num_rows:" << coo_num_rows << std::endl;
    std::cout << "coo_num_cols:" << coo_num_cols << std::endl;

    //--------------------------------------------------------------------------
    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
    // BSR device malloc
    float* cu_csrVal=0;
    cudaStat1 = cudaMalloc((void**)&cu_csrVal, nnz * sizeof(float));
    int* cu_csrRowPtr=0;
    cudaStat2 = cudaMalloc((void**)&cu_csrRowPtr, (num_rows + 1) * sizeof(int));
    int* cu_csrColInd=0;
    cudaStat3 = cudaMalloc((void**)&cu_csrColInd, nnz * sizeof(int));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat1 = cudaMemcpy(cu_csrVal, csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cu_csrRowPtr, csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cu_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    //--------------------------------------------------------------------------
    // COOdevice malloc
    float* cu_cooVal=0;
    CHECK_CUDA( cudaMalloc((void**)&cu_cooVal, coo_nnz * sizeof(float)))
    int* cu_cooRow=0;
    CHECK_CUDA( cudaMalloc((void**)&cu_cooRow, coo_nnz * sizeof(int)))
    int* cu_cooCol=0;
    CHECK_CUDA( cudaMalloc((void**)&cu_cooCol, coo_nnz * sizeof(int)))

    // memcpy from host to device
    CHECK_CUDA( cudaMemcpy(cu_cooVal, cooVal, coo_nnz * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA( cudaMemcpy(cu_cooRow, cooRow, coo_nnz * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA( cudaMemcpy(cu_cooCol, cooCol, coo_nnz * sizeof(int), cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------


    float* InMat = (float*)malloc(sizeof(float) * ldb * num_cols_b);
    for (int i = 0; i < ldb * num_cols_b; i++) {
        InMat[i] = 1.0;
    }
    float* OutMat = (float*)malloc(sizeof(float) * ldc * num_cols_b);
    for (int i = 0; i < ldc * num_cols_b; i++) {
        OutMat[i] = 0.0;
    }

    //--------------------------------------------------------------------------
    // BSR mem copy
    float* cu_InMat_BSR=0;
    cudaStat4 = cudaMalloc((void**)&cu_InMat_BSR, ldb * num_cols_b * sizeof(float));
    float* cu_OutMat_BSR=0;
    cudaStat5 = cudaMalloc((void**)&cu_OutMat_BSR, ldc * num_cols_b * sizeof(float));

    // memcpy from host to device
    cudaStat4 = cudaMemcpy(cu_InMat_BSR, InMat, ldb * num_cols_b * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(cu_OutMat_BSR, OutMat, ldc * num_cols_b * sizeof(float), cudaMemcpyHostToDevice);
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }
    //--------------------------------------------------------------------------
    // COO mem copy
    float* cu_InMat_COO=0;
    CHECK_CUDA( cudaMalloc((void**)&cu_InMat_COO, ldb * num_cols_b * sizeof(float)))
    float* cu_OutMat_COO=0;
    CHECK_CUDA( cudaMalloc((void**)&cu_OutMat_COO, ldc * num_cols_b * sizeof(float)))

    // memcpy from host to device
    CHECK_CUDA( cudaMemcpy(cu_InMat_COO, InMat, ldb * num_cols_b * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA( cudaMemcpy(cu_OutMat_COO, OutMat, ldc * num_cols_b * sizeof(float), cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------


    //--------------------------------------------------------------------------
    // BSR kernel
    // initialize cusparse library
    cusparseHandle_t handle_BSR = NULL;
    // cusparseSpMatDescr_t descrA;
    cusparseMatDescr_t descrA, descrC;
    // cusparseDnMatDescr_t matX, matY;
    
    // void* dBuffer = NULL;
    // size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle_BSR) )
    // Create sparse matrix
    CHECK_CUSPARSE( cusparseCreateMatDescr(&descrA) )
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
    CHECK_CUSPARSE( cusparseCreateMatDescr(&descrC) )
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    // cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE);
    int base;
    int nnzb=0;
    int* cu_bsrRowPtr = 0;
    int* cu_bsrColInd = 0;
    float* cu_bsrVal=0;
    int *nnzTotalDevHostPtr = &nnzb;
    // cudaMalloc((void**)&cu_bsrRowPtr, sizeof(int) *(mb+1));
    // CHECK_CUSPARSE( cusparseXcsr2bsrNnz(handle_BSR, dir, num_rows, num_cols,
    //                     descrA, csrRowPtr, csrColInd,
    //                     blockDim,
    //                     descrC, cu_bsrRowPtr,
    //                     &nnzb) )
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    // int mb = (num_rows + blockDim-1)/blockDim;
    // int nb = (num_rows + blockDim-1)/blockDim;
    cudaMalloc((void**)&cu_bsrRowPtr, sizeof(int) *(mb+1));
    std::cout << "before BSR nnzb" << std::endl;
    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle_BSR, dir, num_rows, num_cols,
            descrA, cu_csrRowPtr, cu_csrColInd, blockDim,
            descrC, cu_bsrRowPtr, &nnzb))
    if (NULL != nnzTotalDevHostPtr){
        nnzb = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&nnzb, cu_bsrRowPtr+mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, cu_bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }
    std::cout << "nnzb=" << nnzb <<std::endl;
    cudaMalloc((void**)&cu_bsrColInd, sizeof(int)*nnzb);
    cudaMalloc((void**)&cu_bsrVal, sizeof(float)*(blockDim*blockDim)*nnzb);

    std::cout << "before CSR 2 BSR" << std::endl;
    CHECK_CUSPARSE( cusparseScsr2bsr(handle_BSR,
                                    dir,
                                    num_rows,
                                    num_cols,
                                    descrA,
                                    cu_csrVal,
                                    cu_csrRowPtr,
                                    cu_csrColInd,
                                    blockDim,
                                    descrC,
                                    cu_bsrVal,
                                    cu_bsrRowPtr,
                                    cu_bsrColInd) )

    std::cout << "before BSR SpMM" << std::endl;
    CHECK_CUSPARSE( cusparseSbsrmm( handle_BSR,
                                    dir,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    mb, num_cols_b, kb, nnzb,
                                    &bsr_alpha, descrC, cu_bsrVal,
                                    cu_bsrRowPtr, cu_bsrColInd,
                                    blockDim, cu_InMat_BSR,
                                    ldb, &bsr_beta, cu_OutMat_BSR, ldc) )
    std::cout << "finish BSR SpMM" << std::endl;
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // COO kernel
    cusparseHandle_t handle_COO = NULL;
    cusparseSpMatDescr_t matA_COO;
    cusparseDnMatDescr_t matX_COO, matY_COO;
    void* dBuffer_COO = NULL;
    size_t bufferSize_COO = 0;
    
    CHECK_CUSPARSE( cusparseCreate(&handle_COO) )
    // Create sparse matrix
    CHECK_CUSPARSE( cusparseCreateCoo(&matA_COO, coo_num_rows, coo_num_cols, coo_nnz,
                                      cu_cooRow, cu_cooCol, cu_cooVal,
                                      CUSPARSE_INDEX_32I, 
                                      CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F) )
    
    // Create dense vector input
    CHECK_CUSPARSE( cusparseCreateDnMat(&matX_COO, coo_num_cols, num_cols_b, coo_num_cols, cu_InMat_COO, CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense vector output
    CHECK_CUSPARSE( cusparseCreateDnMat(&matY_COO, coo_num_rows, num_cols_b, coo_num_rows, cu_OutMat_COO, CUDA_R_32F, CUSPARSE_ORDER_COL) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize_COO) )

    CHECK_CUDA( cudaMalloc(&dBuffer_COO, bufferSize_COO) )
    std::cout << "before COO SpMM" << std::endl;
    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMM(handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE,
			         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_COO) )
    std::cout << "finish COO SpMM" << std::endl;
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
    extent['a'] = coo_num_rows;
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
    CHECK_CUDA( cudaMalloc((void**)&cu_OutMat, coo_num_rows * num_cols_b * sizeof(float)) )
    // CHECK_CUDA( cudaMemcpy(cu_OutMat, OutMat, coo_num_rows * num_cols_b * sizeof(float), cudaMemcpyHostToDevice) )
    std::cout << "before element add" << std::endl;
    cutensorElementwiseBinary(&handle,
                (void*)&alpha, cu_OutMat_COO, &descIn1, modeIn1.data(),
                (void*)&gamma, cu_OutMat_BSR, &descIn2, modeIn2.data(),
                               cu_OutMat, &descOut, modeOut.data(),
                CUTENSOR_OP_ADD, CUDA_R_32F, 0 /* stream */);
    std::cout << "finish element add" << std::endl;

    //--------------------------------------------------------------------------
    
    cudaDeviceSynchronize();

    int num_runs = 10;
    float elapsed_time_ms = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        std::cout << "iteration " << i << std::endl;
        CHECK_CUSPARSE( cusparseSpMM(handle_COO, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &coo_alpha, matA_COO, matX_COO, &coo_beta, matY_COO, CUDA_R_32F,
                                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer_COO) )
        CHECK_CUSPARSE( cusparseSbsrmm( handle_BSR,
                                    dir,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    mb, num_cols_b, kb, nnzb,
                                    &bsr_alpha, descrC,
                                    cu_bsrVal, cu_bsrRowPtr, cu_bsrColInd,
                                    blockDim,
                                    cu_InMat_BSR,
                                    ldb, &bsr_beta, cu_OutMat_BSR, ldc) )
        cutensorElementwiseBinary(&handle,
                (void*)&alpha, cu_OutMat_COO, &descIn1, modeIn1.data(),
                (void*)&gamma, cu_OutMat_BSR, &descIn2, modeIn2.data(),
                               cu_OutMat, &descOut, modeOut.data(),
                CUTENSOR_OP_ADD, CUDA_R_32F, 0 /* stream */);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "Total time = " << elapsed_time_ms / 1000 << "s" << std::endl;
    std::cout << "average_time = " << elapsed_time_ms / num_runs << " ms" << std::endl;
    std::cout << "nnz is " << nnz << " and num_cols is " << num_cols << std::endl;
    double throughput = double(nnz) * double(2 * num_cols_b * num_runs) / double(elapsed_time_ms) / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    // destroy matrix/vector descriptors
    // CHECK_CUSPARSE( cusparseDestroySpMat(descrA) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrA) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrC) )
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matX) )
    // CHECK_CUSPARSE( cusparseDestroyDnMat(matY) )
    CHECK_CUSPARSE( cusparseDestroy(handle_BSR) )

    CHECK_CUSPARSE( cusparseDestroySpMat(matA_COO) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matX_COO) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matY_COO) )
    CHECK_CUSPARSE( cusparseDestroy(handle_COO) )

    // free memory
    cudaFree(cu_bsrVal);
    cudaFree(cu_bsrColInd);
    cudaFree(cu_bsrRowPtr);
    cudaFree(cu_InMat_BSR);
    cudaFree(cu_OutMat_BSR);

    // free memory
    cudaFree(dBuffer_COO);
    cudaFree(cu_cooVal);
    cudaFree(cu_cooCol);
    cudaFree(cu_cooRow);
    cudaFree(cu_InMat_COO);
    cudaFree(cu_OutMat_COO);


    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(0);
    std::string dataset1 = argv[1];
    std::string dataset2 = argv[2];
    benchmark_spmv_bsr(dataset1, dataset2);
}
