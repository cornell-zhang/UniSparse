#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <cnpy.h>

/*
float test_spmv_csr(cusparseHandle_t handle,
                    cusparseMatDescr_t descr,
                    int* cu_csrRowPtr,
                    int* cu_csrColInd,
                    float* cu_csrVal,
                    int num_rows,
                    int num_cols,
                    int nnz,
                    int num_runs) {
    // seems dense matrix (InVec and OutVec) is treated as column-major in cusparseScsrmm2; check again

    float alpha = 1.0;
    float beta = 0.0;

    // warm up run
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        num_rows, num_cols, nnz, &alpha, descr, cu_csrVal, cu_csrRowPtr, cu_csrColInd,
        cu_InVec, &beta, cu_OutVec);
    cudaDeviceSynchronize();

    // measure time
    float elapsed_time_ms = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            num_rows, num_cols, nnz, &alpha, descr, cu_csrVal, cu_csrRowPtr, cu_csrColInd,
            cu_InVec, &beta, cu_OutVec);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // cudaMemcpy(OutVec, cu_OutVec, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 8; i++) {
    //     std::cout << OutVec[i] << std::endl;
    // }

    // clean up
    free(InVec);
    free(OutVec);
    cudaFree(cu_InVec);
    cudaFree(cu_OutVec);

    return elapsed_time_ms / num_runs / 1000;
}
*/

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

int benchmark_spmv_csr(std::string dataset) {
    // load csr matrix
    cnpy::npz_t npz = cnpy::npz_load(dataset);
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
    float alpha = 1.0;
    float beta = 0.0;

    // std::cout << "nnz:" << nnz << std::endl;
    // std::cout << "num_rows:" << num_rows << std::endl;
    // std::cout << "num_cols:" << num_cols << std::endl;

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;

    // device malloc
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

    float* InVec = (float*)malloc(sizeof(float) * num_cols);
    for (int i = 0; i < num_cols; i++) {
        InVec[i] = 1.0;
    }
    float* OutVec = (float*)malloc(sizeof(float) * num_rows);
    for (int i = 0; i < num_rows; i++) {
        OutVec[i] = 0.0;
    }

    // device malloc
    float* cu_InVec=0;
    cudaStat4 = cudaMalloc((void**)&cu_InVec, num_cols * sizeof(float));
    float* cu_OutVec=0;
    cudaStat5 = cudaMalloc((void**)&cu_OutVec, num_rows * sizeof(float));
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat4 = cudaMemcpy(cu_InVec, InVec, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(cu_OutVec, OutVec, num_rows * sizeof(float), cudaMemcpyHostToDevice);
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    // initialize cusparse library
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                      cu_csrRowPtr, cu_csrColInd, cu_csrVal,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector input
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, num_cols, cu_InVec, CUDA_R_32F) )
    // Create dense vector output
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, num_rows, cu_OutVec, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )

    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
    cudaDeviceSynchronize();

    int num_runs = VAR;
    float elapsed_time_ms = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                     CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "Total time = " << elapsed_time_ms / 1000 << " s" << std::endl;
    std::cout << "average_time = " << elapsed_time_ms / num_runs << " ms" << std::endl;
    std::cout << "nnz is " << nnz << std::endl;
    double throughput = double(nnz) * double(2 * num_runs) / double(elapsed_time_ms) / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // free memory
    cudaFree(cu_csrVal);
    cudaFree(cu_csrColInd);
    cudaFree(cu_csrRowPtr);
    cudaFree(cu_InVec);
    cudaFree(cu_OutVec);


    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(6);
    std::string dataset = argv[1];
    benchmark_spmv_csr(dataset);
}
