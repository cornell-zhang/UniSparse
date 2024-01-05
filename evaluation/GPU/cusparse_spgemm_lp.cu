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

int benchmark_spmv_csr(std::string dataset, std::string dataset1) {
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

    cnpy::npz_t npz1 = cnpy::npz_load(dataset1);
    cnpy::NpyArray npy_shape1 = npz1["shape"];
    cnpy::NpyArray npy_data1 = npz1["data"];
    cnpy::NpyArray npy_indptr1 = npz1["indptr"];
    cnpy::NpyArray npy_indices1 = npz1["indices"];
    int nnz1 = npy_data1.shape[0];
    int num_rows1 = npy_shape1.data<int>()[0];
    int num_cols1 = npy_shape1.data<int>()[2];
    float* csrVal1 = npy_data1.data<float>();
    int* csrRowPtr1 = npy_indptr1.data<int>();
    int* csrColInd1 = npy_indices1.data<int>();

    float alpha = 1.0;
    float beta = 0.0;

    // std::cout << "nnz:" << nnz << std::endl;
    // std::cout << "num_rows:" << num_rows << std::endl;
    // std::cout << "num_cols:" << num_cols << std::endl;

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5, cudaStat6;

    // device malloc
    float* cu_csrVal1=NULL;
    cudaStat1 = cudaMalloc((void**)&cu_csrVal1, nnz * sizeof(float));
    int* cu_csrRowPtr1=NULL;
    cudaStat2 = cudaMalloc((void**)&cu_csrRowPtr1, (num_rows + 1) * sizeof(int));
    int* cu_csrColInd1=NULL;
    cudaStat3 = cudaMalloc((void**)&cu_csrColInd1, nnz * sizeof(int));
    float* cu_csrVal2=NULL;
    cudaStat4 = cudaMalloc((void**)&cu_csrVal2, nnz1 * sizeof(float));
    int* cu_csrRowPtr2=NULL;
    cudaStat5 = cudaMalloc((void**)&cu_csrRowPtr2, (num_cols + 1) * sizeof(int));
    int* cu_csrColInd2=NULL;
    cudaStat6 = cudaMalloc((void**)&cu_csrColInd2, nnz1 * sizeof(int));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
	(cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat1 = cudaMemcpy(cu_csrVal1, csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cu_csrRowPtr1, csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cu_csrColInd1, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(cu_csrVal2, csrVal1, nnz1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat5 = cudaMemcpy(cu_csrRowPtr2, csrRowPtr1, (num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaStat6 = cudaMemcpy(cu_csrColInd2, csrColInd1, nnz1 * sizeof(int), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess) ||
	(cudaStat4 != cudaSuccess) ||
        (cudaStat5 != cudaSuccess) ||
        (cudaStat6 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    // initialize cusparse library
    cusparseSpMatDescr_t matA, matB;
    
    // Create sparse matrix
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                      cu_csrRowPtr1, cu_csrColInd1, cu_csrVal1,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_cols, num_cols1, nnz1,
                                      cu_csrRowPtr2, cu_csrColInd2, cu_csrVal2,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )


    int runtime = VAR;
    float elapsed_time_preprocessing_ms = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < runtime; i++) {
        cusparseHandle_t handle = NULL;
        void* dBuffer1 = NULL;
        void* dBuffer2 = NULL;
        size_t bufferSize1 = 0;
        size_t bufferSize2 = 0;
        CHECK_CUSPARSE( cusparseCreate(&handle) )
        cusparseSpMatDescr_t matC;
        CHECK_CUSPARSE( cusparseCreateCsr(&matC, num_rows, num_cols1, 0,
                                        NULL, NULL, NULL,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
        
        // SpGEMM Computation
        cusparseSpGEMMDescr_t spgemmDesc;
        
        
        CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )
        CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, NULL) )
        CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )

        CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize1, dBuffer1) )

        CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemmDesc, &bufferSize2, NULL) )
        CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )
    
        // compute the intermediate product of A * B
        CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, matB, &beta, matC,
                                            CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc, &bufferSize2, dBuffer2) )

        
        int64_t C_num_rows1, C_num_cols1, C_nnz1;
        int* cu_csrRowPtr3 = NULL;
        int* cu_csrColInd3 = NULL;
        float* cu_csrVal3 = NULL;
        CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                            &C_nnz1) )
        // allocate matrix 
        CHECK_CUDA( cudaMalloc((void**) &cu_csrRowPtr3, (num_rows + 1) * sizeof(int))   )
        CHECK_CUDA( cudaMalloc((void**) &cu_csrColInd3, C_nnz1 * sizeof(int))   )
        CHECK_CUDA( cudaMalloc((void**) &cu_csrVal3,  C_nnz1 * sizeof(float)) )

        // update matC with the new pointers
        CHECK_CUSPARSE( cusparseCsrSetPointers(matC, cu_csrRowPtr3, cu_csrColInd3, cu_csrVal3) )
        // copy the final products to the matrix C
        CHECK_CUSPARSE( cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC,
                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

        CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
        CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
        CHECK_CUSPARSE( cusparseDestroy(handle) )
        CHECK_CUDA( cudaFree(dBuffer1) )
        CHECK_CUDA( cudaFree(dBuffer2) )
        CHECK_CUDA( cudaFree(cu_csrVal3) )
        CHECK_CUDA( cudaFree(cu_csrColInd3) )
        CHECK_CUDA( cudaFree(cu_csrRowPtr3) )
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_preprocessing_ms, start, stop);
    std::cout << "total_time = " << elapsed_time_preprocessing_ms / 1000 << " s" << std::endl;
    
    // free memory
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUDA( cudaFree(cu_csrVal1) )
    CHECK_CUDA( cudaFree(cu_csrColInd1) )
    CHECK_CUDA( cudaFree(cu_csrRowPtr1) )
    CHECK_CUDA( cudaFree(cu_csrVal2) )
    CHECK_CUDA( cudaFree(cu_csrColInd2) )
    CHECK_CUDA( cudaFree(cu_csrRowPtr2) )
    
    return 0;
}


int main(int argc, char** argv) {
    std::string sp_matrix = argv[1];
    std::string sp_vector = argv[2];
    benchmark_spmv_csr(sp_matrix, sp_vector);
}
