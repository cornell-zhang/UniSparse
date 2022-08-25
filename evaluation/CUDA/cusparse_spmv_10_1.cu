#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <cnpy.h>


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
    float* InVec = (float*)malloc(sizeof(float) * num_cols);
    for (int i = 0; i < num_cols; i++) {
        InVec[i] = 1.0;
    }
    float* OutVec = (float*)malloc(sizeof(float) * num_rows);
    for (int i = 0; i < num_rows; i++) {
        OutVec[i] = 0.0;
    }

    cudaError_t cudaStat1, cudaStat2;

    // device malloc
    float* cu_InVec=0;
    cudaStat1 = cudaMalloc((void**)&cu_InVec, num_cols * sizeof(float));
    float* cu_OutVec=0;
    cudaStat2 = cudaMalloc((void**)&cu_OutVec, num_rows * sizeof(float));
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat1 = cudaMemcpy(cu_InVec, InVec, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cu_OutVec, OutVec, num_rows * sizeof(float), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

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

    // std::cout << "nnz:" << nnz << std::endl;
    // std::cout << "num_rows:" << num_rows << std::endl;
    // std::cout << "num_cols:" << num_cols << std::endl;

    cudaError_t cudaStat1, cudaStat2, cudaStat3;

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

    cusparseStatus_t status;

    // initialize cusparse library
    cusparseHandle_t handle=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library initialization failed\n");
        exit(-1);
    }

    // create and setup matrix descriptor
    cusparseMatDescr_t descr=0;
    status= cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Matrix descriptor initialization failed\n");
        exit(-1);
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int num_runs = 10;
    float average_time_in_sec = test_spmv_csr(handle, descr, cu_csrRowPtr, cu_csrColInd, cu_csrVal,
                                              num_rows, num_cols, nnz, num_runs);
    std::cout << "average_time = " << average_time_in_sec * 1000 << " ms" << std::endl;
    float throughput = nnz / average_time_in_sec / 1000 / 1000 / 1000;
    std::cout << "THROUGHPUT = " << throughput << " GOPS" << std::endl;

    // destroy handle
    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library release of resources failed\n");
        exit(-1);
    }

    // free memory
    cudaFree(cu_csrVal);
    cudaFree(cu_csrColInd);
    cudaFree(cu_csrRowPtr);

    return 0;
}


int main(int argc, char** argv) {
    std::string dataset = argv[1];
    benchmark_spmv_csr(dataset);
}
