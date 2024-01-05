#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <cnpy.h>


int test_spmm_csr(cusparseHandle_t handle,
                  cusparseMatDescr_t descr,
                  int* cu_csrRowPtr,
                  int* cu_csrColInd,
                  float* cu_csrVal,
                  int num_rows,
                  int num_cols,
                  int nnz,
                  int feat_len,
                  int num_runs) {
    // seems dense matrix (FeatMatrix and OutMatrix) is treated as column-major in cusparseScsrmm2; check again
    float* FeatMatrix = (float*)malloc(sizeof(float) * num_cols * feat_len);
    for (int i = 0; i < num_cols * feat_len; i++) {
        FeatMatrix[i] = 1.0;
    }
    float* OutMatrix = (float*)malloc(sizeof(float) * num_cols * feat_len);
    for (int i = 0; i < num_cols * feat_len; i++) {
        OutMatrix[i] = 2.0;
    }

    cudaError_t cudaStat1, cudaStat2;

    // device malloc
    float* cu_FeatMatrix=0;
    cudaStat1 = cudaMalloc((void**)&cu_FeatMatrix, num_cols * feat_len * sizeof(float));
    float* cu_OutMatrix=0;
    cudaStat2 = cudaMalloc((void**)&cu_OutMatrix, num_rows * feat_len * sizeof(float));
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }

    // memcpy from host to device
    cudaStat1 = cudaMemcpy(cu_FeatMatrix, FeatMatrix, num_cols * feat_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cu_OutMatrix, OutMatrix, num_rows * feat_len * sizeof(float), cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    float alpha = 1.0;
    float beta = 0.0;

    cusparseStatus_t status;

    // warm up run
    status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        num_rows, feat_len, num_cols, nnz, &alpha, descr, cu_csrVal, cu_csrRowPtr, cu_csrColInd,
        cu_FeatMatrix, feat_len, &beta, cu_OutMatrix, num_rows);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("cusparseSPMM failed\n");
        exit(-1);
    }
    cudaDeviceSynchronize();

    // measure time
    float elapsed_time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        status= cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            num_rows, feat_len, num_cols, nnz, &alpha, descr, cu_csrVal, cu_csrRowPtr, cu_csrColInd,
            cu_FeatMatrix, feat_len, &beta, cu_OutMatrix, num_rows);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    std::cout << "average time of " << num_runs << " runs: " << elapsed_time / num_runs << " ms" << std::endl;

    // cudaMemcpy(OutMatrix, cu_OutMatrix, num_nodes * feat_len * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 8; i++) {
    //     std::cout << OutMatrix[i] << std::endl;
    // }

    // clean up
    free(FeatMatrix);
    free(OutMatrix);
    cudaFree(cu_FeatMatrix);
    cudaFree(cu_OutMatrix);

    return 0;
}


int benchmark_spmm_csr() {
    std::string file_name = "/work/shared/users/phd/yh457/data/sparse_matrix_graph/uniform_100K_100_csr_float32.npz";

    // load csr matrix
    cnpy::npz_t npz = cnpy::npz_load(file_name);
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

    std::vector<int> feat_len_values{1, 4, 32, 64, 128, 256, 512};
    int num_runs = 10;
    for (int feat_len : feat_len_values) {
        std::cout << "\nfeat_len is: " << feat_len << std::endl;
        test_spmm_csr(handle, descr, cu_csrRowPtr, cu_csrColInd, cu_csrVal, num_rows, num_cols, nnz, feat_len, num_runs);
    }

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


int main () {
    benchmark_spmm_csr();
}
