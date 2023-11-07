void _mlir_ciface_calculateCSRSpMV(StridedMemRefType<double, 1> *out, 
                                      StridedMemRefType<uint64_t, 1> *ptr, 
                                      StridedMemRefType<uint64_t, 1> *col, 
                                      StridedMemRefType<double, 1> *value, 
                                      StridedMemRefType<double, 1> *input) {
    uint64_t row = ptr->sizes[0] - 1;
    double *result = new double[row];
//      printf("row size is: %d\n", row);
    for(uint64_t i = 0; i < row; i++) {
      double temp = 0;
      for(uint64_t j = ptr->data[i]; j < ptr->data[i+1]; j++) {
      temp += value->data[j] * input->data[col->data[j]];
  //	  printf("value->data[%d] is: %f, col->data[%d] is: %d, input->data[%d] is: %f\n", j, value->data[j], j, col->data[j], col->data[j], input->data[col->data[j]]);
      }
      result[i] = temp;
//        printf("outdata[%d] is %f\n", i, out->data[i]);
    }
    out->data = result;
    out->basePtr = result;
    out->offset = 0;  
    out->strides[0] = 1;
  }  

// Vanilla DIA SpMM
    void _mlir_ciface_kernel_dia_spmm(StridedMemRefType<float, 2> *outC,
                                      void* inA, 
                                      StridedMemRefType<float, 2> *inB, 
                                      StridedMemRefType<float, 2> *inC) {
        printf("enter in kernel_dia_spmm\n");
        UniSparseStorage* spA = (UniSparseStorage*)inA;
        // printf("spA->vLevel.size = %zu \n", spA->vLevel.size());
        std::shared_ptr<LevelStorage> spA_dim0 = spA->vLevel[0];
        std::shared_ptr<LevelStorage> spA_dim1 = spA->vLevel[1];
        // std::vector<float> spA_data = spA->valueArray;
        std::vector< std::vector<float> > spA_vector = spA->vectorArray;
        // printf("spA_data.size = %zu\n", spA_data.size());
        // printf("spA_vector.size = %zu\n", spA_vector.size());
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];
        int64_t kSize = inC->sizes[1];
        printf("iSize = %ld, jSize = %ld, kSize = %ld\n", iSize, jSize, kSize);
        // std::vector<int> spA_dim0_crd = spA_dim0->crd;
        std::vector<int> spA_dim1_crd = spA_dim1->crd;
        // // int spA_dim0_size = spA_dim0->size;
        // // int spA_dim1_size = spA_dim1->size;
        // printf("spA_dim0_crd = ");
        // for (auto elm: spA_dim0_crd) {
        //   printf("%d ", elm);
        // }
        // printf("\n");
        // printf("spA_dim1_crd = ");
        // for (auto elm: spA_dim1_crd) {
        //   printf("%d ", elm);
        // }
        // printf("\n");
        // printf("spA_dim0_size = %d, spA_dim1_size = %d \n",spA_dim0_size,spA_dim1_size);
        
        // printf("spA_vector = \n");
        // for (auto v: spA_vector) {
        //   for (auto elm: v) {
        //     printf("%f ", elm);
        //   }
        //   printf("\n");
        // }
        // printf("\n");

        // A*B + C
        outC->basePtr = outC->data = inC->data;
        outC->offset = inC->offset;
        outC->strides[0] = outC->strides[1] = 1;
        outC->sizes[0] = inC->sizes[0];
        outC->sizes[1] = inC->sizes[1];
        // printf("inB_data = \n");
        // for (unsigned j=0; j < jSize; j++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", inB->data[j*kSize+k]);
        //   printf("\n");
        // }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", outC->data[i*iSize+k]);
        //   printf("\n");
        // }
        for (unsigned diag = 0; diag < spA_dim1_crd.size(); diag++) {
          for (int i = 0; i < iSize; i++) {
            int j = spA_dim1_crd[diag] + i;
            if (j >=0 && j < jSize) {
              for (int k = 0; k < kSize; k++) {
                outC->data[i*kSize+k] += spA_vector[diag][i] * inB->data[j*kSize+k];
              }
            }
          }
        }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", outC->data[i*kSize+k]);
        //   printf("\n");
        // }
        // printf("\n");
    }

    void _mlir_ciface_kernel_dia_spmv(StridedMemRefType<float, 1> *outC,
                                      void* inA, 
                                      StridedMemRefType<float, 1> *inB, 
                                      StridedMemRefType<float, 1> *inC) {
        UniSparseStorage* spA = (UniSparseStorage*)inA;
        int32_t* spA_dim0_crd = spA->vLevel[1]->crd.data();
        uint64_t spA_dim0_size = spA->vLevel[1]->crd.size();
        std::vector< std::vector<float> > spA_vector = spA->vectorArray;
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];

        // A*B + C
        outC->basePtr = outC->data = inC->data;
        outC->offset = inC->offset;
        outC->strides[0] = 1;
        outC->sizes[0] = inC->sizes[0];

        uint64_t diag;
        int i, j;
        float sum;
        double start = omp_get_wtime();
        for (unsigned time = 0; time < 10000; time++) {
          #pragma omp parallel for private(diag,i,j,sum) 
          for (diag = 0; diag < spA_dim0_size; diag++) {
            sum=0;
            #pragma omp simd reduction(+:sum)
            for (i = 0; i < iSize; i++) {
              j = spA_dim0_crd[diag] + i;
              if (j >=0 && j < jSize) {
                sum += spA_vector[diag][i] * inB->data[j];
              }
            }
            outC->data[i]=sum;
          }
        }
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/10000 << " ms"<< std::endl;
    }

    void _mlir_ciface_calculateCOOSpMV(StridedMemRefType<float, 1> *out, void *ptr, 
                                     StridedMemRefType<float, 1> *input, StridedMemRefType<float, 1> *ref) {
    UniSparseStorage* sparT = (UniSparseStorage*)(ptr);
    int32_t *row_crd = sparT->vLevel[1]->crd.data();
    int32_t *col_crd = sparT->vLevel[2]->crd.data();
    float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::cout << "nnz is " << nnz << std::endl;
    std::cout << input->data << std::endl;
    std::cout << ref->data << std::endl;
    for(uint64_t i = 0; i < nnz; i++) {
      int32_t rowInd =row_crd[i];
      int32_t colInd = col_crd[i];
      ref->data[rowInd] += values[i] * input->data[colInd];
    }
    std::cout << "End loop " << std::endl;
    out->data = ref->data;
    out->basePtr = ref->data;
    out->offset = 0;  
    out->strides[0] = 1;
  }

  void _mlir_ciface_calculateCOOSpMM(StridedMemRefType<float, 2> *out, void *ptr, 
                                     StridedMemRefType<float, 2> *input, StridedMemRefType<float, 2> *ref) {
    UniSparseStorage* sparT = (UniSparseStorage*)(ptr);
    int32_t *row_crd = sparT->vLevel[1]->crd.data();
    int32_t *col_crd = sparT->vLevel[2]->crd.data();
    float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    uint64_t kSize = input->sizes[1];
    std::cout << "nnz is " << nnz << std::endl;
    std::cout << input->data << std::endl;
    std::cout << ref->data << std::endl;
    for(uint64_t i = 0; i < nnz; i++) {
      for(uint64_t k = 0; k < kSize; k++) {
        int32_t rowInd =row_crd[i];
        int32_t colInd = col_crd[i];
        ref->data[rowInd*kSize + k] += values[i] * input->data[colInd*kSize + k];
      }
    }
    std::cout << "End loop " << std::endl;
    out->data = ref->data;
    out->basePtr = ref->data;
    out->offset = 0;  
    out->strides[0] = 1;
    out->strides[1] = 1;
  }

  #pragma omp declare simd uniform(x, y) linear(i : 1) aligned(x, y : 32) notinbranch
  void xpy(float* x, float* y, int i) {
    y[i] = x[i] + y[i];
  }

  void _mlir_ciface_kernel_hetero_bdia_spmv_iter(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
    
    int ib, i, k, diag, is, ie;
    UniSparseStorage* spA_CSR = (UniSparseStorage*)inA_CSR;
    UniSparseStorage* spA_BDIA = (UniSparseStorage*)inA_BDIA;
    int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
    int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
    int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
    int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
    int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
    int32_t csr_nnz = spA_CSR->vLevel[2]->crd.size();
//    std::cout << "CSR NNZ is " << csr_nnz << std::endl;
    DataType* CSR_value = spA_CSR->valueArray.data();

    int blockSize = spA_BDIA->vLevel[3]->size;
    std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
    int32_t num_rows = (int32_t)inC->sizes[0];
    int32_t num_cols = (int32_t)inB->sizes[0];
    int runs = 50;
    float alpha = 1.0;
    float beta = 0.0;
    DataType* out = inC->data;
    
    int32_t vec_block = 256;
    int32_t vec_num_blocks = std::ceil((float)num_rows / (float)vec_block);
    DataType* csr_out = (float*)std::malloc(num_rows * sizeof(DataType));

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
    // device malloc
    float* cu_csrVal=0;
    cudaStat1 = cudaMalloc((void**)&cu_csrVal, csr_nnz * sizeof(DataType));
    int* cu_csrRowPtr=0;
    cudaStat2 = cudaMalloc((void**)&cu_csrRowPtr, (num_rows + 1) * sizeof(int));
    int* cu_csrColInd=0;
    cudaStat3 = cudaMalloc((void**)&cu_csrColInd, csr_nnz * sizeof(int));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }
    cudaStat1 = cudaMemcpy(cu_csrVal, CSR_value, csr_nnz * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << "CSR_value: " << CSR_value[0] << " " << CSR_value[1] << " " << CSR_value[2] << " " << CSR_value[3] << std::endl;
    cudaStat2 = cudaMemcpy(cu_csrRowPtr, CSR_dim1_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    std::cout << "CSR_dim1_ptr: " << CSR_dim1_ptr[0] << " " << CSR_dim1_ptr[1] << " " << CSR_dim1_ptr[2] << " " << CSR_dim1_ptr[3] << std::endl;
    cudaStat3 = cudaMemcpy(cu_csrColInd, CSR_dim2_crd, csr_nnz * sizeof(int), cudaMemcpyHostToDevice);
//    std::cout << "CSR_dim2_crd: " << CSR_dim2_crd[0] << " " << CSR_dim2_crd[1] << " " << CSR_dim2_crd[2] << " " << CSR_dim2_crd[3] << std::endl;
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    float* cu_InVec=0;
    cudaStat4 = cudaMalloc((void**)&cu_InVec, num_cols * sizeof(DataType));
    float* cu_OutVec=0;
    cudaStat5 = cudaMalloc((void**)&cu_OutVec, num_rows * sizeof(DataType));
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }
    cudaStat4 = cudaMemcpy(cu_InVec, inB->data, num_cols * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << inB->data[0] << " " << inB->data[1] << " " << inB->data[2] << " " << inB->data[3] << std::endl;
    cudaStat5 = cudaMemcpy(cu_OutVec, inC->data, num_rows * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t gpu_csr_matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix
    cusparseCreateCsr(&gpu_csr_matA, num_rows, num_cols, csr_nnz,
                      cu_csrRowPtr, cu_csrColInd, cu_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F); 
    cusparseCreateDnVec(&vecX, num_cols, cu_InVec, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_rows, cu_OutVec, CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();
    double start0 = omp_get_wtime();
    for (int i = 0; i < runs; i++) {
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer);
//      if(cudaStat6 != CUSPARSE_STATUS_SUCCESS) {
//        printf("Fail compute CSR SpMV on GPU\n");
//        exit(-1);
//      }
//      cudaDeviceSynchronize();
      #pragma omp parallel for private(ib,k,diag,is,ie)
      for (ib = 0; ib < n_blocks-1; ib++) {
        for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
          diag = BDIA_dim2_crd[k];
          is = std::max(ib*blockSize, -diag);
          ie = std::min({(ib+1)*blockSize, (int)num_rows-diag, (int)num_rows});
          #pragma omp simd
          for (i = is; i < ie; i++) {
            inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
          }
        }
      }
      cudaDeviceSynchronize();
//      std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;
    }
    double end0 = omp_get_wtime();
    std::cout << "bdia on CPU and csr on GPU total time = " << end0-start0 << " s"<< std::endl;
    std::cout << "Heterogeneous avg time = " << (end0-start0)*1000/runs << " ms"<< std::endl;
    
    double start1 = omp_get_wtime();
    cudaStat5 = cudaMemcpy(csr_out, cu_OutVec, num_rows * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    std::cout << csr_out[0] << " " << csr_out[1] << " " << csr_out[2] << " " << csr_out[3] << std::endl;
//    for(int i = 0; i < num_rows; i++) {
//      inC->data[i] = inC->data[i] + csr_out[i];
//    }
    #pragma omp parallel for private(ib,i) 
    for (ib = 0; ib < vec_num_blocks; ib++) {
      # pragma omp simd aligned (csr_out, out : 32)
      for (i = ib*vec_block; i < std::min((ib+1)*vec_block, (int)num_rows); i++) {
        xpy(csr_out, out, i);
      }
    }

    double end1 = omp_get_wtime();
    std::cout << "Merge time = " << end1-start1 << " s"<< std::endl;
    std::cout << "Merge avg time = " << (end1-start1)*1000/runs << " ms"<< std::endl;
    outC->data = inC->data;
    outC->basePtr = inC->basePtr;
    outC->offset = inC->offset;  
    outC->sizes[0] = inC->sizes[0];
    outC->strides[0] = inC->strides[0];
  }

  void _mlir_ciface_kernel_bdia_spmv_iter(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
        
        int ib, i, k, diag, is, ie;
        DataType sum;
        UniSparseStorage* spA_CSR = (UniSparseStorage*)inA_CSR;
        UniSparseStorage* spA_BDIA = (UniSparseStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        // int64_t jSize = inB->sizes[0];
        // double csr_time = 0.0;
        // double bdia_time = 0.0;
        unsigned runs = 50;
        std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;

        double start0 = omp_get_wtime();
        for (unsigned time = 0; time < runs; time++) {
          #pragma omp parallel for private(ib,i,k,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=0;
              #pragma omp simd reduction(+:sum)
              for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                sum+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]]);
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
              }
              inC->data[i] = sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              #pragma omp simd
              for (i = is; i < ie; i++) {
                inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
                // if(i==0) {
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
          }
        }
        double end0 = omp_get_wtime();
        std::cout << "Hybrid total time = " << (end0-start0) << " s"<< std::endl;
        std::cout << "Hybrid avg time = " << (end0-start0)*1000/runs << " ms"<< std::endl;


        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->strides[0] = inC->strides[0];
        for(unsigned i = 0; i <4; i++ )
          std::cout <<"outC->data["<<i<<"]=" << outC->data[i]<<std::endl;

    }

    void _mlir_ciface_kernel_bdia_spmm_iter(StridedMemRefType<DataType, 2> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 2> *inB, 
                                      StridedMemRefType<DataType, 2> *inC) {
        
        int ib, i, k, j, diag, is, ie;
        
        UniSparseStorage* spA_CSR = (UniSparseStorage*)inA_CSR;
        UniSparseStorage* spA_BDIA = (UniSparseStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        // int64_t jSize = inB->sizes[0];
        int64_t kSize = inB->sizes[1];
        assert(kSize == inC->sizes[1]);
        DataType *sum;
        double start = omp_get_wtime();

        for (unsigned time = 0; time < 1000; time++) {
          #pragma omp parallel for private(ib,i,k,j,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=new DataType[kSize]();
              // sum=0;
              
              for (j=0;j<kSize;j++) {
                #pragma omp simd reduction(+:sum[j])
                for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                // std::cout << "i="<<i<<", k="<<k<<", CSR_value[k]="<<CSR_value[k]<<
                //   ", CSR_dim2_crd[k]="<<CSR_dim2_crd[k]<<", inB->data[CSR_dim2_crd[k]]="
                //   <<inB->data[CSR_dim2_crd[k]]<<std::endl;
                
                  sum[j]+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]*kSize+j]);
                }
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
                inC->data[i*kSize+j] = sum[j];
              }
              // for (j=0;j<kSize;j++) {
              //   inC->data[i*kSize+j] = sum[j];
              // }
              delete[] sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              // #pragma omp simd
              for (i = is; i < ie; i++) {
                #pragma omp simd
                for(j=0; j<kSize;j++) {
                
                  inC->data[i*kSize+j] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[(i+diag)*kSize+j];
                }
                // if(i==0) { 
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
            // Robot motion planing
            // for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
            //   for(j=0; j<kSize;j++) {
            //     inB->data[i*kSize+j] = inC->data[i*kSize+j];
            //   }
            // }
          }
        }
        
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/1000 << " ms"<< std::endl;

        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->sizes[1] = inC->sizes[1];
        outC->strides[0] = inC->strides[0];
        outC->strides[1] = inC->strides[1];
        for(unsigned i = 0; i <4; i++ ) {
          for(unsigned j = 0; j <4; j++ )
            std::cout <<outC->data[i*kSize+j]<<"  ";
          std::cout << std::endl;
        }

    }


  void _mlir_ciface_kernel_bdia_spmv(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
        
        // std::cout << "inB = ";
        // for (unsigned i = 0; i < jSize; i++)
        //   std::cout << inB->data[i] << " ";
        // std::cout << std::endl;

        

        // A*B + C
        // outC->basePtr = outC->data = inC->data;
        // outC->offset = inC->offset;
        // outC->strides[0] = 1;
        // outC->sizes[0] = inC->sizes[0];
        // printf("inC_data = \n");
        // for (unsigned i=0; i < 4; i++)
        //     printf("%f ", inC->data[i]);
        // printf("\n");
        int ib, i, k, diag, is, ie;
        DataType sum;
        UniSparseStorage* spA_CSR = (UniSparseStorage*)inA_CSR;
        UniSparseStorage* spA_BDIA = (UniSparseStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        // int64_t jSize = inB->sizes[0];
        double start = omp_get_wtime();
        // std::cout << "n_blocks = " <<  n_blocks << std::endl;
        // // std::cout << "BDIA_dim1_ptr_size = " << BDIA_dim1_ptr_size << std::endl;
        // // std::cout << "spA->vLevel[3]->size = " << blockSize << std::endl;
        // std::cout << "spA_BDIA->vLevel[1]->ptr = ";
        // for (auto x: spA_BDIA->vLevel[1]->ptr)
        //   std::cout << x << "  ";
        // std::cout << std::endl;
        // std::cout << "spA_BDIA->vLevel[2]->crd = ";
        // for (auto x: spA_BDIA->vLevel[2]->crd)
        //   std::cout << x << "  ";
        // std::cout << std::endl;
        // std::cout << "spA_BDIA->vectorArray = ";
        // // for (auto i:spA_vector) {
        // //   for (auto j: i) {
        // //     std::cout << j << "  ";
        // //   }
        // //   std::cout << std::endl;
        // // }
        // for (unsigned i = 0; i < BDIA_vector.size(); i++) {
        //   // for (unsigned j = 0; j < BDIA_vector[i].size(); j++)
        //     std::cout << BDIA_vector[i] << " ";
        //   std::cout << std::endl;
        // }
        // std::cout << "CSR_dim1_ptr size = " <<  spA_CSR->vLevel[1]->ptr.size() << std::endl;
        // std::cout << "spA_CSR->vLevel[1]->ptr = ";
        // for (auto j: spA_CSR->vLevel[1]->ptr) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        // std::cout << "CSR_dim2_crd size = " <<  spA_CSR->vLevel[2]->crd.size() << std::endl;
        // std::cout << "spA_CSR->vLevel[2]->crd = ";
        // for (auto j: spA_CSR->vLevel[2]->crd) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        // std::cout << "CSR_value size = " <<  spA_CSR->valueArray.size() << std::endl;
        // std::cout << "spA_CSR->valueArray = ";
        // for (auto j: spA_CSR->valueArray) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        for (unsigned time = 0; time < 10000; time++) {
          #pragma omp parallel for private(ib,i,k,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=0;
              #pragma omp simd reduction(+:sum)
              for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                // std::cout << "i="<<i<<", k="<<k<<", CSR_value[k]="<<CSR_value[k]<<
                //   ", CSR_dim2_crd[k]="<<CSR_dim2_crd[k]<<", inB->data[CSR_dim2_crd[k]]="
                //   <<inB->data[CSR_dim2_crd[k]]<<std::endl;
                sum+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]]);
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
              }
              inC->data[i] = sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              #pragma omp simd
              for (i = is; i < ie; i++) {
                inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
                // if(i==0) {
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
          }
        }
        
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/10000 << " ms"<< std::endl;

        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->strides[0] = inC->strides[0];
        for(unsigned i = 0; i <4; i++ )
          std::cout <<"outC->data["<<i<<"]=" << outC->data[i]<<std::endl;
          // std::cout <<"inC->data[0]=" <<inC->data[0] << ", outC->data[0]=" << outC->data[0]<<std::endl;
        // std::cout << "spA->vectorArray = " << std::endl;
        // // for (auto i:spA_vector) {
        // //   for (auto j: i) {
        // //     std::cout << j << "  ";
        // //   }
        // //   std::cout << std::endl;
        // // }
        // for (unsigned i = 0; i < spA_vector.size(); i++) {
        //   for (unsigned j = 0; j < spA_vector[i].size(); j++)
        //     std::cout << spA_vector[i][j] << " ";
        //   std::cout << std::endl;
        // }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++)
        //     printf("%f ", outC->data[i]);
        // printf("\n");
    }

    void* _mlir_ciface_decompose_BDIA(void* ptr, int32_t blockSize, float thres) {
    UniSparseStorage* sparT = (UniSparseStorage*)ptr;
    // int32_t *row_crd = sparT->vLevel[1]->crd.data();
    // int32_t *col_crd = sparT->vLevel[2]->crd.data();
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    // float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
//    std::vector<int> row_crd(sparT->vLevel[1]->crd);
//    std::vector<int> col_crd(sparT->vLevel[2]->crd);
//    std::vector<float> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[1]->same_path.push_back(0);
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->vLevel[2]->same_path.push_back(0);
    sparT->valueArray.clear();
    // for (unsigned i = 0; i < nnz; i++) {
    //   std::cout << "row = " << row_crd[i] << "col = "<< col_crd[i] << std::endl;
    // }
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    // std::cout << "row_size = " << row_size << ", col_size = " << col_size << ", nnz = " << nnz << std::endl;
    assert(col_size >= row_size);
    // bool *root_same_path = sparT->vLevel[0]->same_path.data();
    // bool *row_same_path = sparT->vLevel[1]->same_path.data();
    // bool *col_same_path = sparT->vLevel[2]->same_path.data();
    // std::cout << "root_same_path size= " << sparT->vLevel[0]->same_path.size() <<  std::endl;
    // std::cout << "row_same_path size= " << sparT->vLevel[1]->same_path.size() <<  std::endl;
    // std::cout << "col_same_path size= " << sparT->vLevel[2]->same_path.size() <<  std::endl;
    // std::cout << "root_ptr size= " << sparT->vLevel[0]->ptr.size() <<  std::endl;
    // std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;

    int** diag_nnz = new int *[((row_size-1)/blockSize)+1];
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
      diag_nnz[i] = new int[row_size+col_size-1];
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
      for (unsigned j = 0; j < row_size+col_size-1; j++)
        diag_nnz[i][j] = 0;
    for(uint64_t i = 0; i < nnz; i++) {
      // if (values[i] == 0) 
      //   continue;
      int new_dim0 = sparT->vLevel[1]->crd[i]/blockSize;
      int new_dim1 = sparT->vLevel[2]->crd[i]-sparT->vLevel[1]->crd[i];
      diag_nnz[new_dim0][new_dim1+col_size-1] += 1;
    }
    // std::cout << "diag_nnz:" << std::endl;
    // for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++) {
    //   for (unsigned j = 0; j < row_size+col_size-1; j++)
    //    std::cout <<  diag_nnz[i][j] << "  ";
    //   std::cout << std::endl;
    // }
    // split the matrix
    // step 1: initialize vectorArray
    auto T_BDIA = new UniSparseStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.push_back(0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);
    // UniSparseStorage* T_COO = new UniSparseStorage;
    // T_COO->initCOO(row_size,col_size);
    

    int row_diag_count = 0;
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      for (unsigned j = 0; j < row_size+col_size-1; j++) {
        if (diag_nnz[i][j] > blockSize*thres) {
          row_diag_count++;
          T_BDIA->vLevel[2]->crd.push_back(j-col_size+1);
          // std::vector<float> new_vec(blockSize, 0.0);
          // T_BDIA->vectorArray.push_back(new_vec);
          for (int k = 0; k < blockSize; k++)
            T_BDIA->vector_1d.push_back(0);
        }
      }
      T_BDIA->vLevel[1]->ptr.push_back(row_diag_count);
    }
    // std::cout << "T_BDIA->vLevel[1]->ptr = ";
    // for (auto elm: T_BDIA->vLevel[1]->ptr)
    //   std::cout << elm << "  ";
    // std::cout << std::endl;
    // std::cout << "T_BDIA->vLevel[2]->crd = ";
    // for (auto elm: T_BDIA->vLevel[2]->crd)
    //   std::cout << elm << "  ";
    // std::cout << std::endl;
    // std::cout << "T_BDIA->vectorArray.size = " << T_BDIA->vectorArray.size() << std::endl;

    //step 2: distribute values
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    // std::vector<int> punch_pos;
    int dia_nnz_count = 0;
    std::string output_file_path = "/work/shared/users/staff/zz546/Sparse_Layout_Dialect/test/Data/output_matrix_market.mtx";
    std::ofstream outfile(output_file_path);
    output_header(outfile, row_size, col_size, nnz);
    for(unsigned i = 0; i < nnz; i++) {
      // if (values[i] == 0) 
      //   continue;
      int new_dim1 = sparT->vLevel[1]->crd[i]/blockSize;
      int new_dim2 = sparT->vLevel[2]->crd[i]-sparT->vLevel[1]->crd[i];
      int new_dim3 = sparT->vLevel[1]->crd[i]%blockSize;
      if (diag_nnz[new_dim1][new_dim2+col_size-1] > blockSize*thres) {
        outfile << sparT->vLevel[1]->crd[i]+1 << " " << sparT->vLevel[2]->crd[i]+1 << " " << std::scientific << std::setprecision(3) << sparT->valueArray[i] << "\n"; 
        // if (row_crd[i] == 0)
        //   std::cout << "col = "<< col_crd[i] << ", values =" << values[i] << std::endl;
        // BDIA
        int diag_block;
        for (diag_block = dim1_ptr[new_dim1]; diag_block < dim1_ptr[new_dim1+1]; diag_block++) 
          if (dim2_crd[diag_block] == new_dim2)
            break;
        // T_BDIA->vectorArray[diag_block][new_dim3] = values[i];
        T_BDIA->vector_1d[diag_block*blockSize+new_dim3] = sparT->valueArray[i];
        dia_nnz_count++;
        // punch_pos.push_back(i);
      } 
      else {
        if (sparT->valueArray.size() > 0) {
          sparT->vLevel[1]->same_path.push_back(sparT->vLevel[1]->crd[i] == sparT->vLevel[1]->crd.back());
          sparT->vLevel[2]->same_path.push_back(
              (sparT->vLevel[1]->crd[i] == sparT->vLevel[1]->crd.back()) && (sparT->vLevel[2]->crd[i] == sparT->vLevel[2]->crd.back()));
        }
        sparT->vLevel[1]->crd.push_back(sparT->vLevel[1]->crd[i]);
        sparT->vLevel[2]->crd.push_back(sparT->vLevel[2]->crd[i]);
        sparT->valueArray.push_back(sparT->valueArray[i]);
        // T_COO->vLevel[1]->crd.push_back(row_crd[i]);
        // T_COO->vLevel[1]->same_path.push_back(row_crd[i]== T_COO->vLevel[1]->crd.back());
        // T_COO->vLevel[2]->crd.push_back(col_crd[i]);
        // T_COO->valueArray.push_back(values[i]);
      }
    }
    outfile.seekp(0);
    output_header(outfile, row_size, col_size, dia_nnz_count);
    outfile.close();


    // for (auto pos: punch_pos) {
    //     sparT->valueArray[pos]=0;
    // }
    sparT->vLevel[0]->ptr.push_back(sparT->vLevel[1]->crd.size());
    
    // std::cout << "row_same_path size= " << sparT->vLevel[1]->same_path.size() <<  std::endl;
    // std::cout << "col_same_path size= " << sparT->vLevel[2]->same_path.size() <<  std::endl;
    // std::cout << "root_ptr size= " << sparT->vLevel[0]->ptr.size() <<  std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << dia_nnz_count <<  std::endl;
    // std::cout << "T_BDIA->vectorArray = " << std::endl;
    // for (auto i=dim1_ptr[0]; i < dim1_ptr[1]; i++) {
    //     std::cout << "diag=" << dim2_crd[i] << ", "<< T_BDIA->vectorArray[i][0] << "  "<<std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "T_COO->valueArray = " << std::endl;
    // for (unsigned x = 0; x < T_COO->vLevel[1]->crd.size(); x++) {
    //   if (T_COO->vLevel[1]->crd[x]==0)
    //     std::cout <<T_COO->vLevel[1]->crd[x]<<", "<<T_COO->vLevel[2]->crd[x]<<", " <<T_COO->valueArray[x] << "  "<<std::endl;
    //   else 
    //     break;
    // }
    // std::cout << std::endl;
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
        free(diag_nnz[i]);
    free(diag_nnz);

    // T_COO->finalizeCOO();
    UniSparseStruct* ret = new UniSparseStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }

  void* _mlir_ciface_decompose_BDIA_opt(void* ptr, int32_t blockSize, float thres) {
    UniSparseStorage* sparT = (UniSparseStorage*)ptr;
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    uint64_t nnz = sparT->vLevel[2]->crd.size();
//    std::vector<int> row_crd(sparT->vLevel[1]->crd);
//    std::vector<int> col_crd(sparT->vLevel[2]->crd);
//    std::vector<float> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->valueArray.clear();
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    assert(col_size >= row_size);
    
    // step 1: initialize vectorArray
    auto T_BDIA = new UniSparseStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.push_back(0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);

    // assume read-in data is in row-major order
    double start = omp_get_wtime();

    uint64_t diag_block_count = 0;
    uint64_t diag_nnz_count = 0;
    std::vector<unsigned> row_ptr;
    std::vector<unsigned> dia_row_ptr;
    row_ptr.push_back(0);
    dia_row_ptr.push_back(0);
    int* diag_nnz = new int[blockSize+col_size-1];
    for(unsigned i = 0; i < blockSize+col_size-1; i++) 
      diag_nnz[i] = 0;
    int prev_row_block = sparT->vLevel[1]->crd[0]/blockSize;

    for(uint64_t i = 0; i < nnz; i++) {
      int new_dim1 = sparT->vLevel[1]->crd[i] / blockSize;
      int new_dim2 = sparT->vLevel[2]->crd[i] - sparT->vLevel[1]->crd[i];
      if (new_dim1 == prev_row_block) {
        diag_nnz[new_dim2+(new_dim1+1)*blockSize-1] += 1;
      } else {
        for (uint64_t j = 0; j < blockSize+col_size-1; j++) {
          if (diag_nnz[j]> blockSize*thres) {
            diag_block_count++;
            diag_nnz_count += diag_nnz[j];
            int64_t offset=j-(prev_row_block+1)*blockSize+1;
            // std::cout <<"row="<<prev_row_block<<", j="<<j<< ", diag="<< offset
            // <<", diag_nnz[j]="<<diag_nnz[j]<< ", diag_nnz_count=" << diag_nnz_count <<std::endl;
            T_BDIA->vLevel[2]->crd.push_back(offset);
            for (int k = 0; k < blockSize; k++)
              T_BDIA->vector_1d.push_back(0);
          }
          diag_nnz[j] = 0;
        }
        for (int m = prev_row_block; m <new_dim1; m++) {
          T_BDIA->vLevel[1]->ptr.push_back(diag_block_count);
          row_ptr.push_back(i);
          dia_row_ptr.push_back(diag_nnz_count);
        }
        
        prev_row_block = new_dim1;
        diag_nnz[new_dim2+(new_dim1+1)*blockSize-1] += 1;
      }
    }
    for (uint64_t j = 0; j < blockSize+col_size-1; j++) {
      if (diag_nnz[j]> blockSize*thres) {
        diag_block_count++;
        diag_nnz_count += diag_nnz[j];
        int64_t offset=j-(prev_row_block+1)*blockSize+1;
        T_BDIA->vLevel[2]->crd.push_back(offset);
        for (int k = 0; k < blockSize; k++)
          T_BDIA->vector_1d.push_back(0);
      }
    }

    for (unsigned m = prev_row_block; m <((row_size-1)/blockSize)+1; m++) {
      T_BDIA->vLevel[1]->ptr.push_back(diag_block_count);
      row_ptr.push_back(nnz);
      dia_row_ptr.push_back(diag_nnz_count);
    }
    
    delete []diag_nnz;

    // parallelize
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    sparT->vLevel[1]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path[0]=0;
    sparT->vLevel[2]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path[0]=0;
    sparT->valueArray.resize(nnz-diag_nnz_count);
    sparT->vLevel[0]->ptr.push_back(nnz-diag_nnz_count);
    unsigned i, pos;
    int iter2_dim1, iter2_dim2, iter2_dim3, start_pos, end_pos, insert_pos;
    unsigned COO_pos;
    bool is_BDIA;
    double end_1 = omp_get_wtime();
    for (unsigned time = 0; time < 100; time++) {
    #pragma omp parallel for private(i, pos,iter2_dim1, iter2_dim2, \
          iter2_dim3, start_pos, end_pos, insert_pos, COO_pos, is_BDIA)
    for (i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      COO_pos=row_ptr[i]-dia_row_ptr[i];
      for(pos = row_ptr[i]; pos < row_ptr[i+1]; pos++) {
        iter2_dim1 = sparT->vLevel[1]->crd[pos]/blockSize;
        iter2_dim2 = sparT->vLevel[2]->crd[pos]-sparT->vLevel[1]->crd[pos];
        iter2_dim3 = sparT->vLevel[1]->crd[pos]%blockSize;
        start_pos = dim1_ptr[iter2_dim1];
        end_pos = dim1_ptr[iter2_dim1+1];
        // std::cout<<"start_pos="<<start_pos<<", end_pos="<<end_pos<<std::endl;
        is_BDIA=false;
        for (insert_pos = start_pos; 
            insert_pos < end_pos; insert_pos++) {
          if (iter2_dim2 == dim2_crd[insert_pos]) {
            is_BDIA = true;
            break;
          }
        }
        if (is_BDIA) {
          T_BDIA->vector_1d[insert_pos*blockSize+iter2_dim3] = sparT->valueArray[pos];
        } else {
          // COO_pos = COO_start_pos+COO_pos_iter;
          if (COO_pos > 0) {
            sparT->vLevel[1]->same_path[COO_pos]=(sparT->vLevel[1]->crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]);
            sparT->vLevel[2]->same_path[COO_pos]=(
                (sparT->vLevel[1]->crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]) && (sparT->vLevel[2]->crd[pos] == sparT->vLevel[2]->crd[COO_pos-1]));
          }
          sparT->vLevel[1]->crd[COO_pos]=(sparT->vLevel[1]->crd[pos]);
          sparT->vLevel[2]->crd[COO_pos]=(sparT->vLevel[2]->crd[pos]);
          sparT->valueArray[COO_pos]=(sparT->valueArray[pos]);
          // std::cout<<"COO_pos="<<COO_pos<<", crd[1] size="<<sparT->vLevel[1]->crd.size()
          // <<", crd="<<sparT->vLevel[1]->crd[COO_pos]<<", value="<<sparT->valueArray[COO_pos]<< std::endl;
          COO_pos++;
        }
      }
    }
    }
    double end = omp_get_wtime();
    std::cout << "decompose before omp time = " << end_1-start << " s"<< std::endl;
    std::cout << "decompose omp time = " << end-end_1 << " s"<< std::endl;
    std::cout << "decompose total time = " << (end-end_1)/100+end_1-start << " s"<< std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << diag_nnz_count <<  std::endl;
    // std::cout<< "row_ptr: ";
    // for(auto elm : row_ptr)
    //   std::cout<<elm<< "  ";
    // std::cout<<std::endl;

    UniSparseStruct* ret = new UniSparseStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }

  void* _mlir_ciface_decompose_BDIA_opt2(void* ptr, int32_t blockSize, float thres) {
    UniSparseStorage* sparT = (UniSparseStorage*)ptr;
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::vector<int> row_crd(sparT->vLevel[1]->crd);
    std::vector<int> col_crd(sparT->vLevel[2]->crd);
    std::vector<DataType> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->valueArray.clear();
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    assert(col_size >= row_size);
    
    // step 1: initialize vectorArray
    auto T_BDIA = new UniSparseStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.resize(((row_size-1)/blockSize)+2, 0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);
    
    // assume read-in data is in row-major order
    double start = omp_get_wtime();

    std::vector<int> diag_block_count(((row_size-1)/blockSize)+1, 0);
    // std::cout<<"diag_block_count size="<<diag_block_count.size()<<std::endl;
    uint64_t diag_nnz_count = 0;
    double mem_0 = omp_get_wtime();
    std::vector<unsigned> row_ptr(((row_size-1)/blockSize)+2, 0);
    // row_ptr.push_back(0);
    double mem_1 = omp_get_wtime();
    std::vector<unsigned> dia_row_ptr(((row_size-1)/blockSize)+2, 0);
    double mem_2 = omp_get_wtime();
    
    double mem_3 = omp_get_wtime();
    // std::vector<int> diag_nnz((((row_size-1)/blockSize)+1)*(blockSize+col_size-1), 0);
    std::vector<std::vector<int>> diag_off(((row_size-1)/blockSize)+1);

    double mem_4 = omp_get_wtime();

    int first_dim1 = row_crd[0]/blockSize;
    for (int m = 0; m < first_dim1; m++)
      row_ptr[m+1]=0;
    int prev_dim1, new_dim1, init_j;
    unsigned init_i;
    double end_0 = omp_get_wtime();
    #pragma omp parallel for private(prev_dim1, new_dim1, init_j, init_i)
    for(init_i = 1; init_i < nnz; init_i++) {
      prev_dim1 = row_crd[init_i-1]/blockSize;
      new_dim1 = row_crd[init_i]/blockSize;
      if (new_dim1 != prev_dim1) {
        for (init_j = prev_dim1; init_j < new_dim1; init_j++)
          row_ptr[init_j+1]=init_i;
      } 
    }
    // std::cout << "new_dim1 = "<<new_dim1<<std::endl;
    for (unsigned m = row_crd[nnz-1]/blockSize; m < ((row_size-1)/blockSize)+1; m++)
      row_ptr[m+1] = nnz;
    assert(row_ptr.size() == ((row_size-1)/blockSize)+2);
    // std::sort(row_ptr.begin(), row_ptr.end());
    // std::cout << "row_ptr = ";
    // for (unsigned m = 0; m < ((row_size-1)/blockSize)+2; m++)
    //   std::cout << row_ptr[m] << "  ";
    // std::cout << std::endl;

    //parallelize
    double end_1 = omp_get_wtime();
    unsigned iter1_i, iter1_pos, iter1_j;
    int iter1_dim2;
    std::vector<int> diag_nnz;
    for(unsigned time = 0; time < 1; time++) {
    #pragma omp parallel for private(diag_nnz, iter1_i, iter1_pos, iter1_j, iter1_dim2)
    for (iter1_i = 0; iter1_i < ((row_size-1)/blockSize)+1; iter1_i++) {
      diag_nnz.clear();
      diag_nnz.resize(blockSize+col_size-1, 0);
      diag_block_count[iter1_i] = 0;
      dia_row_ptr[iter1_i+1] = 0;
      std::vector<int>().swap(diag_off[iter1_i]);
      for(iter1_pos = row_ptr[iter1_i]; iter1_pos < row_ptr[iter1_i+1]; iter1_pos++) {
        // iter1_dim1 = row_crd[iter1_pos]/blockSize;
        iter1_dim2 = col_crd[iter1_pos]-row_crd[iter1_pos];
        // iter1_dim3 = row_crd[iter1_pos]%blockSize;
        diag_nnz[iter1_dim2+(iter1_i+1)*blockSize-1] += 1;
      }
      // std::cout<<"row block="<<iter1_i<<", diag_nnz=";
      // for (auto elm: diag_nnz)
      //   std::cout<<elm<<"  ";
      // std::cout<<std::endl;
      for (iter1_j = 0; iter1_j < blockSize+col_size-1; iter1_j++) {
        if (diag_nnz[iter1_j] > blockSize*thres) {
          diag_block_count[iter1_i] += 1;
          dia_row_ptr[iter1_i+1] += diag_nnz[iter1_j];
          int offset = (int)iter1_j-(iter1_i+1)*blockSize+1;
          diag_off[iter1_i].push_back(offset);
          // if (iter1_i==0)
            // std::cout<<"dia_row_ptr["<<iter1_i+1<<  "] = "<<dia_row_ptr[iter1_i+1]<<std::endl;
          // diag_nnz_count += diag_nnz[iter1_i][iter1_j];
        }
      }
    }
    }

    double end_2 = omp_get_wtime();
    

    int total_dia_block = 0;
    // std::cout<<"((row_size-1)/blockSize)+1="<<((row_size-1)/blockSize)+1<<std::endl;
    for (unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
      dia_row_ptr[init+1] += dia_row_ptr[init];
      total_dia_block += diag_block_count[init];
      T_BDIA->vLevel[1]->ptr[init+1] = total_dia_block;
      // std::cout<<"init="<<init<<", dia_row_ptr[init+1]="<<dia_row_ptr[init+1]<<", diag_block_count[init]="<<diag_block_count[init]<<std::endl;
      // std::cout << "T_BDIA->vLevel[1]->ptr["<<init+1<<"]="<<T_BDIA->vLevel[1]->ptr[init+1]<<std::endl;
      
    }
      // std::cout << "T_BDIA->vLevel[1]->ptr.size()="<<T_BDIA->vLevel[1]->ptr.size()<<std::endl;
      
    diag_nnz_count = dia_row_ptr[((row_size-1)/blockSize)+1];
    T_BDIA->vLevel[2]->crd.reserve(total_dia_block);
    for (unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
      for (auto elm: diag_off[init])
        T_BDIA->vLevel[2]->crd.push_back(elm);
      std::vector<int>().swap(diag_off[init]);
    }
    std::vector<std::vector<int>>().swap(diag_off);
    std::vector<int>().swap(diag_block_count);
    
    
    assert(T_BDIA->vLevel[2]->crd.size()==(unsigned)total_dia_block);
    T_BDIA->vector_1d.resize(total_dia_block*blockSize, 0.0);

    // unsigned iter3_i, iter3_j;
    // #pragma omp parallel for private(iter3_i, iter3_j)
    // for (iter3_i = 0; iter3_i < ((row_size-1)/blockSize)+1; iter3_i++) {
    //   for (iter3_j = 0; iter3_j < blockSize+col_size-1; iter3_j++) {
    //     if (diag_nnz[iter3_i*(blockSize+col_size-1) + iter3_j] > blockSize*thres) {
    //       int64_t offset = iter3_j-(iter3_i+1)*blockSize+1;
    //       unsigned pos = T_BDIA->vLevel[1]->ptr[iter3_i];
    //       T_BDIA->vLevel[2]->crd[pos] = offset;
    //     }
    //   }
    // }

    // parallelize
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    // std::cout << "T_BDIA->vLevel[2]->crd:"<<std::endl;
    // // for(unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
    //   for (auto elm: T_BDIA->vLevel[2]->crd)
    //     std::cout << elm <<"  ";
    //   std::cout<<std::endl;
    // // }
    // std::cout << "nnz-diag_nnz_count: "<<nnz-diag_nnz_count<< ", sparT->vLevel[1]->crd.maxsize="<<
    // sparT->vLevel[1]->crd.max_size()<<std::endl;

    sparT->vLevel[1]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path[0]=0;
    sparT->vLevel[2]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path[0]=0;
    sparT->valueArray.resize(nnz-diag_nnz_count);
    sparT->vLevel[0]->ptr.push_back(nnz-diag_nnz_count);
    unsigned i, pos;
    int iter2_dim1, iter2_dim2, iter2_dim3, start_pos, end_pos, insert_pos;
    unsigned COO_pos;
    bool is_BDIA;
    double end_3 = omp_get_wtime();
    for (unsigned time = 0; time < 1; time++) {
    #pragma omp parallel for private(i, pos,iter2_dim1, iter2_dim2, \
          iter2_dim3, start_pos, end_pos, insert_pos, COO_pos, is_BDIA)
    for (i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      COO_pos=row_ptr[i]-dia_row_ptr[i];
      for(pos = row_ptr[i]; pos < row_ptr[i+1]; pos++) {
        iter2_dim1 = row_crd[pos]/blockSize;
        iter2_dim2 = col_crd[pos]-row_crd[pos];
        iter2_dim3 = row_crd[pos]%blockSize;
        start_pos = dim1_ptr[iter2_dim1];
        end_pos = dim1_ptr[iter2_dim1+1];
        // std::cout<<"start_pos="<<start_pos<<", end_pos="<<end_pos<<std::endl;
        is_BDIA=false;
        for (insert_pos = start_pos; 
            insert_pos < end_pos; insert_pos++) {
          if (iter2_dim2 == dim2_crd[insert_pos]) {
            is_BDIA = true;
            break;
          }
        }
        if (is_BDIA) {
          T_BDIA->vector_1d[insert_pos*blockSize+iter2_dim3] = values[pos];
          // std::cout<<"T_BDIA->vector_1d["<<insert_pos*blockSize+iter2_dim3<<"] = "<<values[pos]<<std::endl;
        } else {
          // COO_pos = COO_start_pos+COO_pos_iter;
          if (COO_pos > 0) {
            sparT->vLevel[1]->same_path[COO_pos]=(row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]);
            sparT->vLevel[2]->same_path[COO_pos]=(
                (row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]) && (col_crd[pos] == sparT->vLevel[2]->crd[COO_pos-1]));
          }
          sparT->vLevel[1]->crd[COO_pos]=(row_crd[pos]);
          sparT->vLevel[2]->crd[COO_pos]=(col_crd[pos]);
          sparT->valueArray[COO_pos]=(values[pos]);
          // std::cout<<"COO_pos="<<COO_pos<<", crd[1] size="<<sparT->vLevel[1]->crd.size()
          // <<", crd="<<sparT->vLevel[1]->crd[COO_pos]<<", value="<<sparT->valueArray[COO_pos]<< std::endl;
          COO_pos++;
        }
      }
    }
    }
    // std::cout << "sparT->vLevel[1]->crd.size() = " << sparT->vLevel[1]->crd.size() <<  std::endl;
    // std::cout << "sparT->vLevel[0]->ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    // std::cout << "sparT->vLevel[1]->same_path: ";
    // for (auto elm: sparT->vLevel[1]->same_path)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[2]->same_path: ";
    // for (auto elm: sparT->vLevel[2]->same_path)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[1]->crd: ";
    // for (auto elm: sparT->vLevel[1]->crd)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[2]->crd: ";
    // for (auto elm: sparT->vLevel[2]->crd)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->valueArray: ";
    // for (auto elm: sparT->valueArray)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    double end = omp_get_wtime();
    std::cout << "mem_0-start time = " << mem_0-start << " s"<< std::endl;
    std::cout << "mem_1-mem_0 time = " << mem_1-mem_0 << " s"<< std::endl;
    std::cout << "mem_2-mem_1 time = " << mem_2-mem_1 << " s"<< std::endl;
    std::cout << "mem_3-mem_2 time = " << mem_3-mem_2 << " s"<< std::endl;
    std::cout << "mem_4-mem_3 time = " << mem_4-mem_3 << " s"<< std::endl;
    std::cout << "end_0-mem_4 time = " << end_0-mem_4 << " s"<< std::endl;
    std::cout << "decompose end_0 - start time = " << end_0 - start << " s"<< std::endl;
    std::cout << "decompose end_1 - end_0 time = " << end_1 - end_0 << " s"<< std::endl;
    std::cout << "decompose end_2 - end_1 time = " << end_2 - end_1 << " s"<< std::endl;
    std::cout << "decompose end_3 - end_2 time = " << end_3 - end_2 << " s"<< std::endl;
    std::cout << "decompose end - end_3 time = " << end - end_3 << " s"<< std::endl;
    std::cout << "decompose end_3 - start time = " << end_3-start << " s"<< std::endl;
    std::cout << "decompose total time = " << (end-end_3)/1000+end_3-end_2 + (end_2-end_1)/1000+(end_1-start)<< " s"<< std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << diag_nnz_count <<  std::endl;

    UniSparseStruct* ret = new UniSparseStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }

  void* _mlir_ciface_decompose_BELL_COO(void* ptr, int32_t blockSize, float block_thres, float col_thres) {
    UniSparseStorage* sparT = (UniSparseStorage*)ptr;
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::vector<int> row_crd(sparT->vLevel[1]->crd);
    std::vector<int> col_crd(sparT->vLevel[2]->crd);
    std::vector<DataType> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->valueArray.clear();
    std::cout << "blockSize = " << blockSize << ", block_thres = " << block_thres << ", col_thres = " << col_thres << std::endl;
    
    // step 1: initialize vectorArray
    auto T_BELL = new UniSparseStorage();
    for (unsigned i = 0; i <= 5; i++) 
      T_BELL->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    // T_BELL->vLevel[0]->type = LVTRIM ; // how to add FUSE in the same level?
    T_BELL->vLevel[1]->type = LVFUSE | LVTRIM ;
    T_BELL->vLevel[2]->size = ((row_size-1)/blockSize)+1;
    T_BELL->vLevel[2]->type = LVFUSE ;
    T_BELL->vLevel[3]->type = LVFUSE ;
    T_BELL->vLevel[4]->size = blockSize;
    T_BELL->vLevel[5]->size = blockSize;
    T_BELL->dimSizes.push_back(row_size);
    T_BELL->dimSizes.push_back(col_size);
    
    // step 2: assume read-in data is in row-major order
    std::vector<unsigned> row_block_ptr(((row_size-1)/blockSize)+2, 0);
    int prev_dim0, new_dim0, init_j;
    unsigned init_i;
    #pragma omp parallel for private(prev_dim0, new_dim0, init_j, init_i)
    for (init_i = 1; init_i < nnz; init_i++) {
      prev_dim0 = row_crd[init_i-1]/blockSize;
      new_dim0 = row_crd[init_i]/blockSize;
      if (new_dim0 != prev_dim0) {
        for (init_j = prev_dim0; init_j < new_dim0; init_j++)
          row_block_ptr[init_j+1]=init_i;
      } 
    }
    for (unsigned m = row_crd[nnz-1]/blockSize; m < ((row_size-1)/blockSize)+1; m++)
      row_block_ptr[m+1] = nnz;
    // std::cout << "row_block_ptr = ";
    // for (unsigned n = 0; n < row_block_ptr.size(); n++) {
    //   std::cout << row_block_ptr[n] << "  ";
    // }
    // std::cout << "\n";

    // step 2: compute level 1 crd - the max
    unsigned iter1_i, iter1_pos, iter1_j;
    int col_block_id;
    std::vector<unsigned> col_blocks(((row_size-1)/blockSize)+1, 0); 
    // std::vector<unsigned> col_block_nnz(((col_size-1)/blockSize)+1, 0);
    std::vector<std::vector<unsigned>> col_block_crd(((row_size-1)/blockSize)+1);
    #pragma omp parallel for private(iter1_i, iter1_pos, iter1_j, col_block_id)  
    for (iter1_i = 0; iter1_i < ((row_size-1)/blockSize)+1; iter1_i++) {
      std::vector<unsigned> col_block_nnz(((col_size-1)/blockSize)+1, 0);
      for (iter1_pos = row_block_ptr[iter1_i]; iter1_pos < row_block_ptr[iter1_i+1]; iter1_pos++) {
        col_block_id = col_crd[iter1_pos]/blockSize;
        col_block_nnz[col_block_id] += 1;
      }
      // std::cout << "col_block_nnz = ";
      // for (unsigned n = 0; n < col_block_nnz.size(); n++) {
      //   std::cout << col_block_nnz[n] << "  ";
      // }
      // std::cout << "\n";
      for (iter1_j = 0; iter1_j < ((col_size-1)/blockSize)+1; iter1_j++) {
        if (col_block_nnz[iter1_j] > blockSize*blockSize*block_thres) {
          col_blocks[iter1_i] += 1;
          col_block_crd[iter1_i].push_back(iter1_j);
        }
      }
    }
    unsigned max_nnz = *std::max_element(col_blocks.begin(), col_blocks.end());
    unsigned level1_size = unsigned(std::ceil(max_nnz * col_thres));
    for (unsigned i = 0; i < level1_size; i++)
      T_BELL->vLevel[1]->crd.push_back(i);
    // std::cout << "col_blocks = ";
    // for (unsigned n = 0; n < col_blocks.size(); n++) {
    //   std::cout << col_blocks[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "col_block_crd = ";
    // for (unsigned n = 0; n < col_block_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < col_block_crd[n].size(); m++) 
    //     std::cout << col_block_crd[n][m] << "  ";
    // }
    // std::cout << "\n";

    // step 3: compute level 3 crd. Only after the crd order is determined can values be dispatched
    T_BELL->vLevel[3]->crd.resize(T_BELL->vLevel[2]->size * level1_size, 0);
    unsigned iter2_i, iter2_j, pad_val;
    #pragma omp parallel for private(iter2_i, iter2_j, pad_val)  
    for (iter2_i = 0; iter2_i < ((row_size-1)/blockSize)+1; iter2_i++) {
      for (iter2_j = 0; iter2_j < std::min(level1_size, (unsigned)col_block_crd[iter2_i].size()); iter2_j++) {
        T_BELL->vLevel[3]->crd[iter2_i*level1_size + iter2_j] = col_block_crd[iter2_i][iter2_j];
      }
      pad_val = 0;
      while(iter2_j < level1_size) {
        if (std::count(col_block_crd[iter2_i].begin(), col_block_crd[iter2_i].end(), pad_val)) {
          pad_val += 1;
          continue;
        } else {
          T_BELL->vLevel[3]->crd[iter2_i*level1_size + iter2_j] = pad_val;
          pad_val += 1;
          iter2_j += 1;
        }
      }
    }
    // std::cout << "level3_crd = ";
    // for (unsigned n = 0; n < T_BELL->vLevel[3]->crd.size(); n++) {
    //   std::cout << T_BELL->vLevel[3]->crd[n] << "  ";
    // }
    // std::cout << "\n";

    // step 4: compute nnz, and row insert boundaries
    unsigned iter3_i, iter3_pos;
    unsigned inner_row_id, inner_col_id;
    bool is_BELL;
    unsigned find_val_id;
    std::vector<std::vector<unsigned>> COO_row_crd(((row_size-1)/blockSize)+1);
    std::vector<std::vector<unsigned>> COO_col_crd(((row_size-1)/blockSize)+1);
    std::vector<std::vector<float>> COO_val(((row_size-1)/blockSize)+1);
    T_BELL->vectorArray.resize(T_BELL->vLevel[2]->size * level1_size, std::vector<float>(blockSize*blockSize, 0));
    #pragma omp parallel for private(iter3_i, iter3_pos, col_block_id, inner_row_id, inner_col_id, is_BELL)
    for (iter3_i = 0; iter3_i < ((row_size-1)/blockSize)+1; iter3_i++) {
      for (iter3_pos = row_block_ptr[iter3_i]; iter3_pos < row_block_ptr[iter3_i+1]; iter3_pos++) {
        col_block_id = col_crd[iter3_pos]/blockSize;
        inner_row_id = row_crd[iter3_pos]%blockSize;
        inner_col_id = col_crd[iter3_pos]%blockSize;
        // check if col_block_id is in BELL
        is_BELL = false;
        for (find_val_id = iter3_i*level1_size;
             find_val_id < (iter3_i + 1)*level1_size;
             find_val_id ++) {
          if (T_BELL->vLevel[3]->crd[find_val_id] == col_block_id) {
            is_BELL = true;
            break;
          }
        }
        if (is_BELL) {
          unsigned correct_col_id = find_val_id%level1_size;
          // std::cout << "correct_col_id = " << correct_col_id << "\n";
          T_BELL->vectorArray[iter3_i*level1_size+correct_col_id][inner_row_id*blockSize+inner_col_id] = values[iter3_pos];
        } else {
          COO_row_crd[iter3_i].push_back(row_crd[iter3_pos]);
          COO_col_crd[iter3_i].push_back(col_crd[iter3_pos]);
          COO_val[iter3_i].push_back(values[iter3_pos]);
        }
      }
    }
    // std::cout << "vectorArray = ";
    // for (unsigned n = 0; n < T_BELL->vectorArray.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < T_BELL->vectorArray[n].size(); m++) 
    //     std::cout << T_BELL->vectorArray[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_row_crd, size = " << COO_row_crd.size();
    // for (unsigned n = 0; n < COO_row_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_row_crd[n].size(); m++) 
    //     std::cout << COO_row_crd[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_col_crd, size = " << COO_col_crd.size();
    // for (unsigned n = 0; n < COO_col_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_col_crd[n].size(); m++) 
    //     std::cout << COO_col_crd[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_val, size = " << COO_val.size();
    // for (unsigned n = 0; n < COO_val.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_val[n].size(); m++) 
    //     std::cout << COO_val[n][m] << "  ";
    // }
    // std::cout << "\n";

    // step 5: compute valueArray. Dispatch the values to COO and BELL.
    unsigned COO_nnz = 0;
    unsigned iter4_i, iter4_j;
    sparT->vLevel[1]->same_path.push_back(0);
    sparT->vLevel[2]->same_path.push_back(0);
    for (iter4_i = 0; iter4_i < ((row_size-1)/blockSize)+1; iter4_i++) {
      for (iter4_j = 0; iter4_j < COO_row_crd[iter4_i].size(); iter4_j++) {
        sparT->vLevel[1]->crd.push_back(COO_row_crd[iter4_i][iter4_j]);
        sparT->vLevel[2]->crd.push_back(COO_col_crd[iter4_i][iter4_j]);
        sparT->valueArray.push_back(COO_val[iter4_i][iter4_j]);
        if (COO_nnz > 0) {
          bool same_row = (sparT->vLevel[1]->crd[COO_nnz] == sparT->vLevel[1]->crd[COO_nnz-1]);
          bool same_col = (sparT->vLevel[2]->crd[COO_nnz] == sparT->vLevel[2]->crd[COO_nnz-1]);
          sparT->vLevel[1]->same_path.push_back(same_row);
          sparT->vLevel[2]->same_path.push_back(same_row && same_col);
        }
        COO_nnz += 1;
      }
    }    
    sparT->vLevel[1]->ptr.push_back(COO_nnz);
    // std::cout << "COO level1 crd = ";
    // for (unsigned n = 0; n < sparT->vLevel[1]->crd.size(); n++) {
    //   std::cout << sparT->vLevel[1]->crd[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO level2 crd = ";
    // for (unsigned n = 0; n < sparT->vLevel[2]->crd.size(); n++) {
    //   std::cout << sparT->vLevel[2]->crd[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO value = ";
    // for (unsigned n = 0; n < sparT->valueArray.size(); n++) {
    //   std::cout << sparT->valueArray[n] << "  ";
    // }
    // std::cout << "\n";

    UniSparseStruct* ret = new UniSparseStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BELL);
    return (void*) ret;
  }