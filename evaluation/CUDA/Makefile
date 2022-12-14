CUDA_INCLUDE_DIR=/usr/local/cuda/include
CUDA_LIBRARY_DIR=/usr/local/cuda/lib64

#CNPY_INCLUDE_DIR=/work/shared/common/project_build/graphblas/software/cnpy
#CNPY_LIBRARY_DIR=/work/shared/common/project_build/graphblas/software/cnpy/build
CUSPARSELT_DIR=/work/shared/common/Libraries/libcusparse_lt-linux-x86_64-0.3.0.3-archive
CUTENSOR_DIR=/work/shared/common/Libraries/libcutensor-linux-x86_64-1.6.1.5-archive
CNPY_INCLUDE_DIR=/work/shared/common/datasets/versatile_sparse_xcel/cnpy
CNPY_LIBRARY_DIR=/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build


BUILD_DIR=./build

cusparse_spmm: cusparse_spmm.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_spmm.cu -o ${BUILD_DIR}/cusparse_spmm \
		 ${CUDA_CFLAGS} \
		-L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart \
		-L${CUSPARSELT_DIR}/lib -lcusparseLt -ldl -I${CUSPARSELT_DIR}/include \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

cusparse_hybrid_spmm: cusparse_hybrid_ell_coo_spmm.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_hybrid_ell_coo_spmm.cu -o ${BUILD_DIR}/cusparse_hybrid_spmm \
		 ${CUDA_CFLAGS} \
		 -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart \
		-L${CUSPARSELT_DIR}/lib -lcusparseLt -ldl -I${CUSPARSELT_DIR}/include \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

cusparse_spmv: cusparse_spmv.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_spmv.cu -o ${BUILD_DIR}/cusparse_spmv \
		 ${CUDA_CFLAGS} \
		 -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -L${CUSPARSELT_DIR}/lib -lcnpy -lcusparse -lcudart \
		 -I${CUSPARSELT_DIR}/include -lcusparseLt -ldl \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

cusparse_spgemm: cusparse_spgemm.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_spgemm.cu -o ${BUILD_DIR}/cusparse_spgemm \
		-L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -L${CUSPARSELT_DIR}/lib -lcnpy -lcusparse -lcudart \
		${CUDA_CFLAGS} \
		-I${CUSPARSELT_DIR}/include -lcusparseLt -ldl \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

cusparse_spmspv: cusparse_spmspv.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_spmspv.cu -o ${BUILD_DIR}/cusparse_spmspv \
		-L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -L${CUSPARSELT_DIR}/lib -lcnpy -lcusparse -lcudart \
		-I${CUSPARSELT_DIR}/include -lcusparseLt -ldl \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

cusparse_spgemm_lp: cusparse_spgemm_lp.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} \
		cusparse_spgemm_lp.cu -o ${BUILD_DIR}/cusparse_spgemm_lp \
		-L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -L${CUSPARSELT_DIR}/lib -lcnpy -lcusparse -lcudart \
		${CUDA_CFLAGS} \
		-I${CUSPARSELT_DIR}/include -lcusparseLt -ldl \
		-L${CUTENSOR_DIR}/lib/11.0 -lcutensor -I${CUTENSOR_DIR}/include

clean:
	rm -rf ${BUILD_DIR}
