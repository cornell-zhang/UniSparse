CUDA_INCLUDE_DIR=/usr/local/cuda-11.2/include
CUDA_LIBRARY_DIR=/usr/local/cuda-11.2/lib64

#CNPY_INCLUDE_DIR=/work/shared/common/project_build/graphblas/software/cnpy
#CNPY_LIBRARY_DIR=/work/shared/common/project_build/graphblas/software/cnpy/build

CNPY_INCLUDE_DIR=/work/shared/common/datasets/versatile_sparse_xcel/cnpy
CNPY_LIBRARY_DIR=/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/datasets/versatile_sparse_xcel/cnpy/build

BUILD_DIR=./build

cusparse_spmm: cusparse_spmm.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} cusparse_spmm.cu -o ${BUILD_DIR}/cusparse_spmm \
		 ${CUDA_CFLAGS} \
		 -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart

cusparse_spmv: cusparse_spmv.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} cusparse_spmv.cu -o ${BUILD_DIR}/cusparse_spmv \
		 ${CUDA_CFLAGS} \
		 -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart

cusparse_spgemm: cusparse_spgemm.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} cusparse_spgemm.cu -o ${BUILD_DIR}/cusparse_spgemm \
		-L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart \
		${CUDA_CFLAGS} \

cusparse_spmspv: cusparse_spmspv.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} cusparse_spmspv.cu -o ${BUILD_DIR}/cusparse_spmspv \
                -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart

cusparse_spgemm_lp: cusparse_spgemm_lp.cu
	mkdir -p $(BUILD_DIR)
	nvcc -std=c++11 -O3 -I${CUDA_INCLUDE_DIR} -I${CNPY_INCLUDE_DIR} cusparse_spgemm_lp.cu -o ${BUILD_DIR}/cusparse_spgemm_lp \
                -L${CUDA_LIBRARY_DIR} -L${CNPY_LIBRARY_DIR} -lcnpy -lcusparse -lcudart \
		${CUDA_CFLAGS} \

clean:
	rm -rf ${BUILD_DIR}
