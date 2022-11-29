# TACO_INCLUDE_DIR=/home/ubuntu/efs/software/taco/include
# TACO_LIBRARY_DIR=/home/ubuntu/efs/software/taco/build/lib

# GBLAS_INCLUDE_DIR=/home/ubuntu/project/GraphBLAS/include
# GBLAS_LIBRARY_DIR=/home/ubuntu/project/GraphBLAS/lib

#MKL_INCLUDE_DIR=/work/shared/common/CAD_tool/Intel/mkl/mkl/include
#MKL_LIBRARY_DIR=/work/shared/common/CAD_tool/Intel/mkl/mkl/lib/intel64

MKL_INCLUDE_DIR=/opt/intel/oneapi/mkl/2022.2.1/include
MKL_LIBRARY_DIR=/opt/intel/oneapi/mkl/2022.2.1/lib/intel64

# CNPY_INCLUDE_DIR=/work/shared/common/project_build/graphblas/software/cnpy
# CNPY_LIBRARY_DIR=/work/shared/common/project_build/graphblas/software/cnpy/build

BUILD_DIR=./build

# taco_sddmm: taco_sddmm.cpp
# 	mkdir -p $(BUILD_DIR)
# 	g++ -std=c++14 -O3 -I${TACO_INCLUDE_DIR} -L${TACO_LIBRARY_DIR} taco_sddmm.cpp -o $(BUILD_DIR)/taco_sddmm -ltaco

# graphblas_spmm: graphblas_spmm.cpp
# 	mkdir -p $(BUILD_DIR)
# 	g++ -std=c++11 -O3 -I${GBLAS_INCLUDE_DIR} -Wl,-rpath,${GBLAS_LIBRARY_DIR} -L${GBLAS_LIBRARY_DIR} \
# 		graphblas_spmm.cpp -o $(BUILD_DIR)/graphblas_spmm -llagraph -lgraphblas -lm -lcnpy

mkl_spmm: mkl_spmm.cpp
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -O3 -I${MKL_INCLUDE_DIR} \
		${CXXFLAGS} \
		-Wl,--no-as-needed \
		mkl_spmm.cpp -o $(BUILD_DIR)/mkl_spmm \
		-Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a \
		${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a -Wl,--end-group \
		-fopenmp -lpthread -lm -ldl

mkl_spmv: mkl_spmv.cpp
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -O3 -I${MKL_INCLUDE_DIR} \
		${CXXFLAGS} \
		-Wl,--no-as-needed \
		mkl_spmv.cpp -o $(BUILD_DIR)/mkl_spmv \
		-Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a \
		${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a -Wl,--end-group \
		-fopenmp -lpthread -lm -ldl 

mkl_spgemm: mkl_spgemm.cpp
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -O3 -I${MKL_INCLUDE_DIR} \
				${CXXFLAGS} \
                -Wl,--no-as-needed \
                mkl_spgemm.cpp -o $(BUILD_DIR)/mkl_spgemm \
                -Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a \
                ${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a -Wl,--end-group \
                -fopenmp -lpthread -lm -ldl

mkl_spmspv: mkl_spmspv.cpp
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -O3 -I${MKL_INCLUDE_DIR} \
                -Wl,--no-as-needed \
                mkl_spmspv.cpp -o $(BUILD_DIR)/mkl_spmspv \
                -Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a \
                ${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a -Wl,--end-group \
                -fopenmp -lpthread -lm -ldl

mkl_spgemm_lp: mkl_spgemm_lp.cpp
	mkdir -p $(BUILD_DIR)
	g++ -std=c++11 -O3 -I${MKL_INCLUDE_DIR} \
				${CXXFLAGS} \
                -Wl,--no-as-needed \
                mkl_spgemm_lp.cpp -o $(BUILD_DIR)/mkl_spgemm_lp \
                -Wl,--start-group ${MKL_LIBRARY_DIR}/libmkl_intel_lp64.a \
                ${MKL_LIBRARY_DIR}/libmkl_core.a ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a -Wl,--end-group \
                -fopenmp -lpthread -lm -ldl

clean:
	rm -rf ${BUILD_DIR}
