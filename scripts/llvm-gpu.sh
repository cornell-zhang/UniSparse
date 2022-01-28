# configure environment variables for LLVM 13.0.0 and MLIR for GPU 
# add self-installed LLVM 13.0.0 path
export PATH=/work/shared/users/phd/jl3952/installs/MLIR/mlir-aie/llvm-project/build/bin:$PATH
# add MLIR build bin tools
export PATH=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-gpu/build/bin/:$PATH
export CPATH=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-gpu/mlir/include:$CPATH
export LD_LIBRARY_PATH=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-gpu/build/lib:$LD_LIBRARY_PATH

# please source this bash file on servers other than zhang-21
# add new libc++ to the dynamic library path
# export LD_LIBRARY_PATH=/home/jl3952/Packages/anaconda3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jl3952/Packages/anaconda3/pkgs/libstdcxx-ng-9.3.0-hdf63c60_16/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# Zhang-x2 error: libedit.so.0
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jl3952/Packages/anaconda3/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jl3952/Packages/anaconda3/pkgs/libedit-3.1.20191231-he28a2e2_2/lib
