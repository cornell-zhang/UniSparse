# add self-installed CMake 3.18.0 path
# override the host /usr/local/cmake one with a higher version
export PATH=/work/shared/users/phd/jl3952/installs/cmake-3.18.0-Linux-x86_64/bin/:$PATH
# add self-installed ninja
export PATH=/work/shared/users/phd/jl3952/installs/ninja/build-cmake/:$PATH
# import lld
# export PATH=/work/shared/users/phd/jl3952/installs/MLIR/lld/bin/:$PATH
# maybe specific to LLVM version, tentative to delete later on
# add self-installed LLVM 12.0.0 path
# export PATH=/work/shared/users/phd/jl3952/installs/mlir-aie/llvm-project/build/bin/:$PATH

# add clang-14 and lld path
# export PATH=/work/shared/users/phd/jl3952/installs/MLIR/llvm-project-fork/build/bin:$PATH
export PATH=/work/shared/users/phd/jl3952/installs/MLIR/mlir-aie/llvm-project/build/bin:$PATH
