# Sparse Layout dialect

This is a standalone sparse layout dialect along with a standalone `opt`-like tool to operate on that dialect. 

## Development
Outline: https://zhang-21.ece.cornell.edu/doku.php?id=research:personal:jieliu:daily:2022_01_25
## Building

<!-- This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone
``` -->

```sh
mkdir build && cd build
source ./scripts/cmake-config.sh # please change the CMAKE ENV variables to your own path
ninja
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

