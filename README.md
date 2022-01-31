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
ninja # build the project
```

To run the tests, type
```sh
ninja check-sparlay
```

To build the documentation from the TableGen description of the dialect operations, run
```sh
ninja mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

## Operations 
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `sparlay.compress` (::mlir::sparlay::CompressOp)

Compress the coordinate arrays selectively from the input tensor.


Syntax:

```
operation ::= `sparlay.compress` `(` $in_indices `,` $in_values `)` attr-dict `:`
              type($in_indices) `,` type($in_values) `to`
              type($out_pointers) `,` type($out_indices) `,` type($out_values)
```

Example:
```mlir
%csr_ptr, %csr_crd, %csr_val = sparlay.compress %in_crd, %in_val 
    { compress_dim = "i", storage_order = affine_map<(i,j)->(i,j)> } :
    !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
    tensor<?xf32> to 
    !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
    !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
    tensor<?xf32>
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`compress_dim` | ::mlir::StringAttr | string attribute
`storage_order` | ::mlir::AffineMapAttr | AffineMap attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`in_indices` | Sparlay struct type
`in_values` | tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`out_pointers` | Sparlay struct type
`out_indices` | Sparlay struct type
`out_values` | tensor of any type values

### `sparlay.pack` (::mlir::sparlay::PackOp)

Remove zero elements selectively from the input tensor.


Syntax:

```
operation ::= `sparlay.pack` `(` $input `)` attr-dict `:` type($input) `to`
              type($out_indices) `,` type($out_values)
```

Example:
```mlir
%A_crd, %A_val = sparlay.pack %A 
    { reduce_dim = "j", padding = "none", 
    storage_order = affine_map<(i,j) -> (i,j)> } :
    tensor<4x4xf32> to 
    !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`reduce_dim` | ::mlir::StringAttr | string attribute
`padding` | ::mlir::StringAttr | string attribute
`storage_order` | ::mlir::AffineMapAttr | AffineMap attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`input` | tensor of any type values

#### Results:

| Result | Description |
| :----: | ----------- |
`out_indices` | Sparlay struct type
`out_values` | tensor of any type values

