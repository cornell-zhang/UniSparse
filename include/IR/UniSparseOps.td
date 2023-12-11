//===- UniSparseOps.td - UniSparse dialect ops -----------*- tablegen -*-========//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_OPS
#define UNISPARSE_OPS

include "UniSparseDialect.td"
include "UniSparseAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

//===----------------------------------------------------------------------===//
// Base unisparse operation definition.
//===----------------------------------------------------------------------===//

class UniSparse_Op<string mnemonic, list<Trait> traits = []> :
        Op<UniSparse_Dialect, mnemonic, traits>;

def UniSparse_NewOp : UniSparse_Op<"new", [NoSideEffect]> {
    let summary = "Construct a unisparse tensor type from .";
    let description = [{
        Example:
        ```mlir
        %A_COO = unisparse.new 
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs UniSparse_Type:$output);
    let assemblyFormat = [{
       `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def UniSparse_fromFileOp: UniSparse_Op<"fromFile", [NoSideEffect]> {
    let summary = "Construct a unisparse tensor from COO file.";
    let description = [{
        Example:
        ```mlir
        %A_COO = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
    let assemblyFormat = [{
       `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def UniSparse_printStorageOp: UniSparse_Op<"printStorage", [NoSideEffect]> {
    let summary = "Print the Storage of a tensor.";
    let description = [{
        Example:
        ```mlir
        unisparse.printStorage %A_CSR : tensor<?x?xf32, #CSR>
        ```
    }];
    let arguments = (ins AnyType:$input);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input)
    }];
}

def UniSparse_copyOp: UniSparse_Op<"copy", [NoSideEffect]> {
    let summary = "Copy the Storage of a tensor.";
    let description = [{
        Example:
        ```mlir
        %A_CSR_1 = unisparse.copy (%A_CSR)
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input)  `to` type($output)
    }];
}

def UniSparse_checkOp: UniSparse_Op<"check", [NoSideEffect]> {
    let summary = "Check the equivalence of two Storage of a tensor.";
    let description = [{..}];
    let arguments = (ins AnyType:$inputL, AnyType:$inputR);
    let assemblyFormat = [{
        `(` $inputL `,` $inputR `)` attr-dict `:` type($inputL) `,` type($inputR)
    }];
}

def UniSparse_DecomposeOp: UniSparse_Op<"decompose", [NoSideEffect]> {
    let summary = "Split a tensor into sub-tensors according to reduce dimensions";
    let arguments = (
        ins AnyTensor: $input,
            AnyType: $threshold,
            AffineMapAttr: $rmap
    );
    let results = (outs UniSparse_StructType: $output);
    let assemblyFormat = "`(` $input `,` $threshold `)`"
        "attr-dict `:`  type($input) `,` type($threshold) `to` type($output)";
}

def UniSparse_StructConvertOp : UniSparse_Op<"struct_convert", [NoSideEffect]> {
    let summary = "Bulk convert from a struct of source formats into target formats";
    let description = [{
        Example:
        ```mlir
        %A_final = unisparse.struct_convert(%A_split): 
           !unisparse.struct<tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>> to
           !unisparse.struct<tensor<?x?xf32, #ELL>, tensor<?x?xf32, #BCSR>, tensor<?x?xf32, #COO>>
        ```
    }];
    let arguments = (ins UniSparse_StructType:$input);
    let results = (outs UniSparse_StructType:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def UniSparse_ConvertOp : UniSparse_Op<"convert", [NoSideEffect,SameOperandsAndResultElementType]> {
    let summary = "Convert from the source format to the target format.";
    let description = [{
        Example:
        ```mlir
        %A_CSR = unisparse.convert (%A_COO) :
            tensor<4x4xf32, #COO> to tensor<4x4xf32, #CSR>
        ```
    }];
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def UniSparse_ticOp: UniSparse_Op<"tic", [NoSideEffect]> {
    let summary = "..";
    let description = [{..}];
    let assemblyFormat = [{
        `(` `)` attr-dict
    }];
}

def UniSparse_tocOp: UniSparse_Op<"toc", [NoSideEffect]> {
    let summary = "..";
    let description = [{..}];
    let assemblyFormat = [{
        `(` `)` attr-dict
    }];
}

def UniSparse_ToPtrOp: UniSparse_Op<"ptr", [NoSideEffect]> {
    let summary = "Extracts pointers array at given dimension from a tensor";
    let description = [{
        Example:
    }];
    let arguments = (ins AnyTensor: $tensor, Index: $dim);
    let results = (outs AnyStridedMemRefOfRank<1>: $result);
    let assemblyFormat = [{
        $tensor `,` $dim attr-dict `:` type($tensor) `to` type($result)
    }];
}

def UniSparse_ToCrdOp: UniSparse_Op<"crd", [NoSideEffect]> {
    let summary = "Extracts coordinates array at given dimension from a tensor";
    let description = [{
        Example:
    }];
    let arguments = (ins AnyTensor: $tensor, Index: $dim);
    let results = (outs AnyStridedMemRefOfRank<1>: $result);
    let assemblyFormat = [{
        $tensor `,` $dim attr-dict `:` type($tensor) `to` type($result)
    }];
}

def UniSparse_ToValueOp: UniSparse_Op<"value", [NoSideEffect]> {
    let summary = "Extracts values array at given dimension from a tensor";
    let description = [{
        Example:
    }];
    let arguments = (ins AnyTensor: $tensor, Index: $dim);
    let results = (outs AnyStridedMemRefOfRank<1>: $result);
    let assemblyFormat = [{
        $tensor `,` $dim attr-dict `:` type($tensor) `to` type($result)
    }];
}

def UniSparse_To2DValueOp: UniSparse_Op<"2d_value", [NoSideEffect]> {
    let summary = "Extracts values array at given dimension from a tensor";
    let description = [{
        Example:
    }];
    let arguments = (ins AnyTensor: $tensor, Index: $dim);
    let results = (outs AnyStridedMemRefOfRank<2>: $result);
    let assemblyFormat = [{
        $tensor `,` $dim attr-dict `:` type($tensor) `to` type($result)
    }];
}

def UniSparse_ToSizeOp: UniSparse_Op<"size", [NoSideEffect]> {
    let summary = "Extracts size of coordinate at given dimension from a tensor";
    let description = [{
        Example:
    }];
    let arguments = (ins AnyTensor: $tensor, Index: $dim);
    let results = (outs Index: $result);
    let assemblyFormat = [{
        $tensor `,` $dim attr-dict `:` type($tensor) `to` `index`
    }];
}

def UniSparse_StructConstructOp : UniSparse_Op<"struct_construct", [NoSideEffect]> {
    let summary = "struct construct";
    let description = [{

    }];
    let arguments = (ins Variadic<UniSparse_Type>:$input);
    let results = (outs UniSparse_StructType:$output);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output)
    }];
}

def UniSparse_StructAccessOp : UniSparse_Op<"struct_access", [NoSideEffect]>, 
                                         Arguments<(ins UniSparse_StructType:$input, I64Attr:$index)>,
                                         Results<(outs UniSparse_Type:$output)> {
    let summary = "struct access";
    let description = [{

    }];

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `to` type($output)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];
    let hasFolder = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sparse Tensor Management Operations. These operations are "impure" in the
// sense that they do not properly operate on SSA values. Instead, the behavior
// is solely defined by side-effects. These operations provide a bridge between
// the code generator and the support library. The semantics of these operations
// may be refined over time as our sparse abstractions evolve.
//===----------------------------------------------------------------------===//

def UniSparse_InsertOp : UniSparse_Op<"insert", []>,
    Arguments<(ins AnyTensor:$tensor,
               StridedMemRefRankOf<[Index], [1]>:$indices,
               AnyType:$value)> {
  string summary = "Inserts a value into given sparse tensor in lexicographical index order";
  string description = [{
    Inserts the given value at given indices into the underlying sparse
    storage format of the given tensor with the given indices. This
    operation can only be applied when a tensor materializes unintialized
    with a `bufferization.alloc_tensor` operation, the insertions occur in
    strict lexicographical index order, and the final tensor is constructed
    with a `load` operation that has the `hasInserts` attribute set.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    unisparse.insert %tensor, %indices, %val
      : tensor<1024x1024xf64, #CSR>, memref<?xindex>, memref<f64>
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $value attr-dict `:`"
                       " type($tensor) `,` type($indices) `,` type($value)";
}

def UniSparse_ExpandOp : UniSparse_Op<"expand", []>,
    Arguments<(ins AnyTensor:$tensor)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$values,
                  StridedMemRefRankOf<[I1],[1]>:$filled,
                  StridedMemRefRankOf<[Index],[1]>:$added,
                  Index:$count)> {
  string summary = "Expands an access pattern for insertion";
  string description = [{
    Performs an access pattern expansion for the innermost dimensions of the
    given tensor. This operation is useful to implement kernels in which a
    sparse tensor appears as output. This technique is known under several
    different names and using several alternative implementations,
    for example, phase counter [Gustavson72], expanded or switch array
    [Pissanetzky84], in phase scan [Duff90], access pattern expansion [Bik96],
    and workspaces [Kjolstad19].

    The values and filled array have sizes that suffice for a *dense* innermost
    dimension (e.g. a full row for matrices). The added array and count are used
    to store new indices when a false value is encountered in the filled array.
    All arrays should be allocated before the loop (possibly even shared between
    loops in a future optimization) so that their *dense* initialization can be
    amortized over many iterations. Setting and resetting the dense arrays in
    the loop nest itself is kept *sparse* by only iterating over set elements
    through an indirection using the added array, so that the operations are
    kept proportional to the number of nonzeros.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    %values, %filled, %added, %count = unisparse.expand %0
      : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($values)"
                       " `,` type($filled) `,` type($added) `,` type($count)";
}

def UniSparse_CompressOp : UniSparse_Op<"compress", []>,
    Arguments<(ins AnyTensor:$tensor,
                   StridedMemRefRankOf<[Index],[1]>:$indices,
                   AnyStridedMemRefOfRank<1>:$values,
                   StridedMemRefRankOf<[I1],[1]>:$filled,
                   StridedMemRefRankOf<[Index],[1]>:$added,
                   Index:$count)> {
  string summary = "Compressed an access pattern for insertion";
  string description = [{
    Finishes a single access pattern expansion by moving inserted elements
    into the sparse storage scheme. The values and filled array are reset
    in a *sparse* fashion by only iterating over set elements through an
    indirection using the added array, so that the operations are kept
    proportional to the number of nonzeros. See the 'expand' operation
    for more details.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    unisparse.compress %0, %1, %values, %filled, %added, %2
        : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
	  memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $values `,` $filled `,`"
                        " $added `,` $count attr-dict `:` type($tensor) `,`"
			" type($indices) `,` type($values) `,` type($filled) `,`"
			" type($added) `,` type($count)";
}

def UniSparse_LoadOp : UniSparse_Op<"load", [SameOperandsAndResultType]>,
    Arguments<(ins AnyTensor:$tensor, UnitAttr:$hasInserts)>,
    Results<(outs AnyTensor:$result)> {
  let summary =
    "Rematerializes tensor from underlying sparse storage format";
  let description = [{
    Rematerializes a tensor from the underlying sparse storage format of the
    given tensor. This is similar to the `bufferization.to_tensor` operation
    in the sense that it provides a bridge between a bufferized world view
    and a tensor world view. Unlike the `bufferization.to_tensor` operation,
    however, this sparse operation is used only temporarily to maintain a
    correctly typed intermediate representation during progressive
    bufferization.

    The `hasInserts` attribute denote whether insertions to the underlying
    sparse storage format may have occurred, in which case the underlying
    sparse storage format needs to be finalized. Otherwise, the operation
    simply folds away.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    %1 = unisparse.load %0 : tensor<8xf64, #SV>
    ```
  }];
  let assemblyFormat = "$tensor (`hasInserts` $hasInserts^)? attr-dict `:` type($tensor)";
}

//===----------------------------------------------------------------------===//
// Temporary Compute Operators
//===----------------------------------------------------------------------===//
def UniSparse_DiaSpmmOp : UniSparse_Op<"dia_spmm", []>,
    Arguments<(ins AnyTensor:$inA,
                   AnyStridedMemRefOfRank<2>:$inB,
                   AnyStridedMemRefOfRank<2>:$inC)>,
    Results<(outs AnyStridedMemRefOfRank<2>:$outC)> {
  string summary = "Compressed an access pattern for insertion";
  string description = [{
    Finishes a single access pattern expansion by moving inserted elements
    into the sparse storage scheme. The values and filled array are reset
    in a *sparse* fashion by only iterating over set elements through an
    indirection using the added array, so that the operations are kept
    proportional to the number of nonzeros. See the 'expand' operation
    for more details.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    unisparse.compress %0, %1, %values, %filled, %added, %2
        : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
	  memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "`(` $inA `,` $inB `,` $inC  `)` attr-dict `:` type($inA) `,`"
			" type($inB) `,` type($inC) `to` type($outC) ";
}

def UniSparse_DiaSpmvOp : UniSparse_Op<"dia_spmv", []>,
    Arguments<(ins AnyTensor:$inA,
                   AnyStridedMemRefOfRank<1>:$inB,
                   AnyStridedMemRefOfRank<1>:$inC)>,
    Results<(outs AnyStridedMemRefOfRank<1>:$outC)> {
  string summary = "Compressed an access pattern for insertion";
  string description = [{
    Finishes a single access pattern expansion by moving inserted elements
    into the sparse storage scheme. The values and filled array are reset
    in a *sparse* fashion by only iterating over set elements through an
    indirection using the added array, so that the operations are kept
    proportional to the number of nonzeros. See the 'expand' operation
    for more details.

    Note that this operation is "impure" in the sense that its behavior
    is solely defined by side-effects and not SSA values. The semantics
    may be refined over time as our sparse abstractions evolve.

    Example:

    ```mlir
    unisparse.compress %0, %1, %values, %filled, %added, %2
        : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
	  memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "`(` $inA `,` $inB `,` $inC  `)` attr-dict `:` type($inA) `,`"
			" type($inB) `,` type($inC) `to` type($outC) ";
}

def UniSparse_COOSpMVOp: UniSparse_Op<"coo_spmv", [NoSideEffect]> {
    let summary = "A COO SpMV runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmv %tensor, %vector : <?x?xf32, #COO>, memref<?xf32> to memref<?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor, AnyStridedMemRefOfRank<1>: $vector, AnyStridedMemRefOfRank<1>: $out_vec);
    let results = (outs AnyStridedMemRefOfRank<1>: $result);
    let assemblyFormat = [{
        $tensor `,` $vector `,` $out_vec attr-dict `:` type($tensor) `,` type($vector) `,` type($out_vec) `to` type($result)
    }];
}

def UniSparse_COOSpMMOp: UniSparse_Op<"coo_spmm", [NoSideEffect]> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor, AnyStridedMemRefOfRank<2>: $matrix, AnyStridedMemRefOfRank<2>: $out_mat);
    let results = (outs AnyStridedMemRefOfRank<2>: $result);
    let assemblyFormat = [{
        $tensor `,` $matrix `,` $out_mat attr-dict `:` type($tensor) `,` type($matrix) `,` type($out_mat) `to` type($result)
    }];
}

def UniSparse_DecomposeBDIAOp: UniSparse_Op<"decompose_BDIA", [NoSideEffect]> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor, AnyType: $blockSize, AnyType: $thres);
    let results = (outs UniSparse_StructType: $output);
    let assemblyFormat = [{
        $tensor `,` $blockSize `,` $thres  attr-dict `:` type($tensor) `,` type($blockSize) `,` type($thres) `to` type($output)
    }];
}

def UniSparse_DecomposeBELLOp: UniSparse_Op<"decompose_BELL", [NoSideEffect]> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor, AnyType: $blockSize, AnyType: $block_thres, AnyType: $col_thres);
    let results = (outs UniSparse_StructType: $output);
    let assemblyFormat = [{
        $tensor `,` $blockSize `,` $block_thres `,` $col_thres attr-dict `:` type($tensor) `,` type($blockSize) `,` type($block_thres) `,` type($col_thres) `to` type($output)
    }];
}

def UniSparse_BDIASpMVOp: UniSparse_Op<"bdia_spmv", [NoSideEffect]> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor_CSR, AnyTensor: $tensor_BDIA, AnyTensor: $vector, AnyTensor: $out_vec);
    let results = (outs AnyStridedMemRefOfRank<1>: $result);
    let assemblyFormat = [{
        $tensor_CSR `,` $tensor_BDIA `,` $vector `,` $out_vec attr-dict `:` type($tensor_CSR) `,` type($tensor_BDIA) `,` type($vector) `,` type($out_vec) `to` type($result)
    }];
}

def UniSparse_BDIASpMMOp: UniSparse_Op<"bdia_spmm", [NoSideEffect]> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyTensor: $tensor_CSR, AnyTensor: $tensor_BDIA, AnyTensor: $in_mat, AnyTensor: $out_mat);
    let results = (outs AnyStridedMemRefOfRank<2>: $result);
    let assemblyFormat = [{
        $tensor_CSR `,` $tensor_BDIA `,` $in_mat `,` $out_mat attr-dict `:` type($tensor_CSR) `,` type($tensor_BDIA) `,` type($in_mat) `,` type($out_mat) `to` type($result)
    }];
}

def UniSparse_ReleaseOp: UniSparse_Op<"release"> {
    let summary = "A COO SpMM runtime";
    let description = [{
        Example: %0 = unisparse.coo_spmm %tensor, %matrix : <?x?xf32, #COO>, memref<?x?xf32> to memref<?x?xf32>
    }];
    let arguments = (ins AnyType: $tensor);
    let assemblyFormat = [{
        $tensor attr-dict `:` type($tensor)
    }];
}

#endif // UNISPARSE_OPS
