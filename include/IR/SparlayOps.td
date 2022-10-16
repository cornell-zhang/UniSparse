//===- SparlayOps.td - Sparlay dialect ops -----------*- tablegen -*-========//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARLAY_OPS
#define SPARLAY_OPS

include "SparlayDialect.td"
include "SparlayAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

//===----------------------------------------------------------------------===//
// Base sparlay operation definition.
//===----------------------------------------------------------------------===//

class Sparlay_Op<string mnemonic, list<Trait> traits = []> :
        Op<Sparlay_Dialect, mnemonic, traits>;

def Sparlay_NewOp : Sparlay_Op<"new", [NoSideEffect]> {
    let summary = "Construct a sparlay tensor type from .";
    let description = [{
        Example:
        ```mlir
        %A_COO = sparlay.new 
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs Sparlay_Type:$output);
    let assemblyFormat = [{
       `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_fromFileOp: Sparlay_Op<"fromFile", [NoSideEffect]> {
    let summary = "Construct a sparlay tensor from COO file.";
    let description = [{
        Example:
        ```mlir
        %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
    let assemblyFormat = [{
       `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_printStorageOp: Sparlay_Op<"printStorage", [NoSideEffect]> {
    let summary = "Print the Storage of a tensor.";
    let description = [{
        Example:
        ```mlir
        sparlay.printStorage %A_CSR : tensor<?x?xf32, #CSR>
        ```
    }];
    let arguments = (ins AnyType:$input);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input)
    }];
}

def Sparlay_copyOp: Sparlay_Op<"copy", [NoSideEffect]> {
    let summary = "Copy the Storage of a tensor.";
    let description = [{
        Example:
        ```mlir
        %A_CSR_1 = sparlay.copy (%A_CSR)
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input)  `to` type($output)
    }];
}

def Sparlay_checkOp: Sparlay_Op<"check", [NoSideEffect]> {
    let summary = "Check the equivalence of two Storage of a tensor.";
    let description = [{..}];
    let arguments = (ins AnyType:$inputL, AnyType:$inputR);
    let assemblyFormat = [{
        `(` $inputL `,` $inputR `)` attr-dict `:` type($inputL) `,` type($inputR)
    }];
}

def Sparlay_DecomposeOp: Sparlay_Op<"decompose", [NoSideEffect]> {
    let summary = "Split a tensor into sub-tensors according to reduce dimensions";
    let arguments = (
        ins AnyTensor: $input,
            AnyType: $threshold,
            AffineMapAttr: $rmap
    );
    let results = (outs Sparlay_StructType: $output);
    let assemblyFormat = "`(` $input `,` $threshold `)`"
        "attr-dict `:`  type($input) `,` type($threshold) `to` type($output)";
}

def Sparlay_StructConvertOp : Sparlay_Op<"struct_convert", [NoSideEffect]> {
    let summary = "Bulk convert from a struct of source formats into target formats";
    let description = [{
        Example:
        ```mlir
        %A_final = sparlay.struct_convert(%A_split): 
           !sparlay.struct<tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>> to
           !sparlay.struct<tensor<?x?xf32, #ELL>, tensor<?x?xf32, #BCSR>, tensor<?x?xf32, #COO>>
        ```
    }];
    let arguments = (ins Sparlay_StructType:$input);
    let results = (outs Sparlay_StructType:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_ConvertOp : Sparlay_Op<"convert", [NoSideEffect,SameOperandsAndResultElementType]> {
    let summary = "Convert from the source format to the target format.";
    let description = [{
        Example:
        ```mlir
        %A_CSR = sparlay.convert (%A_COO) :
            tensor<4x4xf32, #COO> to tensor<4x4xf32, #CSR>
        ```
    }];
    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_ticOp: Sparlay_Op<"tic", [NoSideEffect]> {
    let summary = "..";
    let description = [{..}];
    let assemblyFormat = [{
        `(` `)` attr-dict
    }];
}

def Sparlay_tocOp: Sparlay_Op<"toc", [NoSideEffect]> {
    let summary = "..";
    let description = [{..}];
    let assemblyFormat = [{
        `(` `)` attr-dict
    }];
}

def Sparlay_ToPtrOp: Sparlay_Op<"ptr", [NoSideEffect]> {
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

def Sparlay_ToCrdOp: Sparlay_Op<"crd", [NoSideEffect]> {
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

def Sparlay_ToValueOp: Sparlay_Op<"value", [NoSideEffect]> {
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

def Sparlay_ToSizeOp: Sparlay_Op<"size", [NoSideEffect]> {
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

def Sparlay_StructConstructOp : Sparlay_Op<"struct_construct", [NoSideEffect]> {
    let summary = "struct construct";
    let description = [{

    }];
    let arguments = (ins Variadic<Sparlay_Type>:$input);
    let results = (outs Sparlay_StructType:$output);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output)
    }];
}

def Sparlay_StructAccessOp : Sparlay_Op<"struct_access", [NoSideEffect]>, 
                                         Arguments<(ins Sparlay_StructType:$input, I64Attr:$index)>,
                                         Results<(outs Sparlay_Type:$output)> {
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

def Sparlay_InsertOp : Sparlay_Op<"insert", []>,
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
    sparlay.insert %tensor, %indices, %val
      : tensor<1024x1024xf64, #CSR>, memref<?xindex>, memref<f64>
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $value attr-dict `:`"
                       " type($tensor) `,` type($indices) `,` type($value)";
}

def Sparlay_ExpandOp : Sparlay_Op<"expand", []>,
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
    %values, %filled, %added, %count = sparlay.expand %0
      : tensor<4x4xf64, #CSR> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($values)"
                       " `,` type($filled) `,` type($added) `,` type($count)";
}

def Sparlay_CompressOp : Sparlay_Op<"compress", []>,
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
    sparlay.compress %0, %1, %values, %filled, %added, %2
        : tensor<4x4xf64, #CSR>, memref<?xindex>, memref<?xf64>,
	  memref<?xi1>, memref<?xindex>, index
    ```
  }];
  let assemblyFormat = "$tensor `,` $indices `,` $values `,` $filled `,`"
                        " $added `,` $count attr-dict `:` type($tensor) `,`"
			" type($indices) `,` type($values) `,` type($filled) `,`"
			" type($added) `,` type($count)";
}

def Sparlay_LoadOp : Sparlay_Op<"load", [SameOperandsAndResultType]>,
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
    %1 = sparlay.load %0 : tensor<8xf64, #SV>
    ```
  }];
  let assemblyFormat = "$tensor (`hasInserts` $hasInserts^)? attr-dict `:` type($tensor)";
}

#endif // SPARLAY_OPS
