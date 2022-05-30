//===- SparlayOps.td - Sparlay dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARLAY_OPS
#define SPARLAY_OPS

include "SparlayDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SparlayAttr.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

//===----------------------------------------------------------------------===//
// Base sparlay operation definition.
//===----------------------------------------------------------------------===//

class Sparlay_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Sparlay_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// sparlay::FooOp
//===----------------------------------------------------------------------===//

def Sparlay_FooOp : Sparlay_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `sparlay.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = sparlay.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Sparlay_NewOp : Sparlay_Op<"new", [NoSideEffect]> {
    let summary = "Construct a sparlay tensor type from .";
    let description = [{
        Example:
        ```mlir
        // %A_crd, %A_val = sparlay.pack %A 
        //     { reduce_dim = 1, padding = "none", 
        //     storage_order = affine_map<(i,j) -> (i,j)> } :
        //     tensor<4x4xf32> to 
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
        %A_COO = sparlay.pack (%A) 
            { reduce_dim = 1, padding = "none", 
            storage_order = affine_map<(i,j) -> (i,j)> } :
            tensor<4x4xf32> to 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>>
        ```
    }];
    let arguments = (ins AnyType:$input);
    // let results = (outs Sparlay_Type:$out_indices,
    //                     Sparlay_Type:$out_values);
    let results = (outs Sparlay_Type:$output);
    // let builders = [
    //     OpBuilder<(ins "Value":$input, "IndexAttr":$reduce_dim,
    //                     "StringRef":$padding, "AffineMapAttr":$storage_order)>,
    // ];
    let assemblyFormat = [{
       `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_PackOp : Sparlay_Op<"pack", [NoSideEffect]> {
    let summary = "Remove zero elements selectively from the input tensor.";
    let description = [{
        Example:
        ```mlir
        // %A_crd, %A_val = sparlay.pack %A 
        //     { reduce_dim = 1, padding = "none", 
        //     storage_order = affine_map<(i,j) -> (i,j)> } :
        //     tensor<4x4xf32> to 
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
        %A_COO = sparlay.pack (%A) 
            { reduce_dim = 1, padding = "none", 
            storage_order = affine_map<(i,j) -> (i,j)> } :
            tensor<4x4xf32> to 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>>
        ```
    }];
    let arguments = (ins Sparlay_Type:$input,
                        IndexAttr:$reduce_dim,
                        StrAttr:$padding,
                        AffineMapAttr:$storage_order);
    // let results = (outs Sparlay_Type:$out_indices,
    //                     Sparlay_Type:$out_values);
    let results = (outs Sparlay_Type:$output);
    // let builders = [
    //     OpBuilder<(ins "Value":$input, "IndexAttr":$reduce_dim,
    //                     "StringRef":$padding, "AffineMapAttr":$storage_order)>,
    // ];
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

def Sparlay_CompressOp : Sparlay_Op<"compress", [NoSideEffect]> {
    let summary = "Compress the coordinate arrays selectively from the input tensor.";
    let description = [{
        Example:
        ```mlir
        // %csr_ptr, %csr_crd, %csr_val = sparlay.compress %in_crd, %in_val 
        //     { compress_dim = 0, storage_order = affine_map<(i,j)->(i,j)> } :
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
        //     tensor<?xf32> to 
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
        //     tensor<?xf32>
        %A_CSR = sparlay.compress (%A_COO)
            { compress_dim = 0, storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<!sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>> to 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>>, !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>>
        ```
    }];
    // let arguments = (ins Sparlay_Type:$in_indices,
    //                      Sparlay_Type:$in_values,
    //                      StrAttr:$compress_dim,
    //                      AffineMapAttr:$storage_order);
    let arguments = (ins Sparlay_Type:$input,
                         IndexAttr:$compress_dim,
                         AffineMapAttr:$storage_order);

    // let results = (outs Sparlay_Type:$out_pointers,
    //                     Sparlay_Type:$out_indices,
    //                     Sparlay_Type:$out_values);
    let results = (outs Sparlay_Type:$output);
    // let assemblyFormat = [{
    //     `(` $in_indices `,` $in_values `)` attr-dict `:` 
    //     type($in_indices) `,` type($in_values) `to` 
    //     type($out_pointers) `,` type($out_indices) `,` type($out_values)
    // }];
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output)
    }];
    // let verifier = [{ return ::verify(*this); }];
}

def Sparlay_MultiplyOp : Sparlay_Op<"multiply", [NoSideEffect]> {
    let summary = "multiply";
    let description = [{

    }];
    let arguments = (ins Variadic<Sparlay_Type>:$input, 
                        StrAttr:$target,
                        StrAttr:$pattern);
    let results = (outs Sparlay_Type:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output)
    }];

    // let builders = [
    //     OpBuilder<(ins "Value":$input, "size_t":$index)>
    // ];
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

    // let builders = [
    //     OpBuilder<(ins "Value":$input, "size_t":$index)>
    // ];
    // let verifier = [{ return ::verify(*this); }];
    // let hasFolder = 1;
}

def Sparlay_StructAccessOp : Sparlay_Op<"struct_access", [NoSideEffect]> {
    let summary = "struct access";
    let description = [{

    }];
    let arguments = (ins Sparlay_StructType:$input, I64Attr:$index);
    let results = (outs Sparlay_Type:$output);

    let assemblyFormat = [{
        $input `[` $index `]` attr-dict `:` type($input) `to` type($output)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input, "size_t":$index)>
    ];
    let verifier = [{ return ::verify(*this); }];
    let hasFolder = 1;
}

// def Sparlay_StructConstantOp : 

#endif // SPARLAY_OPS
