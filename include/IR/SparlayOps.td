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
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Base sparlay operation definition.
//===----------------------------------------------------------------------===//

class Sparlay_Op<string mnemonic, list<Trait> traits = []> :
        Op<Sparlay_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// sparlay::NewOp
//===----------------------------------------------------------------------===//

def Sparlay_NewOp : Sparlay_Op<"new", [NoSideEffect]> {
    let summary = "Construct a sparlay tensor type from .";
    let description = [{
        Example:
        ```mlir
        %A_COO = sparlay.new 
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

def Sparlay_fromFileOp: Sparlay_Op<"fromFile", [NoSideEffect]> {
    let summary = "Construct a sparlay tensor from COO file.";
    let description = [{
        Example:
        ```mlir
        %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
        ```
    }];
    let arguments = (ins AnyType:$input);
    let results = (outs AnyTensor:$output);
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

//===----------------------------------------------------------------------===//
// sparlay::ConvertOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// sparlay::AccessOp
//===----------------------------------------------------------------------===//

def Sparlay_AccessOp : Sparlay_Op<"access", [NoSideEffect]> {
    let summary = "Access the dimensional data structures in the tensor.";
    let description = [{
        Example:
        ```mlir
        %A_CSR = sparlay.access (%A_COO) :
            tensor<4x4xf32, #COO> to tensor<4x4xf32, #CSR>
        ```
    }];
    let arguments = (ins AnyTensor:$input,
                         IndexAttr:$dim);
    let results = (outs Sparlay_Type:$output);
    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` type($input) `to` type($output) 
    }];
}

//===----------------------------------------------------------------------===//
// deprecated ops 
//===----------------------------------------------------------------------===//

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
    // let verifier = [{ return ::verify(*this); }];
    let hasFolder = 1;
    let hasVerifier = 1;
}

// def Sparlay_StructConstantOp : 

#endif // SPARLAY_OPS
