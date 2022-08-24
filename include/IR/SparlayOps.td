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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SparlayAttr.td"
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

#endif // SPARLAY_OPS
