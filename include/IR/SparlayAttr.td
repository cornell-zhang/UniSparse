//===-- SparseTensorAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARLAY_ATTR
#define SPARLAY_ATTR

include "SparlayDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class Sparlay_Attr<string name, list<Trait> traits = []>
	: AttrDef<Sparlay_Dialect, name, traits>;

def SparlayCompressAttr : Sparlay_Attr<"SparlayCompress", []> {
  let mnemonic = "compress";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "CompressMap":$value);
}

def SparlayCrdAttr : Sparlay_Attr<"SparlayCrd", []> {
  let mnemonic = "crd";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "CrdMap": $value);
}

// Sparse tensor encoding attribute.
def SparlayEncodingAttr : Sparlay_Attr<"SparlayEncoding", 
              [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let mnemonic = "encoding";
  let hasCustomAssemblyFormat = 1;

  let description = [{
    Tree-base encoding for Sparse Tensor Layout

    Example:

    ```mlir
    #1 = #sparlay.encoding<{
      compressMap = #sparlay.compress<trim(0,1), fuse(1)>,
      crdMap = #sparlay.crd<(i,j,k)[s0,s1] -> ((i+j minus s0)*4 mod 7, (k + (minus i)) floordiv s1)>
      bitwidth = 8
    }>

    !sparlay.tensor<?x?x?xf64, #1>
    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    "CrdMap":$crdMap,
    "CompressMap":$compressMap,
    "unsigned":$bitWidth
  );

  let genVerifyDecl = 1;
}

#endif // SPARLAY_ATTRS