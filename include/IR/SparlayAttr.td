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
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class Sparlay_Attr<string name, list<Trait> traits = []>
	: AttrDef<Sparlay_Dialect, name, traits>;

def SparylayAffineAttr : Sparlay_Attr<"SparlayAffine", []> {
  let mnemonic = "affine";
  let parameters = (ins "SparlayAffineMap":$value);
}

// Sparse tensor encoding attribute.
def SparlayEncodingAttr : Sparlay_Attr<"SparlayEncoding", 
              [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let mnemonic = "encoding";

  let description = [{
    Tree-base encoding for Sparse Tensor Layout

    Example:

    ```mlir
    #BCSR = #sparlay.encoding<{
      secondaryMap = #sparlay.affine<(d0, d1, d2, d3) -> (fuse trim d0, d1)>,
      primaryMap = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
      bitwidth = 8
    }>
 
    !sparlay.tensor<?x?x?xf64, #BCSR>
    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    "AffineMap":$primaryMap,
    "SparlayAffineMap":$secondaryMap,
    "unsigned":$bitWidth
  );

  let genVerifyDecl = 1;
}

#endif // SPARLAY_ATTRS