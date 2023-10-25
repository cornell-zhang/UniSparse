//===-- SparseTensorAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_ATTR
#define UNISPARSE_ATTR

include "UniSparseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the Tensor attributes will extend this class.
class UniSparse_Attr<string name, list<Trait> traits = []>
	: AttrDef<UniSparse_Dialect, name, traits>;

def UniSparseCompressAttr : UniSparse_Attr<"UniSparseCompress", []> {
  let mnemonic = "compress";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "CompressMap":$value);
}

def UniSparseCrdAttr : UniSparse_Attr<"UniSparseCrd", []> {
  let mnemonic = "crd";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "CrdMap": $value);
}

def UniSparseSumAttr : UniSparse_Attr<"UniSparseSum", []> {
  let mnemonic = "sum";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "SumPrim": $value);
}

def UniSparseEnumerateAttr : UniSparse_Attr<"UniSparseEnumerate", []> {
  let mnemonic = "enumerate";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "EnumeratePrim": $value);
}

def UniSparseScheduleAttr : UniSparse_Attr<"UniSparseSchedule", []> {
  let mnemonic = "schedule";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "SchedulePrim": $value);
}

def UniSparseReorderAttr : UniSparse_Attr<"UniSparseReorder", []> {
  let mnemonic = "reorder";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "ReorderPrim": $value);
}

def UniSparseIndirectAttr : UniSparse_Attr<"UniSparseIndirect", []> {
  let mnemonic = "indirect";
  let hasCustomAssemblyFormat = 1;
  let parameters = (
    ins 
    "SumPrim": $sumVal,
    "EnumeratePrim": $enumVal,
    "SchedulePrim": $schedVal,
    "ReorderPrim": $reorderVal
    );
}

def UniSparseLayoutAttr : UniSparse_Attr<"UniSparseLayout", []> {
  let mnemonic = "layout";
  let hasCustomAssemblyFormat = 1;
  let parameters = (ins "LayoutPrim":$value);
}

// Sparse tensor encoding attribute.
def UniSparseEncodingAttr : UniSparse_Attr<"UniSparseEncoding", 
              [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let mnemonic = "encoding";
  let hasCustomAssemblyFormat = 1;

  let description = [{
    Tree-base encoding for Sparse Tensor Layout

    Example:

    ```mlir
    #1 = #unisparse.encoding<{
      compressMap = #unisparse.compress<trim(0,1), fuse(1)>,
      crdMap = #unisparse.crd<(i,j,k)[s0,s1] -> ((i+j minus s0)*4 mod 7, (k + (minus i)) floordiv s1)>
      bitwidth = 8
    }>

    !unisparse.tensor<?x?x?xf64, #1>
    ```
  }];

  // Data in sparse tensor encoding.
  let parameters = (
    ins
    "CrdMap":$crdMap,
    "CompressMap":$compressMap,
    "unsigned":$bitWidth,
    "IndirectFunc": $indirectFunc,
    "LayoutPrim": $layout
  );

  let genVerifyDecl = 1;
}

#endif // UNISPARSE_ATTRS