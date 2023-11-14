//===- UniSparseOps.cpp - UniSparse dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/UniSparseOps.h"
#include "IR/UniSparseDialect.h"
#include "IR/UniSparseTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::unisparse;

//===----------------------------------------------------------------------===//
// UniSparse Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// StructAccessOp

void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state,
                           mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType structTy = input.getType().cast<StructType>();
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::LogicalResult StructAccessOp::verify() {
  StructType structTy = this->input().getType().cast<StructType>();
  size_t index = this->index();
  if (index >= structTy.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[index])
    return emitOpError() << "must have the same result type as the struct "
                               "element referred to by the index";
  return mlir::success();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
  auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index();
  return structAttr[elementIndex];
}

#define GET_OP_CLASSES
#include "IR/UniSparseOps.cpp.inc"
