//===- SparlayOps.cpp - Sparlay dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/SparlayOps.h"
#include "IR/SparlayDialect.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::sparlay;

//===----------------------------------------------------------------------===//
// Sparlay Operations
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

static mlir::LogicalResult verify(StructAccessOp op) {
  StructType structTy = op.input().getType().cast<StructType>();
  size_t index = op.index();
  if (index >= structTy.getNumElementTypes())
    return op.emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = op.getResult().getType();
  if (resultType != structTy.getElementTypes()[index])
    return op.emitOpError() << "must have the same result type as the struct "
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
#include "IR/SparlayOps.cpp.inc"
