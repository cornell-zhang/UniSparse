//===- SparlayOps.cpp - Sparlay dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sparlay/SparlayOps.h"
#include "Sparlay/SparlayDialect.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/AffineMap.h"

#define GET_OP_CLASSES
#include "Sparlay/SparlayOps.cpp.inc"
