//===- SparlayDialect.cpp - Sparlay dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sparlay/SparlayDialect.h"
#include "Sparlay/SparlayOps.h"

using namespace mlir;
using namespace mlir::sparlay;

#include "Sparlay/SparlayOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Sparlay dialect.
//===----------------------------------------------------------------------===//

void SparlayDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Sparlay/SparlayOps.cpp.inc"
      >();
}
