//===- UniSparseDialect.h - UniSparse dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_UNISPARSEDIALECT_H
#define UNISPARSE_UNISPARSEDIALECT_H

#include "mlir/IR/Dialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


#include "IR/UniSparseOpsDialect.h.inc"

#include "IR/UniSparseAttr.h"

namespace mlir {
namespace unisparse {
   UniSparseEncodingAttr getUniSparseEncoding(Type type);
}
}


#endif // UNISPARSE_UNISPARSEDIALECT_H
