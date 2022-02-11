//===- Passes.h - Sparlay Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Sparlay.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_SPARLAY_PASSES_H
#define MLIR_TUTORIAL_SPARLAY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace sparlay {
// Create a pass for flattening format conversion operations to operations in 
// `scf` dialect.
std::unique_ptr<Pass> createFlattenConversionPass();

/// Create a pass for lowering Struct type and related operations.
std::unique_ptr<mlir::Pass> createLowerStructPass();

/// Create a pass for lowering operations the remaining `Sparlay` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
// std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // end namespace sparlay
} // end namespace mlir

#endif // MLIR_TUTORIAL_SPARLAY_PASSES_H