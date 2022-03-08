//===-- Passes.td - Sparlay pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Sparlay/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARLAY_PASSES
#define MLIR_DIALECT_SPARLAY_PASSES

include "mlir/Pass/PassBase.td"

def LowerFormatConversion : Pass<"lower-format-conversion"> {
    let summary = "Lower operations for format conversion";
    let constructor = "mlir::sparlay::createLowerFormatConversionPass()";
    let dependentDialects = [
        "memref::MemRefDialect",
        "scf::SCFDialect",
        "StandardOpsDialect",
        "vector::VectorDialect",
        "linalg::LinalgDialect"
    ];
}

def LowerStruct : Pass<"lower-struct"> {
    let summary = "Lower struct conversion";
    let constructor = "mlir::sparlay::createLowerStructPass()";
}

def DeadCodeElimination : Pass<"dce"> {
    let summary = "dead code elimination";
    let constructor = "mlir::sparlay::createDeadCodeEliminationPass()";
}

#endif // MLIR_DIALECT_SPARLAY_PASSES
