//===-- Passes.td - UniSparse pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the UniSparse/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UNISPARSE_PASSES
#define MLIR_DIALECT_UNISPARSE_PASSES

include "mlir/Pass/PassBase.td"

def LowerFormatConversion : Pass<"lower-format-conversion", "func::FuncOp"> {
    let summary = "Lower operations for format conversion";
    let constructor = "mlir::unisparse::createLowerFormatConversionPass()";
    let dependentDialects = [
        "arith::ArithmeticDialect",
        "func::FuncDialect",
        "memref::MemRefDialect",
        "scf::SCFDialect",
        "vector::VectorDialect",
        "linalg::LinalgDialect",
        "LLVM::LLVMDialect"
    ];
}

def LowerStruct : Pass<"lower-struct", "func::FuncOp"> {
    let summary = "Lower struct conversion";
    let constructor = "mlir::unisparse::createLowerStructPass()";
}

def LowerStructConvert: Pass<"lower-struct-convert", "func::FuncOp"> {
    let summary = "Lower Struct Convert conversion";
    let constructor = "mlir::unisparse::createLowerStructConvertPass()";
    let dependentDialects = [
        "arith::ArithmeticDialect",
        "func::FuncDialect",
        "UniSparseDialect"
    ];
}

def DeadCodeElimination : Pass<"dce", "func::FuncOp"> {
    let summary = "dead code elimination";
    let constructor = "mlir::unisparse::createDeadCodeEliminationPass()";
}

def UniSparseCodegen : Pass<"unisparse-codegen"> {
    let summary = "Automatically generate sparse tensor code from unisparse sparse tensor types";
    let constructor = "mlir::unisparse::createUniSparseCodegenPass()";
    let dependentDialects = [
      "AffineDialect",
      "bufferization::BufferizationDialect",
      "memref::MemRefDialect",
      "scf::SCFDialect",
      "unisparse::UniSparseDialect",
      "vector::VectorDialect"
    ];   
}

#endif // MLIR_DIALECT_UNISPARSE_PASSES
