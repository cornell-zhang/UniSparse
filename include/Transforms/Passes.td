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

def LowerFormatConversion : Pass<"lower-format-conversion", "func::FuncOp"> {
    let summary = "Lower operations for format conversion";
    let constructor = "mlir::sparlay::createLowerFormatConversionPass()";
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
    let constructor = "mlir::sparlay::createLowerStructPass()";
}

def LowerStructConvert: Pass<"lower-struct-convert", "func::FuncOp"> {
    let summary = "Lower Struct Convert conversion";
    let constructor = "mlir::sparlay::createLowerStructConvertPass()";
    let dependentDialects = [
        "arith::ArithmeticDialect",
        "func::FuncDialect",
        "SparlayDialect"
    ];
}

def DeadCodeElimination : Pass<"dce", "func::FuncOp"> {
    let summary = "dead code elimination";
    let constructor = "mlir::sparlay::createDeadCodeEliminationPass()";
}

def SparlayCodegen : Pass<"sparlay-codegen"> {
    let summary = "Automatically generate sparse tensor code from sparlay sparse tensor types";
    let constructor = "mlir::sparlay::createSparlayCodegenPass()";
    let dependentDialects = [
      "AffineDialect",
      "bufferization::BufferizationDialect",
      "memref::MemRefDialect",
      "scf::SCFDialect",
      "sparlay::SparlayDialect",
      "vector::VectorDialect"
    ];   
}

#endif // MLIR_DIALECT_SPARLAY_PASSES
