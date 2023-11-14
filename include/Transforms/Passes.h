//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a set of transforms specific for the UniSparse
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_TRANSFORMS_PASSES_H
#define UNISPARSE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class AffineDialect;

namespace arith {
class ArithmeticDialect;
}

namespace bufferization {
class BufferizationDialect;
}

namespace func {
  class FuncDialect;
}

namespace memref {
class MemRefDialect;
}

namespace scf {
class SCFDialect;
}

namespace vector {
class VectorDialect;
}

namespace linalg {
class LinalgDialect;
}

namespace LLVM {
class LLVMDialect;
}

namespace unisparse {

void populateUniSparseCodegenPatterns(RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createDeadCodeEliminationPass();
std::unique_ptr<mlir::Pass> createLowerStructConvertPass();
std::unique_ptr<mlir::Pass> createLowerFormatConversionPass();
std::unique_ptr<mlir::Pass> createLowerStructPass();
std::unique_ptr<mlir::Pass> createUniSparseCodegenPass();
std::unique_ptr<mlir::Pass> createTmpGenBuffer();



//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

} // namespace unisparse
} // namespace mlir

#endif // UNISPARSE_TRANSFORMS_PASSES_H
