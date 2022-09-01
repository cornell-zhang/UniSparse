//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a set of transforms specific for the Sparlay
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SPARLAY_TRANSFORMS_PASSES_H
#define SPARLAY_TRANSFORMS_PASSES_H

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

namespace sparlay {

void populateSparlayCodegenPatterns(RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createDeadCodeEliminationPass();
std::unique_ptr<mlir::Pass> createLowerStructConvertPass();
std::unique_ptr<mlir::Pass> createLowerFormatConversionPass();
std::unique_ptr<mlir::Pass> createLowerStructPass();
std::unique_ptr<mlir::Pass> createSparlayCodegenPass();
std::unique_ptr<mlir::Pass> createTmpGenBuffer();


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

} // namespace sparlay
} // namespace mlir

#endif // SPARLAY_TRANSFORMS_PASSES_H
