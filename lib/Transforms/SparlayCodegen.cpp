//===- SparlayCodegen.cpp - Implementation of code gen for sparse kernel using sparse tensor defined by Sparlay dialect--------------===//

#include "Transforms/Passes.h"
#include "IR/SparlayDialect.h"
#include "IR/SparlayTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sparlay;

#define DEBUG_TYPE "sparlay-codegen"

namespace {

#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

struct GenericOpSparlayCodegen : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparlayCodegen(MLIRContext *context) : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    return success();
  }
};

struct SparlayCodegenPass : public SparlayCodegenBase<SparlayCodegenPass> {
  SparlayCodegenPass() = default;
  SparlayCodegenPass(const SparlayCodegenPass &pass) : SparlayCodegenBase<SparlayCodegenPass>() {}
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateSparlayCodegenPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}

std::unique_ptr<Pass> mlir::sparlay::createSparlayCodegenPass() {
  return std::make_unique<SparlayCodegenPass>();
}

void mlir::sparlay::populateSparlayCodegenPatterns(RewritePatternSet &patterns) {
  patterns.add<GenericOpSparlayCodegen>(patterns.getContext());
}
