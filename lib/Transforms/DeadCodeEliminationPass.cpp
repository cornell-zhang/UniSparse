//===- DeadCodeEliminationPass.cpp --- Lower Format Conversion pass ------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Transforms/Passes.h"
#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"
#include "IR/SparlayDialect.h"
#include "IR/SparlayTypes.h"

#include <cstdio>
#include <cstring>

using namespace mlir;
using namespace sparlay;

#define DEBUG_TYPE "dce"

namespace {

#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// DeadCodeEliminationPass
//===----------------------------------------------------------------------===//
class DeadCodeElimination : public OpRewritePattern<sparlay::StructConstructOp> {
public:
    using OpRewritePattern<sparlay::StructConstructOp>::OpRewritePattern;

    LogicalResult 
        matchAndRewrite(sparlay::StructConstructOp op, PatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

struct DeadCodeEliminationPass : 
public DeadCodeEliminationBase<DeadCodeEliminationPass> {

    void runOnOperation() override {
        
        func::FuncOp function = getOperation();
        MLIRContext *ctx = function.getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<DeadCodeElimination>(&getContext());
        ConversionTarget target(getContext());
        target.addIllegalOp<sparlay::StructConstructOp>();
        if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    };

};
}

std::unique_ptr<Pass> mlir::sparlay::createDeadCodeEliminationPass() {
    return std::make_unique<DeadCodeEliminationPass>();
}
