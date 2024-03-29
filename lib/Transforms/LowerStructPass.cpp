//===- LowerStructPass.cpp --- Lower Format Conversion pass ------------*-===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Transforms/Passes.h"
#include "IR/UniSparseDialect.h"
#include "IR/UniSparseOps.h"
#include "IR/UniSparseDialect.h"
#include "IR/UniSparseTypes.h"

#include <cstdio>
#include <cstring>

using namespace mlir;
using namespace unisparse;

#define DEBUG_TYPE "lower-struct"

namespace {

#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// RewritePatterns: StructAccessOp 
//===----------------------------------------------------------------------===//

// class StructLowering : public OpRewritePattern<unisparse::StructAccessOp> {
// public:
//     using OpRewritePattern<unisparse::StructAccessOp>::OpRewritePattern;

//     LogicalResult 
//         matchAndRewrite(unisparse::StructAccessOp op, PatternRewriter &rewriter) const override {
//         // Location loc = op->getLoc();
//         Value input = op->getOperand(0);
//         Value output = op->getResult(0);
//         uint64_t index = op.index();
//         Operation* defOp = input.getDefiningOp();
//         Value replaceInput = defOp->getOperand(index);
//         output.replaceAllUsesWith(replaceInput);
//         LLVM_DEBUG(llvm::dbgs() << "inst: " << op->getName() << "' with "
//                   << op->getNumOperands() << " operands and "
//                   << op->getNumResults() << " results\n");
//         LLVM_DEBUG(llvm::dbgs() << "def inst: " << defOp->getName() << "' with "
//                   << defOp->getNumOperands() << " operands and "
//                   << defOp->getNumResults() << " results\n");
//         rewriter.eraseOp(op);
//         return success();
//     }
// };


//===----------------------------------------------------------------------===//
// LowerStructPass
//===----------------------------------------------------------------------===//


struct LowerStructPass : 
public LowerStructBase<LowerStructPass> {

    void runOnOperation() override {
        getOperation().walk([](Operation *op) {
            if (auto accessOp = dyn_cast<unisparse::StructAccessOp>(op)) {
                Value input = accessOp->getOperand(0);
                Value output = accessOp->getResult(0);
                uint64_t index = accessOp.index();
                Operation* defOp = input.getDefiningOp();
                Value replaceInput;
                if (!dyn_cast<unisparse::StructConstructOp>(defOp)) return;
                replaceInput = defOp->getOperand(index);
                // if (!replaceInput.getType().isa<mlir::MemRefType>()) {
                //     Operation* defOp_1 = replaceInput.getDefiningOp();
                //     replaceInput = defOp_1->getOperand(0);
                // }
                output.replaceAllUsesWith(replaceInput);
                LLVM_DEBUG(llvm::dbgs() << "inst: " << accessOp->getName() << 
                    "loc: " << accessOp->getLoc() << "\n");
                LLVM_DEBUG(llvm::dbgs() << "def inst: " << defOp->getName() << 
                    "loc: " << defOp->getLoc() << "\n");
                op->erase();
            }
        });

        // FuncOp function = getFunction();
        // MLIRContext *ctx = function.getContext();
        // RewritePatternSet patterns(ctx);
        // patterns.add<StructLowering>(&getContext());
        // ConversionTarget target(getContext());
        // (void)applyPatternsAndFoldGreedily(function, std::move(patterns));
    };

};
}

std::unique_ptr<Pass> mlir::unisparse::createLowerStructPass() {
    return std::make_unique<LowerStructPass>();
}
