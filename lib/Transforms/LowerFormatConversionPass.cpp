//===- LowerFormatConversionPass.cpp --- Lower Format Conversion pass --*-===//
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

#include "Transforms/Passes.h"
#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace sparlay;

#define DEBUG_TYPE "lower-format-conversion"

namespace {

//===----------------------------------------------------------------------===//
// RewritePatterns: Pack operations
//===----------------------------------------------------------------------===//

class PackOpLowering : public OpConversionPattern<sparlay::PackOp> {
public:
    using OpConversionPattern<sparlay::PackOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::PackOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value input = op->getOperand(0);
        
        // auto reduceDimAttr = op->getAttr("reduce_dim");
        // auto paddingAttr = op->getAttr("padding");
        // auto storageOrderAttr = op->getAttr("storage_order");
        auto reduceDim = op.reduce_dim();
        StringRef padding = op.padding();
        AffineMap storageOrder = op.storage_order();
        LLVM_DEBUG(llvm::dbgs()<< "affinemap: ");
        LLVM_DEBUG(storageOrder.print(llvm::dbgs()));
        LLVM_DEBUG(llvm::dbgs() << "  getNumDims: " << storageOrder.getNumDims()
                    << "  getNumSymbols: " << storageOrder.getNumSymbols() << 
                    "getNumResults: " << storageOrder.getNumResults() << 
                    "getNumInputs: " << storageOrder.getNumInputs());

        ShapedType inputTp = input.getType().cast<ShapedType>();
        ArrayRef<int64_t> shape = inputTp.getShape();
        Type indexTp = rewriter.getIndexType();
        auto indexArrTp = MemRefType::get(shape, indexTp);
        Value index_arr = rewriter.create<memref::AllocaOp>(loc, indexArrTp);

        std::vector<int64_t> reduce_shape;
        for (unsigned i = 0; i < shape.size(); i++) {
            if (i != reduceDim.getSExtValue()) {
                // LLVM_DEBUG(llvm::dbgs() << "i = " << i <<" | reduceDim = " << reduceDim.getSExtValue() <<"\n");
                reduce_shape.push_back(shape[i]);
            }
        }
        // for (auto elm : reduce_shape) 
        //     LLVM_DEBUG(llvm::dbgs() << " reduce_shape vector: " << elm <<"\n");

        // ArrayRef<int64_t> reduce_shape_arr(reduce_shape);
        auto nnzPerRowTp = MemRefType::get(ArrayRef<int64_t>(reduce_shape), indexTp);
        Value nnz_per_row = rewriter.create<memref::AllocaOp>(loc, nnzPerRowTp);

        Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
        rewriter.create<linalg::FillOp>(loc, zero, nnz_per_row);

        rewriter.eraseOp(op);
        // LLVM_DEBUG(llvm::dbgs() << "hello!\n"); 
        return success();
    }
}; 
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LowerFormatConversionPass
//===----------------------------------------------------------------------===//


namespace {
struct LowerFormatConversionPass : 
public PassWrapper<LowerFormatConversionPass, FunctionPass> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<scf::SCFDialect, memref::MemRefDialect, 
                        vector::VectorDialect, StandardOpsDialect>();
    }
    void runOnFunction() final;
};
}

void LowerFormatConversionPass::runOnFunction() {
    // auto function = getFunction();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `MemRef` and `Standard` dialects.
    target.addLegalDialect<scf::SCFDialect, memref::MemRefDialect,
                           vector::VectorDialect, StandardOpsDialect>();

    // We also define the Sparlay dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Sparlay operations that don't want
    // to lower as `legal`.
    target.addIllegalDialect<sparlay::SparlayDialect>();
    target.addLegalOp<sparlay::CompressOp>();
    target.addLegalOp<sparlay::StructAccessOp>();
    target.addLegalOp<sparlay::FooOp>();
    target.addLegalOp<linalg::FillOp>(); //?

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Sparlay operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<PackOpLowering>(&getContext());
    // LLVM_DEBUG(llvm::dbgs() << "Has the pattern rewrite applied?\n");

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::sparlay::createLowerFormatConversionPass() {
    return std::make_unique<LowerFormatConversionPass>();
}
