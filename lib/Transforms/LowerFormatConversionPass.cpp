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
#include "IR/SparlayDialect.h"
#include "IR/SparlayTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
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

#include <cstdio>
#include <cstring>

using namespace mlir;
using namespace sparlay;

#define DEBUG_TYPE "lower-format-conversion"

namespace {
//===----------------------------------------------------------------------===//
// RewritePatterns: New operations
//===----------------------------------------------------------------------===//

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
static FlatSymbolRefAttr getFunc(Operation *op, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 bool emitCInterface = false) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<FuncOp>(
        op->getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (emitCInterface)
      func->setAttr("llvm.emit_c_interface", UnitAttr::get(context));
  }
  return result;
}

class NewOpLowering : public OpConversionPattern<sparlay::NewOp> {
public:
    using OpConversionPattern<sparlay::NewOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::NewOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        
        Value fileName = op->getOperand(0);
        auto resType = op->getResult(0).getType().dyn_cast<StructType>();

        // if (resType.isa<StructType>())
        //     LLVM_DEBUG(llvm::dbgs() << "is a struct type\n");
        
        llvm::ArrayRef<mlir::Type> elmTypes = resType.getElementTypes();
        Type crdType = elmTypes.front();
        Type dataType = elmTypes.back();
        auto resDimSizes = resType.getDimSizes();
        uint64_t resSize = resDimSizes.size();

        CallOp indicesOp[resSize];
        CallOp valueOp;

        auto indexTp = rewriter.getIndexType();
        Type idxResType = MemRefType::get({ShapedType::kDynamicSize}, indexTp);
        // auto f32Tp = rewriter.getF32Type();
        // Type valResType = MemRefType::get({ShapedType::kDynamicSize}, f32Tp);
        StringRef idxFuncName = "getTensorIndices";
        StringRef valFuncName = "getTensorValues";
        
        for (unsigned i = 0; i < resSize; i++) {
            SmallVector<Value, 3> idxParams;
            idxParams.push_back(fileName);
            idxParams.push_back(
                rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(i)));
            indicesOp[i] = rewriter.create<CallOp>(loc, idxResType, 
                getFunc(op, idxFuncName, idxResType, idxParams, /*emitCInterface=*/true),
                idxParams);
        }
        SmallVector<Value, 3> valParams;
        valParams.push_back(fileName);
        valueOp = rewriter.create<CallOp>(loc, dataType, 
            getFunc(op, valFuncName, dataType, valParams, /*emitCInterface=*/true),
            valParams);
            
        // use struct_construct to construct them into the sparse data structure
        // which will be folded with struct_access or eliminated with DCE in finalize_sparlay_lowering
        SmallVector<Value, 3> input_vec;
        for (unsigned i = 0; i < resSize; i++) {
            input_vec.push_back(indicesOp[i].getResult(0));
        }
        ValueRange input = llvm::makeArrayRef(input_vec);

        Value crdStructOp = rewriter.create<sparlay::StructConstructOp>(loc, crdType, input);
        rewriter.replaceOpWithNewOp<sparlay::StructConstructOp>(op, resType, 
            ValueRange({crdStructOp, valueOp.getResult(0)}));
        return success();
    }
};

//===----------------------------------------------------------------------===//
// RewritePatterns: Pack operations
//===----------------------------------------------------------------------===//

class PackOpLowering : public OpConversionPattern<sparlay::PackOp> {
public:
    using OpConversionPattern<sparlay::PackOp>::OpConversionPattern;

    // enum padding_options {"none", "zero"};

    LogicalResult 
        matchAndRewrite(sparlay::PackOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value input = op->getOperand(0);
        auto reduceDim = op.reduce_dim();
        StringRef padding = op.padding();
        AffineMap storageOrder = op.storage_order();

        ShapedType inputTp = input.getType().cast<ShapedType>();
        ArrayRef<int64_t> shape = inputTp.getShape();
        Type indexTp = rewriter.getIndexType();
        LLVM_DEBUG(llvm::dbgs()<< "shape.size() = " << shape.size() << "\n");

        MemRefType indexArrTp = MemRefType::get(shape, indexTp);
        Value index_arr = rewriter.create<memref::AllocaOp>(loc, indexArrTp);

        std::vector<int64_t> reduce_shape;
        int64_t reduceDimValue = reduceDim.getSExtValue();
        for (unsigned i = 0; i < shape.size(); i++) {
            if (i != reduceDimValue) {
                reduce_shape.push_back(shape[i]);
            }
        }

        MemRefType nnzPerRowTp = MemRefType::get(ArrayRef<int64_t>(reduce_shape), indexTp);
        Value nnz_per_row = rewriter.create<memref::AllocaOp>(loc, nnzPerRowTp);

        Type inputElmTp = input.getType().cast<MemRefType>().getElementType();
        Value zeroElm = rewriter.create<ConstantOp>(loc, inputElmTp, rewriter.getZeroAttr(inputElmTp));

        Value zero = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
        Value one = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));
        // Value dim = rewriter.create<memref::DimOp>(loc, input, zero);

        SmallVector<Value> lb_outer, lb, lb_orig;
        SmallVector<Value> hb_outer, hb, hb_orig;
        SmallVector<Value> step_outer, step, step_orig;
        for (unsigned i = 0; i < shape.size(); i++) {
            Value dimSize = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[i]));
            if (i != reduceDimValue) {
                lb_outer.push_back(zero);
                hb_outer.push_back(dimSize);
                step_outer.push_back(one);
            }
            lb_orig.push_back(zero);
            hb_orig.push_back(dimSize);
            step_orig.push_back(one);
        }
        lb.assign(lb_outer.begin(), lb_outer.end());
        hb.assign(hb_outer.begin(), hb_outer.end());
        step.assign(step_outer.begin(), step_outer.end());
        lb.push_back(zero);
        Value reduceDimSize = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[reduceDimValue]));
        hb.push_back(reduceDimSize);
        step.push_back(one);  

        // rewriter.create<linalg::FillOp>(loc, zero, nnz_per_row);
        // rewriter.create<linalg::FillOp>(loc, dim, index_arr);
        scf::buildLoopNest(rewriter, loc, lb_outer, hb_outer, step_outer, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                builder.create<memref::StoreOp>(loc, zero, nnz_per_row, ivs);
                return;
            });
        scf::buildLoopNest(rewriter, loc, lb_orig, hb_orig, step_orig, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                builder.create<memref::StoreOp>(loc, reduceDimSize, index_arr, ivs);
                return;
            });

              
        scf::buildLoopNest(rewriter, loc, lb, hb, step, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                SmallVector<Value, 3> ivs_vec(lb.size());
                ivs_vec.assign(ivs.begin(), ivs.end());
                auto outer_ivs = llvm::makeArrayRef(ivs_vec).take_front(shape.size() - 1); // wrong size?
                Value inner_iv = ivs_vec.pop_back_val();
                unsigned drop_back_size = shape.size() - reduceDimValue - 1;
                // if (drop_back_size > 0)
                auto inner_reduce_dim_ivs = llvm::makeArrayRef(ivs_vec).drop_back(drop_back_size);
                ivs_vec.pop_back_n(shape.size() - reduceDimValue - 1);
                // LLVM_DEBUG(llvm::dbgs() << "shape size = " << shape.size() << "\n");
                // LLVM_DEBUG(llvm::dbgs() << "reduce dim size = " << reduceDimValue << "\n");
                // LLVM_DEBUG(llvm::dbgs() << "drop_back size = " << shape.size() - reduceDimValue - 1 << "\n");
                // LLVM_DEBUG(llvm::dbgs() << "inner_reduce_dim_ivs.size = " << inner_reduce_dim_ivs.size() << "\n");
                
                Value elm = builder.create<memref::LoadOp>(loc, input, ivs);
                Value not_zero = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, 
                                    elm, zeroElm);
                builder.create<scf::IfOp>(loc, not_zero, [&](OpBuilder &b, Location loc) {
                    Value old_nnz = b.create<memref::LoadOp>(loc, nnz_per_row, outer_ivs);
                    Value new_nnz = b.create<AddIOp>(loc, old_nnz, one);
                    b.create<memref::StoreOp>(loc, new_nnz, nnz_per_row, outer_ivs);
                    
                    // LLVM_DEBUG(llvm::dbgs() << "ivs_vec.size before = " << ivs_vec.size() << "\n");
                    ivs_vec.push_back(old_nnz);
                    for (unsigned i = 0; i < drop_back_size; i++) {
                        // Value tmp = inner_reduce_dim_ivs[i];
                        // LLVM_DEBUG(llvm::dbgs()<<"inner_reduce_dim_ivs: " << tmp << "\n");
                        ivs_vec.push_back(inner_reduce_dim_ivs[i]);
                    }
                    // for (unsigned i = 0; i < ivs_vec.size(); i++) {
                    //     Value tmp = ivs_vec[i];
                    //     // LLVM_DEBUG(llvm::dbgs()<<"ivs_vec: " << tmp << "\n");
                    // }
                    // LLVM_DEBUG(llvm::dbgs() << "ivs_vec.size = " << ivs_vec.size() << "\n");

                    ValueRange index_arr_idx = llvm::makeArrayRef(ivs_vec);
                    b.create<memref::StoreOp>(loc, inner_iv, index_arr, index_arr_idx);
                    b.create<scf::YieldOp>(loc, ValueRange{});
                });
            });
        
        Value nnz_count = rewriter.create<memref::AllocaOp>(loc, MemRefType::get(ArrayRef<int64_t>({1}), indexTp));
        rewriter.create<memref::StoreOp>(loc, zero, nnz_count, zero);
        Value max_nnz = rewriter.create<memref::AllocaOp>(loc, MemRefType::get(ArrayRef<int64_t>({1}), indexTp));
        rewriter.create<memref::StoreOp>(loc, zero, max_nnz, zero);
        scf::buildLoopNest(rewriter, loc, lb_outer, hb_outer, step_outer, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                Value row_nnz = builder.create<memref::LoadOp>(loc, nnz_per_row, ivs);
                Value tmp_count = builder.create<memref::LoadOp>(loc, nnz_count, zero);
                Value sum = builder.create<AddIOp>(loc, row_nnz, tmp_count);
                builder.create<memref::StoreOp>(loc, sum, nnz_count, zero);
                Value tmp_max = builder.create<memref::LoadOp>(loc, max_nnz, zero);
                Value is_row_nnz_greater = builder.create<CmpIOp>(loc, CmpIPredicate::ugt, row_nnz, tmp_max);
                builder.create<scf::IfOp>(loc, is_row_nnz_greater, [&](OpBuilder &b, Location loc) {
                    b.create<memref::StoreOp>(loc, row_nnz, max_nnz, zero);
                    b.create<scf::YieldOp>(loc, ValueRange{});
                });
                return;
            });
        
        // Allocate result arrays
        Value nnz = rewriter.create<memref::LoadOp>(loc, nnz_count, zero);
        MemRefType dynamicIndexType = MemRefType::get(-1, indexTp);
        MemRefType dynamicDataType = MemRefType::get(-1, input.getType().cast<MemRefType>().getElementType());
        // switch (padding) {
        //     case "none":
        SmallVector<Value, 3> idx_array;
        Value val_array;
        if (padding == "none") {
            for (unsigned i = 0; i < shape.size(); i++) {
                idx_array.push_back(rewriter.create<memref::AllocOp>(loc, dynamicIndexType, nnz));
            }
            val_array = rewriter.create<memref::AllocOp>(loc, dynamicDataType, nnz);
        } else if (padding == "zero") {
            std::vector<int64_t> ell_shape;
            for (unsigned i = 0; i < shape.size(); i++) {
                if (i != reduceDimValue) 
                    ell_shape.push_back(shape[i]);
                else
                    ell_shape.push_back(-1);
            }
            MemRefType ellIndexTp = MemRefType::get(ArrayRef<int64_t>(ell_shape), indexTp);
            MemRefType ellDataTp = MemRefType::get(ArrayRef<int64_t>(ell_shape), input.getType().cast<MemRefType>().getElementType());
            // Value ell_array_size = rewriter.create<MulOp>(loc, max_nnz, )
            idx_array.push_back(rewriter.create<memref::AllocOp>(loc, ellIndexTp, max_nnz));
            val_array = rewriter.create<memref::AllocOp>(loc, ellDataTp, max_nnz);
        } else {
            LLVM_DEBUG(llvm::dbgs() << "The padding option in PackOp can only support 'none' and 'zero' now.");
        }
         
        Value max_nnz_val = rewriter.create<memref::LoadOp>(loc, max_nnz, zero);
        Value len_count = rewriter.create<memref::AllocaOp>(loc, MemRefType::get(ArrayRef<int64_t>({1}), indexTp));
        rewriter.create<memref::StoreOp>(loc, zero, len_count, zero); //////
        // affine_map: reorder lb, hb 
        SmallVector<Value> lb_ordered, hb_ordered, step_ordered;
        for (unsigned i = 0; i < shape.size(); i++) {
            unsigned reorderedDimPos = storageOrder.getDimPosition(i);
            LLVM_DEBUG(llvm::dbgs() << "storage order dim = " << storageOrder.getDimPosition(i) <<
                    " | permuted dim = " << storageOrder.getPermutedPosition(i) << "\n");
            if (reorderedDimPos != reduceDimValue) {
                lb_ordered.push_back(zero);
                Value reorderedDimSize = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[reorderedDimPos]));
                hb_ordered.push_back(reorderedDimSize);
                step_ordered.push_back(one);
            }
            else {
                lb_ordered.push_back(zero);
                hb_ordered.push_back(max_nnz_val);
                step_ordered.push_back(one);
            }
        }
        scf::buildLoopNest(rewriter, loc, lb_ordered, hb_ordered, step_ordered, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                // prepare the reordered loading index of the reordered index array
                SmallVector<Value, 3> index_load_dim;
                for (unsigned i = 0; i < shape.size(); i++) {
                    unsigned reorderedDimPos = storageOrder.getPermutedPosition(i);
                    index_load_dim.push_back(ivs[reorderedDimPos]);
                }

                Value reordered_idx = builder.create<memref::LoadOp>(loc, index_arr, llvm::makeArrayRef(index_load_dim));
                if (padding == "none") {
                    Value valid_idx = builder.create<CmpIOp>(loc, CmpIPredicate::ult, reordered_idx, reduceDimSize);
                    builder.create<scf::IfOp>(loc, valid_idx, [&](OpBuilder &b, Location loc) {
                        Value len_count_val = b.create<memref::LoadOp>(loc, len_count, zero);
                        
                        // prepare the reordered loading index of input A, store indices into index arrays
                        SmallVector<Value, 3> load_dim;
                        for (unsigned i = 0; i < shape.size(); i++) {
                            unsigned reorderedDimPos = storageOrder.getPermutedPosition(i);
                            if (i != reduceDimValue) {
                                b.create<memref::StoreOp>(loc, ivs[reorderedDimPos], idx_array[i], len_count_val);
                                load_dim.push_back(ivs[reorderedDimPos]);
                            } else {
                                b.create<memref::StoreOp>(loc, reordered_idx, idx_array[i], len_count_val);
                                load_dim.push_back(reordered_idx);
                            }
                        }
                        Value a_mem_val = b.create<memref::LoadOp>(loc, input, llvm::makeArrayRef(load_dim));
                        b.create<memref::StoreOp>(loc, a_mem_val, val_array, len_count_val);
                        Value len_count_sum = b.create<AddIOp>(loc, len_count_val, one);
                        b.create<memref::StoreOp>(loc, len_count_sum, len_count, zero);
                        b.create<scf::YieldOp>(loc, ValueRange{});
                    });
                } else if (padding == "zero") {

                } else 
                    LLVM_DEBUG(llvm::dbgs() << "The padding option in PackOp can only support 'none' and 'zero' now.");
            
                // builder.create<memref::StoreOp>(loc, sum, nnz_count, zero);
                // Value tmp_max = builder.create<memref::LoadOp>(loc, max_nnz, zero);
                // Value is_row_nnz_greater = builder.create<CmpIOp>(loc, CmpIPredicate::ugt, row_nnz, tmp_max);
                // Value sum = builder.create<AddIOp>(loc, row_nnz, tmp_count);
                return;
            });

        // StructType 
        // Value crd_struct = rewriter.create<sparlay::StructConstructOp>(loc, )
        rewriter.eraseOp(op);
        // LLVM_DEBUG(llvm::dbgs() << "hello!\n"); 
        return success();
    }
}; 

//===----------------------------------------------------------------------===//
// RewritePatterns: Compress operations
//===----------------------------------------------------------------------===//

class CompressOpLowering : public OpConversionPattern<sparlay::CompressOp> {
public:
    using OpConversionPattern<sparlay::CompressOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::CompressOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        // Value output = op->getOperand(0);
        // Value input_A = op->getOperand(1);
        // Value input_B = op->getOperand(2);
        // StringRef target = op.target();
        // StringRef pattern = op.pattern();

        // if (target == "CPU" && pattern == "inner") {


        // } else
        //     LLVM_DEBUG(llvm::dbgs() << "Target or pattern not supported yet.\n");
    }
};

//===----------------------------------------------------------------------===//
// RewritePatterns: Multiply operations
//===----------------------------------------------------------------------===//

class MultiplyOpLowering : public OpConversionPattern<sparlay::MultiplyOp> {
public:
    using OpConversionPattern<sparlay::MultiplyOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::MultiplyOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        // Location loc = op->getLoc();
        // Value output = op->getOperand(0);
        // Value input_A = op->getOperand(1);
        // Value input_B = op->getOperand(2);
        // StringRef target = op.target();
        // StringRef pattern = op.pattern();

        // if (target == "CPU" && pattern == "inner") {


        // } else
        //     LLVM_DEBUG(llvm::dbgs() << "Target or pattern not supported yet.\n");
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
                        vector::VectorDialect, linalg::LinalgDialect,
                        StandardOpsDialect>();
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
                           vector::VectorDialect, linalg::LinalgDialect,
                           StandardOpsDialect>();

    // We also define the Sparlay dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Sparlay operations that don't want
    // to lower as `legal`.
    target.addIllegalDialect<sparlay::SparlayDialect>();
    target.addLegalOp<sparlay::CompressOp>();
    target.addLegalOp<sparlay::StructAccessOp>();
    target.addLegalOp<sparlay::StructConstructOp>();
    target.addLegalOp<sparlay::FooOp>();
    target.addLegalOp<linalg::FillOp>(); //?

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Sparlay operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<NewOpLowering, PackOpLowering,
                 CompressOpLowering, MultiplyOpLowering>(&getContext());
    // patterns.add<PackOpLowering>(&getContext());
    // patterns.add<MultiplyOpLowering>(&getContext());
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
