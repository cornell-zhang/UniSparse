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

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"
#include "IR/SparlayTypes.h"
#include "Transforms/Passes.h"
#include "Eigen/Dense"

#include <cstdio>
#include <cstring>
#include <tuple>

using namespace mlir;
using namespace sparlay;

#define DEBUG_TYPE "lower-format-conversion"

namespace {
#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

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
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
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
        Type inputType = fileName.getType();
        auto resType = op->getResult(0).getType().dyn_cast<StructType>();

        // if (resType.isa<StructType>())
        //     LLVM_DEBUG(llvm::dbgs() << "is a struct type\n");
        
        llvm::ArrayRef<mlir::Type> elmTypes = resType.getElementTypes();
        Type crdType = elmTypes.front();
        Type dataType = elmTypes.back();
        auto resDimSizes = resType.getDimSizes();
        uint64_t resSize = resDimSizes.size();

        func::CallOp tensorOp;
        func::CallOp indicesOp[resSize];
        func::CallOp valueOp;

        auto indexTp = rewriter.getIndexType();
        Type idxResType = MemRefType::get({ShapedType::kDynamicSize}, indexTp);
        // auto f32Tp = rewriter.getF32Type();
        // Type valResType = MemRefType::get({ShapedType::kDynamicSize}, f32Tp);
        StringRef readTensorName =  "readSparseCoordinate";
        StringRef idxFuncName = "getTensorIndices";
        StringRef valFuncName = "getTensorValues";

        SmallVector<Value, 1> readParams;
        readParams.push_back(fileName);
        tensorOp = rewriter.create<func::CallOp>(loc, inputType, 
            getFunc(op, readTensorName, inputType, readParams, /*emitCInterface=*/false),
            readParams); //
        
        for (unsigned i = 0; i < resSize; i++) {
            SmallVector<Value, 3> idxParams;
            idxParams.push_back(tensorOp.getResult(0));
            idxParams.push_back(
                rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i)));
            indicesOp[i] = rewriter.create<func::CallOp>(loc, idxResType, 
                getFunc(op, idxFuncName, idxResType, idxParams, /*emitCInterface=*/true),
                idxParams);
        }
        SmallVector<Value, 3> valParams;
        valParams.push_back(tensorOp.getResult(0));
        valueOp = rewriter.create<func::CallOp>(loc, dataType, 
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

class fromFileOpLowering : public OpConversionPattern<sparlay::fromFileOp> {
public:
    using OpConversionPattern<sparlay::fromFileOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::fromFileOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        
        Value fileName = op->getOperand(0);
        Type inputType = fileName.getType();

        func::CallOp readOp;

        StringRef funcName =  "sptFromFile";

        SmallVector<Value, 1> readParams;
        readParams.push_back(fileName);
        readOp = rewriter.create<func::CallOp>(loc, inputType, 
            getFunc(op, funcName, inputType, readParams, /*emitCInterface=*/true),
            readParams);

        rewriter.replaceOp(op, readOp.getResult(0));
        return success();
    }
};

SparlayEncodingAttr getSparlayEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<SparlayEncodingAttr>();
  return nullptr;
}


typedef Eigen::Matrix<double, 2, 2> Matrix2f;
typedef Eigen::Matrix<int, 2, 2> Matrix2i;

Matrix2f toMatrix(const AffineMap& crdMap) {
    assert(crdMap.getNumDims() == 2);
    Matrix2f ret;
    ret(0,0)=ret(0,1)=ret(1,0)=ret(1,1) = 0;
    llvm::SmallBitVector projectedDims(2, 0);
    projectedDims[1] = 1;
    std::cerr << projectedDims.size() << std::endl;
    auto proj1 = getProjectedMap(crdMap, projectedDims);
    std::cerr << "done1" << std::endl;
    int curDim = 0;
    for (AffineExpr expr : proj1.getResults()) {
        expr.dump();
        if (expr != getAffineConstantExpr(0, proj1.getContext())) {
            if (expr == getAffineDimExpr(0, proj1.getContext())) ret(curDim, 0) = 1;
            else ret(curDim, 0) = -1;
        }
        curDim++;
    }
    projectedDims = llvm::SmallBitVector(2, 0);
    projectedDims[0] = 1;
    auto proj0 = getProjectedMap(crdMap, projectedDims);
    curDim = 0;
    for (AffineExpr expr: proj0.getResults()) {
        expr.dump();
        if (expr != getAffineConstantExpr(0, proj0.getContext())) {
            if (expr == getAffineDimExpr(0, proj0.getContext())) ret(curDim, 1) = 1;
            else ret(curDim, 1) = -1;
        }
        curDim++;
    }
    // std::cerr << ret << std::endl;
    // std::cerr << "=======================" << std::endl;
    return ret;
}

Matrix2i toIntMatrix(const Matrix2f& M) {
    Matrix2i ret;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int curVal = (int)floor(M(i,j)+1e-4);
            assert(M(i,j) < curVal + 1e-4);
            assert(M(i,j) > curVal-1e-4);
            ret(i,j) = curVal;
        }
    }
    return ret;
}

enum ConversionOpType {
    NoOp,
    TileMerge,
    TileSplit,
    Move
};

struct GeneralConversionOp {
    int type;
    std::string name;
    std::vector<int> args;
    GeneralConversionOp(int _type = 0, std::string _name = "", std::vector<int> _args = {}) {
        type = _type, name = _name, args = _args;
    }
    void Print(std::ostream& mout) {
        switch(type) {
            case 1:
                mout << "TileMerge(" << args[0] << "," << args[1] << ")" << std::endl;
            break;
            case 2:
                mout << "TileSplit(" << args[0] << "," << args[1] << ")" << std::endl;
            break;
            case 3:
                mout << "Move(" << args[0] << "," << args[1] << ")" << std::endl;
            break;
            default:
                mout << "null()" << std::endl;
        }
    }
};

std::tuple<AffineMap, std::vector<GeneralConversionOp> > rewriteTileAndStashOp(const AffineMap& crdMap, bool isSplit) {
    std::cerr << "Enter Rewrite" << std::endl;
    std::vector<GeneralConversionOp> Ops;
    std::vector<AffineExpr> newExprs;
    std::vector<int> pendingMerge;
    std::vector<AffineExpr> exprs = crdMap.getResults();
    std::vector<bool> vis;
    std::vector<bool> needPush;
    vis.resize(exprs.size(), 0);
    needPush.resize(exprs.size(), 0);
    bool hasChanged = 0;
    do {
        hasChanged = 0;
        for (int i = 0; i < exprs.size(); ++i) {
            if (vis[i]) continue;
            if (exprs[i].getKind() == AffineExprKind::Mod || exprs[i].getKind() == AffineExprKind::FloorDiv) {
                auto binExpr = exprs[i].dyn_cast<AffineBinaryOpExpr>();
                assert(binExpr);
                auto LHS = binExpr.getLHS();
                auto RHS = binExpr.getRHS();
                assert(RHS.isSymbolicOrConstant());
                LHS.dump(), RHS.dump();
                auto targetKind = (exprs[i].getKind() == AffineExprKind::Mod ? AffineExprKind::FloorDiv : AffineExprKind::Mod);
                for (int j = i+1; j < exprs.size(); ++j) {
                    if (vis[j]) continue;
                    if (exprs[j].getKind() == targetKind) {
                        auto _binExpr = exprs[j].dyn_cast<AffineBinaryOpExpr>();
                        auto _LHS = _binExpr.getLHS();
                        auto _RHS = _binExpr.getRHS();
                        assert(_RHS.isSymbolicOrConstant());
                        if (LHS == _LHS && RHS == _RHS) {
                            if (targetKind == AffineExprKind::Mod) {
                                Ops.push_back(GeneralConversionOp(Move, "", (isSplit ? std::vector<int>({i, j-1}) : std::vector<int>({j, i+1}))));
                                hasChanged = 1;
                                if (isSplit) {
                                    auto svExpr = exprs[i];
                                    for (int k = i+1; k <= j-1; ++k) exprs[k-1] = exprs[k], vis[k-1] = vis[k];
                                    exprs[j-1] = svExpr;
                                    vis[j] = vis[j-1] = 1;
                                } else {
                                    auto svExpr = exprs[j];
                                    for (int k = j-1; k >= i+1; --k) exprs[k+1] = exprs[k], vis[k+1] = vis[k];
                                    exprs[i+1] = svExpr;
                                    vis[i] = vis[i+1] = 1;
                                }
                            } else {
                                hasChanged = 1;
                                Ops.push_back(GeneralConversionOp(Move, "", (isSplit ? std::vector<int>({i, j}) : std::vector<int>({j, i}))));
                                if (isSplit) {
                                    auto svExpr = exprs[i];
                                    for (int k = i+1; k <= j; ++k) exprs[k-1] = exprs[k], vis[k-1] = vis[k];
                                    exprs[j] = svExpr;
                                    vis[j-1] = vis[j] = 1;
                                } else {
                                    auto svExpr = exprs[j];
                                    for (int k = j-1; k >= i; --k) exprs[k+1] = exprs[k], vis[k+1] = vis[k];
                                    exprs[i] = svExpr;
                                    vis[i] = vis[i+1] = 1;
                                }
                            }
                            break;
                        }
                    }
                }
                for (int j = 0; j < exprs.size(); ++j) {
                    exprs[j].dump();
                }
            }
        }
    } while (hasChanged);

    for (int i = 0; i < exprs.size(); ++i) {
        if (exprs[i].getKind() == AffineExprKind::FloorDiv) {
            assert(i != exprs.size()-1);
            assert(exprs[i+1].getKind() == AffineExprKind::Mod);
            assert(exprs[i].dyn_cast<AffineBinaryOpExpr>().getLHS() == exprs[i+1].dyn_cast<AffineBinaryOpExpr>().getLHS());
            assert(exprs[i].dyn_cast<AffineBinaryOpExpr>().getRHS() == exprs[i+1].dyn_cast<AffineBinaryOpExpr>().getRHS());
            auto divNum = exprs[i].dyn_cast<AffineBinaryOpExpr>().getRHS().dyn_cast<AffineConstantExpr>().getValue();
            assert(divNum < 5LL);
            Ops.push_back(GeneralConversionOp(TileMerge, "", {i, (int)divNum}));
            newExprs.push_back(exprs[i].dyn_cast<AffineBinaryOpExpr>().getLHS());
        } else if (exprs[i].getKind() != AffineExprKind::Mod) {
            newExprs.push_back(exprs[i]);
        }
    }
    auto newCrdMap = AffineMap::get(crdMap.getNumDims(), 0, newExprs, crdMap.getContext());
    std::cerr << "Leave Rewrite" << std::endl;
    return std::make_tuple(newCrdMap, Ops);
}

class ConvertOpLowering : public OpConversionPattern<sparlay::ConvertOp> {
public:
    using OpConversionPattern<sparlay::ConvertOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::ConvertOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op.getLoc();
        Type resType = op.getType();
        Value src = adaptor.getOperands()[0];
        Type srcType = src.getType();
        auto encSrc = getSparlayEncoding(op->getOperand(0).getType());
        auto encDst = getSparlayEncoding(resType);

        //handle swap only (quick round-about)
        auto srcCrd = encSrc.getCrdMap();
        auto dstCrd = encDst.getCrdMap();

        if (srcCrd.getNumResults() > 5) {
            std::cerr << "Too many source format dimensions!" << std::endl;
            assert(0);
        } else if (dstCrd.getNumResults() > 5) {
            std::cerr << "Too many target format dimensions!" << std::endl;
            assert(0);
        }

        auto srcSecond = encSrc.getCompressMap();
        auto dstSecond = encDst.getCompressMap();

        auto srcTrim = srcSecond.getTrimIndex();
        auto dstTrim = dstSecond.getTrimIndex();
        auto srcFuse = srcSecond.getFuseIndex();
        auto dstFuse = dstSecond.getFuseIndex();

        StringRef fuseName = "sptFuse";
        StringRef separateName = "sptSeparate";
        StringRef trimName = "sptTrim";
        StringRef growName = "sptGrow";
        StringRef swapName = "sptSwap";
        StringRef subName = "sptSub";
        StringRef addName = "sptAdd";
        StringRef negName = "sptNeg";
        StringRef vectorizeName = "sptVectorize";
        StringRef devectorizeName = "sptDevectorize";
        StringRef tileMergeName = "sptTileMerge";
        StringRef tileSplitName = "sptTileSplit";
        StringRef moveName = "sptMove"; //partial sort
        // StringRef lazySortName = "sptLazySort";

        Type prevType = srcType;
        Value prevRes = src;

        //generate function that has only one return value
        auto genFunc1R = [&](const StringRef& name, std::vector<Value> params) {
            auto prevOp = rewriter.create<func::CallOp>(loc, prevType,
                getFunc(op, name, prevType, params, true),
                params
            );
            prevType = prevOp.getType(0);
            prevRes = prevOp.getResult(0);
        };

        mlir::arith::ConstantOp Const[6];
        for (int i = 0; i < 6; ++i) {
            Const[i] = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(i));
        }

        auto genFuncFromOp = [&](
            const GeneralConversionOp& gco
        ) {
            switch (gco.type) {
                case TileMerge: //tiling: merge
                    assert(gco.args.size() == 2);
                    assert(gco.args[0] < 6 && gco.args[0] >= 0);
                    assert(gco.args[1] < 6 && gco.args[1] >= 0);
                    genFunc1R(tileMergeName, {prevRes, Const[gco.args[0]], Const[gco.args[1]]});
                break;
                case TileSplit: //tiling: split
                    assert(gco.args.size() == 2);
                    assert(gco.args[0] < 6 && gco.args[0] >= 0);
                    assert(gco.args[1] < 6 && gco.args[1] >= 0);
                    genFunc1R(tileSplitName, {prevRes, Const[gco.args[0]], Const[gco.args[1]]});
                break;
                case Move: //move
                    assert(gco.args.size() == 2);
                    assert(gco.args[0] < 6 && gco.args[0] >= 0);
                    assert(gco.args[1] < 6 && gco.args[1] >= 0);
                    if (gco.args[0] < gco.args[1]) {
                        for (int i = gco.args[0] + 1; i <= gco.args[1]; ++i) {
                            genFunc1R(moveName, {prevRes, Const[i], Const[i-1]});
                        }
                    } else {
                        genFunc1R(moveName, {prevRes, Const[gco.args[0]], Const[gco.args[1]]});
                    }
                break;
                default:
                    assert(0);
                break;
            }
        };

        bool fuse_vis[10] = {0};
        for (auto ele: srcFuse) {
            fuse_vis[ele] = 1;
        }
        
        int src_mn_trim = 1000, src_mx_trim=-1;
        for (auto ele: srcTrim) {
            src_mn_trim = std::min(src_mn_trim, ele);
            src_mx_trim = std::max(src_mx_trim, ele);
        }
        int dst_mn_trim = 1000, dst_mx_trim=-1;
        for (auto ele: dstTrim) {
            dst_mn_trim = std::min(dst_mn_trim, ele);
            dst_mx_trim = std::max(dst_mx_trim, ele);
        }
        
        //devectorize first, could be optimized.
        if (src_mx_trim < (srcCrd.getNumResults()-1)) {
            int delta = srcCrd.getNumResults()-1-src_mx_trim;
            assert(delta <= 2);
            //TODO: FIXME: change devectorize to support matrix devectorization
            genFunc1R(devectorizeName, {prevRes});
        }

        bool need_move[10] = {0};
        memset(need_move, 0, sizeof(need_move));

        //handle coordinate remapping
        if (srcCrd != dstCrd) {
            assert(src_mn_trim != 1000);
            if (src_mn_trim != 0) {
                src_mn_trim = 0;
                genFunc1R(trimName, {prevRes, Const[0]});
            }
            for (auto ele: srcFuse) {
                if (!fuse_vis[ele]) continue;
                fuse_vis[ele] = 0;
                genFunc1R(separateName, {prevRes, Const[ele]});
            }
            
            AffineMap flatSrcCrd, flatDstCrd;
            std::vector<GeneralConversionOp> removeSrcTiling, removeDstTiling;
            std::tie(flatSrcCrd, removeSrcTiling) = rewriteTileAndStashOp(srcCrd, 0);
            std::tie(flatDstCrd, removeDstTiling) = rewriteTileAndStashOp(dstCrd, 1);
            for (const auto& ele: removeSrcTiling) {
                if (ele.type != TileMerge) {
                    genFuncFromOp(ele);
                }
            }
            {
                int st = 0;
                while (st < removeSrcTiling.size() && removeSrcTiling[st].type != TileMerge) st++;
                for (int i = removeSrcTiling.size()-1; i >= st; --i) {
                    genFuncFromOp(removeSrcTiling[i]);
                }
            }
            Matrix2f dstM = toMatrix(flatDstCrd);
            Matrix2f srcM = toMatrix(flatSrcCrd);

            flatSrcCrd.dump();
            flatDstCrd.dump();

            // trivial Gaussian Elimination with functiion generation
            // Calculate M: (range(dstM)->range(srcM))
            Matrix2f inverse_dstM = dstM.inverse();
            std::cerr << "inverse destination coordinate map = " << std::endl;
            std::cerr << inverse_dstM << std::endl;
            std::cerr << srcM * inverse_dstM << std::endl;
            Matrix2i crdRemapMap = toIntMatrix(srcM * inverse_dstM);

            auto genOpFromAffineMap = [&](Matrix2i& M) {
                std::cerr << M << std::endl;
                for (int i = 0; i < 2; ++i) {
                    if (M(i,i) == 0) {
                        int st;
                        for (st = i+1; st < 2; ++st) {
                            if (M(st,i) == 1) break;
                        }
                        if (st == 2) {
                            for (st = i+1; st < 2; ++st) {
                                if (M(st,i) == -1) break;
                            }
                        }
                        genFunc1R(swapName, {prevRes, Const[i], Const[st]});
                        need_move[i] = 1;
                        for (int j = i; j < 2; ++j) {
                            std::swap(M(st,j), M(i,j));
                        }
                    }
                    // assert(M(i,i) == 1);
                    for (int row = i+1; row < 2; ++row) {
                        if (M(row, i) == -M(i,i)) {
                            genFunc1R(addName, {prevRes, Const[row], Const[i]});
                            for (int j = i; j < 2; ++j) {
                                M(row, j) += M(i,j);
                            }
                        } else if (M(row,i) == M(i,i)) {
                            genFunc1R(subName, {prevRes, Const[row], Const[i]});
                            for (int j = i; j < 2; ++j) {
                                M(row, j) -= M(i,j);
                            }
                        }
                    }
                }
                std::cerr << M << std::endl;
                for (int i = 1; i >= 0; --i) {
                    if (M(i,i) != 1) {
                        std::cerr << M(i,i) << std::endl;
                        assert(M(i,i) == -1);
                        genFunc1R(negName, {prevRes, Const[i]});
                        need_move[i] = 1;
                        for (int j = i; j < 2; ++j) {
                            M(i,j) = -M(i,j);
                        }
                    }
                    for (int row = i-1; row >= 0; --row) {
                        if (M(row, i) == -1) {
                            for (int j = row; j <= i; ++j) {
                                need_move[j] = 1;
                            }
                            genFunc1R(addName, {prevRes, Const[row], Const[i]});
                        } else if (M(row, i) == 1) {
                            for (int j = row; j <= i; ++j) {
                                need_move[j] = 1;
                            }
                            genFunc1R(subName, {prevRes, Const[row], Const[i]});
                        }
                        M(row, i) = 0;
                    }
                }
            };

            std::cerr << "crdRemapMap = " << std::endl << crdRemapMap << std::endl;

            genOpFromAffineMap(crdRemapMap);

            int pt = removeDstTiling.size()-1;
            while (pt >= 0 && removeDstTiling[pt].type == TileMerge) pt--;
            pt++;
            for (int i = removeDstTiling.size()-1; i >= 0; --i) {
                auto& ele = removeDstTiling[i];
                assert(ele.type <= 3);
                assert(ele.type != TileSplit);
                if (ele.type == TileMerge) {
                    ele.type = TileSplit;
                    assert(ele.args.size() == 2);
                    ele.args[0] = ele.args[0] - (i-pt);
                    for (int j = 0; j < ele.args[0]; ++j) {
                        if (need_move[j]) {
                            need_move[j] = 0;
                            genFunc1R(moveName, {prevRes, Const[j], Const[j]});
                        }
                    }
                    genFuncFromOp(ele);
                } else if (ele.type == Move) {
                    assert(ele.args.size() == 2);
                    std::swap(ele.args[0], ele.args[1]);
                    genFuncFromOp(ele);
                } else {
                    std::cerr << "Should not happen!" << std::endl;
                    assert(0);
                }
            }
        }
        if (dst_mx_trim < dstCrd.getNumResults()-1) {
            need_move[dstCrd.getNumResults()-1] = 0;
        }
        for (auto ele: dstFuse) {
            if (!fuse_vis[ele]) {
                for (int i = 0; i < 10; ++i) {
                    if (need_move[i]) {
                        //TODO: FIXME: change the function call of move so that we don't need to move almost all the levels
                        genFunc1R(moveName, {prevRes, Const[i], Const[i]});
                        need_move[i] = 0;
                    }
                }
                break;
            }
        }

        for (auto ele: dstFuse) {
            if (!fuse_vis[ele]) {
                genFunc1R(fuseName, {prevRes, Const[ele]});
            } else {
                fuse_vis[ele] = 0;
            }
        }
        for (auto ele: srcFuse) {
            if (fuse_vis[ele]) {
                fuse_vis[ele] = 0;
                genFunc1R(separateName, {prevRes, Const[ele]});
            }
        }

        if (dst_mn_trim < src_mn_trim) {
            genFunc1R(trimName, {prevRes, Const[dst_mn_trim]});
        } else if (dst_mn_trim > src_mn_trim) {
            genFunc1R(growName, {prevRes, Const[dst_mn_trim-1]});
        }
        if (dst_mx_trim < dstCrd.getNumResults()-1) {
            assert(dst_mx_trim == dstCrd.getNumResults()-2);
            for (int i = 0; i < dstCrd.getNumResults()-1; ++i) {
                if (need_move[i]) {
                    genFunc1R(moveName, {prevRes, Const[i], Const[i]});
                    need_move[i] = 0;
                }
            }
        }
        if (dst_mx_trim < dstCrd.getNumResults()-1) {
            assert(dst_mx_trim == dstCrd.getNumResults()-2);
            genFunc1R(vectorizeName, {prevRes, Const[dstCrd.getNumResults()-1]});
        }
        for (int i = 0; i < dstCrd.getNumResults(); ++i) {
            if (need_move[i]) {
                genFunc1R(moveName, {prevRes, Const[i], Const[i]});
                need_move[i] = 0;
            }
        }
        for (int i = 0; i < 10; ++i) {
            assert(need_move[i] == 0);
        }
        rewriter.replaceOp(op, prevRes);
        return success();
    }
};

class printStorageOpLowering : public OpConversionPattern<sparlay::printStorageOp> {
    using OpConversionPattern<sparlay::printStorageOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::printStorageOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        
        Value candValue = adaptor.getOperands()[0];
        func::CallOp printOp;

        StringRef funcName = "sptPrint";

        SmallVector<Value, 1> printParams;
        printParams.push_back(candValue);

        rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, funcName, llvm::None, printParams, /*emitCInterface=*/true),
            printParams);
        return success();
    }
};

class copyOpLowering : public OpConversionPattern<sparlay::copyOp> {
    using OpConversionPattern<sparlay::copyOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::copyOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Value candValue = adaptor.getOperands()[0];
        func::CallOp copyOp;
        StringRef funcName = "sptCopy";
        SmallVector<Value, 1> params;
        params.push_back(candValue);
        rewriter.replaceOpWithNewOp<func::CallOp>(op, candValue.getType(), 
            getFunc(op, funcName, candValue.getType(), params, /*emitCInterface=*/true),
            params);
        return success();
    }
};

class checkOpLowering : public OpConversionPattern<sparlay::checkOp> {
    using OpConversionPattern<sparlay::checkOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::checkOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Value candValue1 = adaptor.getOperands()[0];
        Value candValue2 = adaptor.getOperands()[1];
        func::CallOp checkOp;
        StringRef funcName = "sptCheck";
        SmallVector<Value, 2> params = {candValue1, candValue2};
        rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, funcName, llvm::None, params, /*emitCInterface=*/true),
            params);
        return success();
    }
};

class ticOpLowering : public OpConversionPattern<sparlay::ticOp> {
    using OpConversionPattern<sparlay::ticOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::ticOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        func::CallOp ticOp;
        StringRef funcName = "sptTic";
        SmallVector<Value> params = {};
        rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, funcName, llvm::None, params, /*emitCInterface=*/true),
            params);
        return success();
    }
};

class tocOpLowering : public OpConversionPattern<sparlay::tocOp> {
    using OpConversionPattern<sparlay::tocOp>::OpConversionPattern;
        LogicalResult 
        matchAndRewrite(sparlay::tocOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        func::CallOp tocOp;
        StringRef funcName = "sptToc";
        SmallVector<Value> params = {};
        rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, funcName, llvm::None, params, /*emitCInterface=*/true),
            params);
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
        Value zeroElm = rewriter.create<arith::ConstantOp>(loc, inputElmTp, rewriter.getZeroAttr(inputElmTp));

        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
        // Value dim = rewriter.create<memref::DimOp>(loc, input, zero);

        SmallVector<Value> lb_outer, lb, lb_orig;
        SmallVector<Value> hb_outer, hb, hb_orig;
        SmallVector<Value> step_outer, step, step_orig;
        for (unsigned i = 0; i < shape.size(); i++) {
            Value dimSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(shape[i]));
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
        Value reduceDimSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(shape[reduceDimValue]));
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
                Value not_zero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, 
                                    elm, zeroElm);
                builder.create<scf::IfOp>(loc, not_zero, [&](OpBuilder &b, Location loc) {
                    Value old_nnz = b.create<memref::LoadOp>(loc, nnz_per_row, outer_ivs);
                    Value new_nnz = b.create<arith::AddIOp>(loc, old_nnz, one);
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
                Value sum = builder.create<arith::AddIOp>(loc, row_nnz, tmp_count);
                builder.create<memref::StoreOp>(loc, sum, nnz_count, zero);
                Value tmp_max = builder.create<memref::LoadOp>(loc, max_nnz, zero);
                Value is_row_nnz_greater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, row_nnz, tmp_max);
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
                Value reorderedDimSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(shape[reorderedDimPos]));
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
                    Value valid_idx = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, reordered_idx, reduceDimSize);
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
                        Value len_count_sum = b.create<arith::AddIOp>(loc, len_count_val, one);
                        b.create<memref::StoreOp>(loc, len_count_sum, len_count, zero);
                        b.create<scf::YieldOp>(loc, ValueRange{});
                    });
                } else if (padding == "zero") {

                } else 
                    LLVM_DEBUG(llvm::dbgs() << "The padding option in PackOp can only support 'none' and 'zero' now.");
            
                // builder.create<memref::StoreOp>(loc, sum, nnz_count, zero);
                // Value tmp_max = builder.create<memref::LoadOp>(loc, max_nnz, zero);
                // Value is_row_nnz_greater = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, row_nnz, tmp_max);
                // Value sum = builder.create<arith::AddIOp>(loc, row_nnz, tmp_count);
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
        Value input = op->getOperand(0);
        Value output = op->getResult(0);
        auto compressDim = op.compress_dim();
        // AffineMap storageOrder = op.storage_order();

        auto inputType = input.getType().dyn_cast<StructType>();
        auto outputType = output.getType().dyn_cast<StructType>();

        llvm::ArrayRef<mlir::Type> inputElmTypes = inputType.getElementTypes();
        llvm::ArrayRef<mlir::Type> outputElmTypes = outputType.getElementTypes();
        int64_t compressDimValue = compressDim.getSExtValue();
        Type inputCrdType = inputElmTypes.front();
        Type inputDataType = inputElmTypes.back();
        llvm::ArrayRef<int64_t> inputDimSizes = inputType.getDimSizes();
        uint64_t inputSize = inputDimSizes.size();
        Type outputPtrType = outputElmTypes[0];
        Type outputCrdType = outputElmTypes[1];
        // Type outputValType = outputElmTypes[2];
        auto indexTp = rewriter.getIndexType();
        Type idxMemRefType = MemRefType::get({ShapedType::kDynamicSize}, indexTp);

        // compose the new crd struct 
        Value i0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
        Value i1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
        Value crd_old = rewriter.create<sparlay::StructAccessOp>(loc, inputCrdType, input, 0);
        Value val_old = rewriter.create<sparlay::StructAccessOp>(loc, inputDataType, input, 1);
        std::vector<Value> crdArray;
        for (uint64_t i = 0; i < inputSize; i++) {
            // Value constI = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(i));
            crdArray.push_back(rewriter.create<sparlay::StructAccessOp>(loc, idxMemRefType, crd_old, i));
        }
        Value crd_new = rewriter.create<sparlay::StructConstructOp>(loc, outputCrdType, 
            llvm::makeArrayRef(crdArray).take_back(inputSize - compressDimValue));

        // compose the ptr struct
        // %ptr = memref.alloca() : memref<4xindex>
        int64_t ptrSizeVal = 1;
        for (int64_t i = 0; i < compressDimValue; i++) {
            ptrSizeVal = ptrSizeVal * inputDimSizes[i];
        }
        ptrSizeVal += 1;
        Value ptrSize = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(ptrSizeVal));
        MemRefType dynamicPtrType = MemRefType::get(-1, indexTp);
        Value ptr = rewriter.create<memref::AllocOp>(loc, dynamicPtrType, ptrSize);

        // memref.store %i0, %ptr[%i0] : memref<4xindex>
        // ValueRange idx0 = llvm::makeArrayRef(i0);
        rewriter.create<memref::StoreOp>(loc, i0, ptr, i0);
        Value crd_dim = rewriter.create<memref::DimOp>(loc, crdArray.front(), i0);

        // %ptr_dim = memref.dim %ptr, %i0 : memref<4xindex>
        // Value ptr_dim = rewriter.create<memref::DimOp>(loc, ptr, i0);

        // scf.for
        SmallVector<Value, 1> lb, hb, step;
        lb.push_back(i1);
        hb.push_back(ptrSize);
        step.push_back(i1);
        scf::buildLoopNest(rewriter, loc, lb, hb, step, 
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                SmallVector<Type, 3> resTypes;
                SmallVector<Value, 3> initArgs;
                resTypes.push_back(indexTp);
                initArgs.push_back(i0);
                auto whileOp = rewriter.create<scf::WhileOp>(loc, resTypes, initArgs);

                // The before block of the while loop.
                Block *before = rewriter.createBlock(&whileOp.getBefore(), {}, resTypes); 
                rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
                // %cond1 = cmpi ult, %arg1, %dim : index
                Value cond1 = builder.create<arith::CmpIOp>(whileOp.getLoc(), arith::CmpIPredicate::ult, before->getArguments()[0], crd_dim);
                // %crd_val = memref.load %crd_0[%arg1] : memref<7xindex>
                Value crd_val = rewriter.create<memref::LoadOp>(whileOp.getLoc(), crdArray.front(), before->getArgument(0));
                // %cond2 = cmpi ult, %crd_val, %arg0 : index
                Value cond2 = builder.create<arith::CmpIOp>(whileOp.getLoc(), arith::CmpIPredicate::ult, crd_val, ivs[0]);
                // %cond = and %cond1, %cond2 : i1
                Value cond = builder.create<arith::AndIOp>(whileOp.getLoc(), cond1, cond2);
                // scf.condition (%cond) %arg1 : index
                rewriter.create<scf::ConditionOp>(whileOp.getLoc(), cond, before->getArguments());

                // ----------------Please revise the logic for general purpose ---------
                // Value i5 = rewriter.create<arith::ConstantOp>(whileOp.getLoc(), rewriter.getIndexAttr(5));
                // Value isLessThanFive = rewriter.create<arith::CmpIOp>(whileOp.getLoc(), 
                //     arith::CmpIPredicate::ult, before->getArgument(0), i5);
                // rewriter.create<scf::ConditionOp>(whileOp.getLoc(), isLessThanFive, before->getArguments());
                
                // The after block of the while loop.
                Block *after = rewriter.createBlock(&whileOp.getAfter(), {}, resTypes);
                rewriter.setInsertionPointToStart(&whileOp.getAfter().front());

                // %sum = addi %arg2, %i1 : index
                Value sum = builder.create<arith::AddIOp>(whileOp.getLoc(), after->getArgument(0), i1);
                // scf.yield %sum : index
                rewriter.create<scf::YieldOp>(whileOp.getLoc(), ValueRange({sum}));

                rewriter.setInsertionPointAfter(whileOp);
                // memref.store %next_sum, %ptr[%arg0] : memref<4xindex>
                builder.create<memref::StoreOp>(loc, whileOp.getResult(0), ptr, ivs);
                return;
            });

        // compose the compressed struct
        Value ptr_new = rewriter.create<sparlay::StructConstructOp>(loc, outputPtrType, ptr);
        rewriter.replaceOpWithNewOp<sparlay::StructConstructOp>(op, output.getType(), 
            ValueRange({ptr_new, crd_new, val_old}));
        // rewriter.eraseOp(op);
        return success();
        // DCE to remove redundant constant allocation - after finalizing lowering
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
        Location loc = op->getLoc();
        Value output = op->getResult(0);
        auto outputType = output.getType();
        Value input_A = op->getOperand(0);
        auto inputType_A = input_A.getType().dyn_cast<StructType>();
        Value input_B = op->getOperand(1);
    //   auto inputdense_vecType = input_B.getType();
        llvm::ArrayRef<int64_t> dimSizes = inputType_A.getDimSizes();
        llvm::ArrayRef<mlir::Type> inputElmTypes = inputType_A.getElementTypes();
//      llvm::ArrayRef<mlir::Type> inputElmTypes_B = inputType_B.getElementTypes();
//      llvm::ArrayRef<mlir::Type> outputElmTypes = output_0.getElementTypes();
      
      if (inputElmTypes.size() > 2) {
        Type inputPtrType = inputElmTypes[0];
        Type inputCrdType = inputElmTypes[1];
        Type inputValType = inputElmTypes[2];
    //      Type inputdense_vecType = inputElmTypes_B[0];
    //      Type outputType = outputElmTypes[0];
        Value ptr = rewriter.create<sparlay::StructAccessOp>(loc, inputPtrType, input_A, 0);
        auto ptrtype = ptr.getType().dyn_cast<StructType>();
        llvm::ArrayRef<mlir::Type> ptrElmtypes = ptrtype.getElementTypes();
        Type input_ptr_type = ptrElmtypes[0];
        Value ptr_memref = rewriter.create<sparlay::StructAccessOp>(loc, input_ptr_type, ptr, 0);
        Value crd = rewriter.create<sparlay::StructAccessOp>(loc, inputCrdType, input_A, 1);
        auto crdtype = crd.getType().dyn_cast<StructType>();
        llvm::ArrayRef<mlir::Type> crdElmtypes = crdtype.getElementTypes();
        Type input_crd_type = crdElmtypes[0];
        Value crd_memref = rewriter.create<sparlay::StructAccessOp>(loc, input_crd_type, crd, 0);
        Value val = rewriter.create<sparlay::StructAccessOp>(loc, inputValType, input_A, 2);
        StringRef call_spmv_name = "calculateCSRSpMV";
        SmallVector<Value, 4> readParams;
        readParams.push_back(ptr_memref);
        readParams.push_back(crd_memref);
        readParams.push_back(val);
        readParams.push_back(input_B);
        rewriter.replaceOpWithNewOp<func::CallOp>(op, outputType, 
            getFunc(op, call_spmv_name, outputType, readParams, /*emitCInterface=*/true),
            readParams);
      } else {
        Type inputCrdType = inputElmTypes[0];
        Type inputValType = inputElmTypes[1];
    //      Type inputdense_vecType = inputElmTypes_B[0];
    //      Type outputType = outputElmTypes[0];
        // Value ptr = rewriter.create<sparlay::StructAccessOp>(loc, inputPtrType, input_A, 0);
        // auto ptrtype = ptr.getType().dyn_cast<StructType>();
        // llvm::ArrayRef<mlir::Type> ptrElmtypes = ptrtype.getElementTypes();
        // Type input_ptr_type = ptrElmtypes[0];
        // Value ptr_memref = rewriter.create<sparlay::StructAccessOp>(loc, input_ptr_type, ptr, 0);

        Value crd = rewriter.create<sparlay::StructAccessOp>(loc, inputCrdType, input_A, 0);
        auto crdtype = crd.getType().dyn_cast<StructType>();
        llvm::ArrayRef<mlir::Type> crdElmtypes = crdtype.getElementTypes();
        Type input_crd_0_type = crdElmtypes[0];
        Type input_crd_1_type = crdElmtypes[1];
        Value crd_0_memref = rewriter.create<sparlay::StructAccessOp>(loc, input_crd_0_type, crd, 0);
        Value crd_1_memref = rewriter.create<sparlay::StructAccessOp>(loc, input_crd_1_type, crd, 1);
        
        Value val = rewriter.create<sparlay::StructAccessOp>(loc, inputValType, input_A, 1);
        StringRef call_spmv_name = "calculateCOOSpMV";
        SmallVector<Value, 4> readParams;
        // readParams.push_back(ptr_memref);
        readParams.push_back(crd_0_memref);
        readParams.push_back(crd_1_memref);
        readParams.push_back(val);
        readParams.push_back(input_B);
        for (unsigned i = 0; i < dimSizes.size(); i++) {
            readParams.push_back(rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(dimSizes[i])));
        }
        rewriter.replaceOpWithNewOp<func::CallOp>(op, outputType, 
            getFunc(op, call_spmv_name, outputType, readParams, /*emitCInterface=*/true),
            readParams);
      }
      return success();
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
public LowerFormatConversionBase<LowerFormatConversionPass> {
    // void getDependentDialects(DialectRegistry &registry) const override {
    //     registry.insert<scf::SCFDialect, memref::MemRefDialect, 
    //                     vector::VectorDialect, linalg::LinalgDialect,
    //                     arith::ArithmeticDialect, LLVM::LLVMDialect>();
    // }
    void runOnOperation() final;
};
}

void LowerFormatConversionPass::runOnOperation() {
    // auto function = getFunction();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `MemRef` and `Standard` dialects.
    target.addLegalDialect<scf::SCFDialect, memref::MemRefDialect,
                           vector::VectorDialect, linalg::LinalgDialect,
                           arith::ArithmeticDialect, LLVM::LLVMDialect, func::FuncDialect>();

    // We also define the Sparlay dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Sparlay operations that don't want
    // to lower as `legal`.
    target.addIllegalDialect<sparlay::SparlayDialect>();
    target.addLegalOp<sparlay::StructAccessOp>();
    target.addLegalOp<sparlay::StructConstructOp>();
    target.addLegalOp<linalg::FillOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Sparlay operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<NewOpLowering, PackOpLowering,
                 CompressOpLowering, MultiplyOpLowering, 
                 fromFileOpLowering, ConvertOpLowering, printStorageOpLowering,
                 checkOpLowering, copyOpLowering, ticOpLowering, tocOpLowering>(&getContext());
    // patterns.add<PackOpLowering>(&getContext());
    // patterns.add<MultiplyOpLowering>(&getContext());
    // LLVM_DEBUG(llvm::dbgs() << "Has the pattern rewrite applied?\n");

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    func::FuncOp curOp = getOperation();
    if (failed(
            applyPartialConversion(curOp, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::sparlay::createLowerFormatConversionPass() {
    return std::make_unique<LowerFormatConversionPass>();
}
