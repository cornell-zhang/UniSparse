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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/../../lib/Dialect/SparseTensor/Transforms/CodegenUtils.h"
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

static FlatSymbolRefAttr getFunc(Operation *op, StringRef name,
                                 TypeRange resultType, ValueRange operands,
                                 bool emitCInterface) {
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
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

static func::CallOp replaceOpWithFuncCall(RewriterBase &rewriter, Operation *op,
                                          StringRef name, TypeRange resultType,
                                          ValueRange operands,
                                          bool emitCInterface) {
  auto fn = getFunc(op, name, resultType, operands, emitCInterface);
  return rewriter.replaceOpWithNewOp<func::CallOp>(op, resultType, fn,
                                                   operands);
}

static func::CallOp createFuncCall(OpBuilder &builder, Operation *op,
                                   StringRef name, TypeRange resultType,
                                   ValueRange operands,
                                   bool emitCInterface) {
  auto fn = getFunc(op, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(op->getLoc(), resultType, fn, operands);
}

static Value genSparlayDimSizeCall(OpBuilder &builder, Operation *op,
                            SparlayEncodingAttr &enc, Value src,
                            int64_t idx) {
  // Permute the index according to an optional dimension ordering.
  if (AffineMap p = enc.getCrdMap())
    idx = p.getPermutedPosition(idx);
  // Generate the call.
  StringRef name = "sparseDimSize";
  SmallVector<Value, 2> params{src, sparse_tensor::constantIndex(builder, op->getLoc(), idx)};
  Type iTp = builder.getIndexType();
  return createFuncCall(builder, op, name, iTp, params, true).getResult(0);
}

static Value genSparlayAlloc(RewriterBase &rewriter, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{sz});
}

static Value genSparlayAlloca(OpBuilder &builder, Location loc, Value sz, Type tp) {
  auto memTp = MemRefType::get({ShapedType::kDynamicSize}, tp);
  return builder.create<memref::AllocaOp>(loc, memTp, ValueRange{sz});
}

static Value genSparlayAlloca(OpBuilder &builder, Location loc, unsigned sz, Type tp) {
  return genSparlayAlloca(builder, loc, sparse_tensor::constantIndex(builder, loc, sz), tp);
}

static Value genSparlayBuffer(OpBuilder &builder, Location loc, ValueRange values) {
  unsigned sz = values.size();
  assert(sz >= 1);
  Value buffer = genSparlayAlloca(builder, loc, sz, values[0].getType());
  for (unsigned i = 0; i < sz; i++) {
    Value idx = sparse_tensor::constantIndex(builder, loc, i);
    builder.create<memref::StoreOp>(loc, values[i], buffer, idx);
  }
  return buffer;
}

std::vector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType> inferDimLevelType(SparlayEncodingAttr enc, int64_t rank) {
  std::vector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType> lt;
  assert(enc);
//  std::cerr << "rank is " << rank << std::endl;
  auto crdmap = enc.getCrdMap();
  auto compress = enc.getCompressMap();
  auto trim = compress.getTrimIndex();
  auto merge = compress.getFuseIndex();
  std::vector<bool> trim_vec(rank, false); 
  int trim_to = trim[0];
  int trim_from = trim[1];
  assert(trim_to <= trim_from);
  for (int i = trim_to; i <= trim_from; i++) {
    trim_vec[i] = true;
  }
  std::vector<bool> merge_vec(rank, false);
  int end = merge[0];
  for(int i = 0; i <= end; i++) {
    merge_vec[i] = true;
  }
//  std::cerr << "start fill lt " << rank << std::endl;
  for(int64_t d = 0; d < rank; d++) {
    if(d == 0) {
      if(trim_vec[d]) {
        lt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
        continue;
      }
    } else if (trim_vec[d] && merge_vec[d-1]) {
      lt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Compressed);
      continue;
    } else {
      lt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Singleton);
      continue;
    }
    lt.push_back(sparse_tensor::SparseTensorEncodingAttr::DimLevelType::Dense);
  }
  return lt;
}

static void SparlaynewParams(OpBuilder &builder, SmallVector<Value, 8> &params, Operation *op, 
                             SparlayEncodingAttr &enc, ValueRange szs, int64_t rank, Value ptr = Value()) {
  Location loc = op->getLoc();
  std::vector<sparse_tensor::SparseTensorEncodingAttr::DimLevelType> dlt = inferDimLevelType(enc, rank);
//  std::cerr << "Finish infer DimLevelType " << std::endl;
  unsigned sz = dlt.size();
  // Sparsity annotations.
  SmallVector<Value, 4> attrs;
  for (unsigned i = 0; i < sz; i++)
    attrs.push_back(constantDimLevelTypeEncoding(builder, loc, dlt[i]));
  params.push_back(genSparlayBuffer(builder, loc, attrs));
// std::cerr << "Finish gen level attr buffer " << std::endl;
  // Dimension sizes array of the enveloping tensor. Useful for either
  // verification of external data, or for construction of internal data.
  params.push_back(genSparlayBuffer(builder, loc, szs));
//  std::cerr << "Finish gen dimension size " << std::endl;
  // Dimension order permutation array. This is the "identity" permutation by
  // default, or otherwise the "reverse" permutation of a given ordering, so
  // that indices can be mapped quickly to the right position.
  SmallVector<Value, 4> rev(sz);
  if (AffineMap p = enc.getCrdMap()) {
    for (unsigned i = 0; i < sz; i++)
      rev[p.getDimPosition(i)] = sparse_tensor::constantIndex(builder, loc, i);
  } else {
    for (unsigned i = 0; i < sz; i++)
      rev[i] = sparse_tensor::constantIndex(builder, loc, i);
  }
  params.push_back(genSparlayBuffer(builder, loc, rev));
//  std::cerr << "Finish gen reverse permutation " << std::endl;
  // Secondary and primary types encoding.
//  Type elemTp = stp.getElementType();
//  params.push_back(sparse_tensor::constantOverheadTypeEncoding(builder, loc, 32));
//  params.push_back(sparse_tensor::constantOverheadTypeEncoding(builder, loc, 32));
//  params.push_back(sparse_tensor::constantPrimaryTypeEncoding(builder, loc, elemTp));
  // User action.
//  params.push_back(constantAction(builder, loc, action));
  // Payload pointer.
  if (!ptr)
    ptr = builder.create<LLVM::NullOp>(loc, LLVM::LLVMPointerType::get(builder.getI8Type()));
  params.push_back(ptr);
}

static Value genSparlayNewCall(OpBuilder &builder, Operation *op,
                        ArrayRef<Value> params) {
  StringRef name = "newSparlayTensor";
  Type pTp = LLVM::LLVMPointerType::get(builder.getI8Type());
  return createFuncCall(builder, op, name, pTp, params, true).getResult(0);
}

//===----------------------------------------------------------------------===//
// RewritePatterns: New operations
//===----------------------------------------------------------------------===//

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
        for (int i = 0; i < (int)exprs.size(); ++i) {
            if (vis[i]) continue;
            if (exprs[i].getKind() == AffineExprKind::Mod || exprs[i].getKind() == AffineExprKind::FloorDiv) {
                auto binExpr = exprs[i].dyn_cast<AffineBinaryOpExpr>();
                assert(binExpr);
                auto LHS = binExpr.getLHS();
                auto RHS = binExpr.getRHS();
                assert(RHS.isSymbolicOrConstant());
                LHS.dump(), RHS.dump();
                auto targetKind = (exprs[i].getKind() == AffineExprKind::Mod ? AffineExprKind::FloorDiv : AffineExprKind::Mod);
                for (int j = i+1; j < (int)exprs.size(); ++j) {
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
                for (size_t j = 0; j < exprs.size(); ++j) {
                    exprs[j].dump();
                }
            }
        }
    } while (hasChanged);

    for (int i = 0; i < (int)exprs.size(); ++i) {
        if (exprs[i].getKind() == AffineExprKind::FloorDiv) {
            assert(i != (int)exprs.size()-1);
            assert(exprs[i+1].getKind() == AffineExprKind::Mod);
            assert(exprs[i].dyn_cast<AffineBinaryOpExpr>().getLHS() == exprs[i+1].dyn_cast<AffineBinaryOpExpr>().getLHS());
            assert(exprs[i].dyn_cast<AffineBinaryOpExpr>().getRHS() == exprs[i+1].dyn_cast<AffineBinaryOpExpr>().getRHS());
            auto divNum = exprs[i].dyn_cast<AffineBinaryOpExpr>().getRHS().dyn_cast<AffineConstantExpr>().getValue();
//            assert(divNum < 5LL);
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
                case TileMerge: { //tiling: merge
                    assert(gco.args.size() == 2);
                    assert(gco.args[0] < 6 && gco.args[0] >= 0);
//                    assert(gco.args[1] < 6 && gco.args[1] >= 0);
                    mlir::arith::ConstantOp factor_merge = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(gco.args[1]));
                    genFunc1R(tileMergeName, {prevRes, Const[gco.args[0]], factor_merge});
                    break; 
                }
                case TileSplit: {//tiling: split
                    assert(gco.args.size() == 2);
                    assert(gco.args[0] < 6 && gco.args[0] >= 0);
//                    assert(gco.args[1] < 6 && gco.args[1] >= 0);
                    mlir::arith::ConstantOp factor_split = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(gco.args[1]));
                    genFunc1R(tileSplitName, {prevRes, Const[gco.args[0]], factor_split});
                    break;
                }
                case Move:  { //move
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
                }
                default: {
                    assert(0);
                    break;
                }
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
        if ((unsigned)src_mx_trim < (srcCrd.getNumResults()-1)) {
            assert(src_mx_trim < srcCrd.getNumResults());
            genFunc1R(devectorizeName, {prevRes, Const[src_mx_trim+1]});
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
                while ((size_t)st < removeSrcTiling.size() && removeSrcTiling[st].type != TileMerge) st++;
                for (int i = removeSrcTiling.size()-1; i >= st; --i) {
                    genFuncFromOp(removeSrcTiling[i]);
                }
            }
            Matrix2f dstM = toMatrix(flatDstCrd);
            Matrix2f srcM = toMatrix(flatSrcCrd);

            flatSrcCrd.dump();
            flatDstCrd.dump();

            // trivial Gaussian Elimination with function generation
            // Calculate M: (range(dstM)->range(srcM))
            std::cerr << "dstM " << dstM << std::endl;
            std::cerr << "srcM " << srcM << std::endl;
            Matrix2f inverse_dstM = dstM.inverse();
            std::cerr << "inverse destination coordinate map = " << std::endl;
            std::cerr << inverse_dstM << std::endl;
            std::cerr << srcM * inverse_dstM << std::endl;
            Matrix2i crdRemapMap = toIntMatrix(srcM * inverse_dstM);

            auto genOpFromAffineMap = [&](Matrix2i& M) {
                std::cerr << "Enter genOpFromAffineMap" << std::endl;
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
        if ((unsigned)dst_mx_trim < dstCrd.getNumResults()-1) {
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
        if ((unsigned)dst_mx_trim < dstCrd.getNumResults()-1) {
//            assert((unsigned)dst_mx_trim == dstCrd.getNumResults()-2);
            for (unsigned i = 0; i < dstCrd.getNumResults()-1; ++i) {
                if (need_move[i]) {
                    genFunc1R(moveName, {prevRes, Const[i], Const[i]});
                    need_move[i] = 0;
                }
            }
            std::cerr << "dst_mx_trim is " << dst_mx_trim << std::endl;
            genFunc1R(vectorizeName, {prevRes, Const[dst_mx_trim+1]});
        }
        for (unsigned i = 0; i < dstCrd.getNumResults(); ++i) {
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

class StructAccessOpLowering: public OpConversionPattern<sparlay::StructAccessOp> {
public:
    using OpConversionPattern<sparlay::StructAccessOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(sparlay::StructAccessOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value inputPtr = adaptor.getOperands()[0];
        uint64_t index = op.index();
        std::vector<Value> params;
        params.push_back(inputPtr);
        params.push_back(rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(index)));
        auto ret = rewriter.create<func::CallOp>(loc, inputPtr.getType(),
            getFunc(op, "structAccess", inputPtr.getType(), params, true),
            params
        );
        rewriter.replaceOp(op, ret.getResult(0));
        return success();
    }
};

AffineMap rewriteTileGenWindow(const AffineMap& crdMap, Location loc, const sparlay::DecomposeOp& op, ConversionPatternRewriter &rewriter, Value& prevRes, Type& prevType) {
    std::vector<AffineExpr> exprs = crdMap.getResults();
    assert(exprs.size() <= (size_t)2);
    std::vector<AffineExpr> new_exprs = {};
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (exprs[i].getKind() == AffineExprKind::Mod || exprs[i].getKind() == AffineExprKind::FloorDiv) {
            auto binExpr = exprs[i].dyn_cast<AffineBinaryOpExpr>();
            assert(binExpr);
            auto LHS = binExpr.getLHS();
            auto RHS = binExpr.getRHS();
            assert(RHS.isSymbolicOrConstant());
            size_t curLv = i;
            if (LHS != getAffineDimExpr((unsigned)i, crdMap.getContext())) {
                assert(exprs.size() == (size_t)1);
                assert(LHS == getAffineDimExpr((unsigned)1, crdMap.getContext()));
                curLv = 1;
            }
            new_exprs.push_back(LHS);
            uint64_t _type = (exprs[i].getKind() == AffineExprKind::Mod);
            auto index = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(curLv));
            auto type = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(_type));
            auto val = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(RHS.dyn_cast<AffineConstantExpr>().getValue()));
            std::vector<Value> params = {prevRes, index, type, val};
            auto prevOp = rewriter.create<func::CallOp>(loc, prevType,
                getFunc(op, "spwTile", prevType, params, true),
                params
            );
            prevType = prevOp.getType(0);
            prevRes = prevOp.getResult(0);
        }
    }
    return AffineMap::get(crdMap.getNumDims(), 0, new_exprs, crdMap.getContext());
}

class DecompseOpLowering : public OpConversionPattern<sparlay::DecomposeOp> {
public:
  using OpConversionPattern<sparlay::DecomposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::DecomposeOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value inputThres = op->getOperand(1);
    AffineMap rmap = adaptor.rmap();

    std::vector<Value> params = {};
    auto prevOp = rewriter.create<func::CallOp>(loc, inputTensor.getType(),
        getFunc(op, "spwNew", inputTensor.getType(), params, true),
        params
    );
    auto prevType = prevOp.getType(0);
    auto prevRes = prevOp.getResult(0);
    auto assembleWindow = [&]() {
        rmap = rewriteTileGenWindow(rmap, loc, op, rewriter, prevRes, prevType);
        auto M = toIntMatrix(toMatrix(rmap));
        std::cerr << M << std::endl;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                if (M(i,j)) {
                    auto val = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(M(i,j)));
                    auto index_i = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
                    auto index_j = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(j));
                    params = {prevRes, index_i, index_j, val};
                    auto prevOp = rewriter.create<func::CallOp>(loc, prevType,
                        getFunc(op, "spwAssign", prevType, params, true),
                        params
                    );
                    prevType = prevOp.getType(0);
                    prevRes = prevOp.getResult(0);
                }
            }
        }
    };
    assembleWindow();
    params = {inputThres, inputTensor, prevRes};
    prevOp = rewriter.create<func::CallOp>(loc, inputTensor.getType(),
        getFunc(op, "sptSplit", inputTensor.getType(), params, true),
        params
    );
    prevRes = prevOp.getResult(0);
    rewriter.replaceOp(op, prevRes);
    return success();
  }
};

class ToPtrOpLowering: public OpConversionPattern<sparlay::ToPtrOp> {
public:
  using OpConversionPattern<sparlay::ToPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::ToPtrOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, index};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "getPtr", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class ToCrdOpLowering: public OpConversionPattern<sparlay::ToCrdOp> {
public:
  using OpConversionPattern<sparlay::ToCrdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::ToCrdOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, index};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "getCrd", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class ToValueOpLowering: public OpConversionPattern<sparlay::ToValueOp> {
public:
  using OpConversionPattern<sparlay::ToValueOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::ToValueOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, index};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "getValue", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class ToSizeOpLowering: public OpConversionPattern<sparlay::ToSizeOp> {
public:
  using OpConversionPattern<sparlay::ToSizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::ToSizeOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, index};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "getSize", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class SparlayReturnConverter : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class SparlayToDimSizeConverter
    : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only rewrite annotated DimOp with constant index.
    auto enc = getSparlayEncoding(op.getSource().getType());
    if (!enc)
      return failure();
    Optional<int64_t> index = op.getConstantIndex();
    if (!index)
      return failure();
    // Generate the call.
    Value src = adaptor.getOperands()[0];
    int64_t idx = *index;
    rewriter.replaceOp(op, genSparlayDimSizeCall(rewriter, op, enc, src, idx));
    return success();
  }
};

class SparlayAllocConverter
    : public OpConversionPattern<bufferization::AllocTensorOp> {
public:
  using OpConversionPattern<bufferization::AllocTensorOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    std::cerr << "Enter the lowering AllocTensorOp " << std::endl;
    if (op.getCopy())
      return rewriter.notifyMatchFailure(op, "sparse tensor copy not implemented");
    RankedTensorType resType = op.getType();
    int64_t rank = resType.getRank();
    auto enc = getSparlayEncoding(resType);
    if (!enc)
      return failure();
    // Gather all dimension sizes as SSA values.
    SmallVector<Value> sizes;
    unsigned int operandCtr = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (resType.isDynamicDim(i)) {
        sizes.push_back(adaptor.getOperands()[operandCtr++]);
      } else {
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), op.getStaticSize(i)));
      }
    }
    // Generate the call to construct empty tensor. The sizes are
    // explicitly defined by the arguments to the alloc operator.
    SmallVector<Value, 8> params;
    std::cerr << "Enter SparlaynewParams" << std::endl;
    SparlaynewParams(rewriter, params, op, enc, sizes, rank);
    std::cerr << "Finish SparlaynewParams" << std::endl;
    rewriter.replaceOp(op, genSparlayNewCall(rewriter, op, params)); 
    return success();
  }
};

class SparlayDeallocConverter
    : public OpConversionPattern<bufferization::DeallocTensorOp> {
public:
  using OpConversionPattern<bufferization::DeallocTensorOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::DeallocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {        
    auto enc = getSparlayEncoding(op.getTensor().getType());
    if (!enc) {
      return failure();
    }
    StringRef name = "delSparlayTensor";
    rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, name, llvm::None, adaptor.getOperands(), false), adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LowerFormatConversionPass
//===----------------------------------------------------------------------===//
class SparlayTensorTypeConverter : public TypeConverter {
public:
  SparlayTensorTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertSparlayTypes);
  }
  // Maps each sparse tensor type to an opaque pointer.
  static Optional<Type> convertSparlayTypes(Type type) {
    if (getSparlayEncoding(type) != nullptr)
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
    return llvm::None;
  }
};

/// Sparse conversion rule for tensor rematerialization.
class SparlayLoadConverter : public OpConversionPattern<sparlay::LoadOp> {
public:
  using OpConversionPattern<sparlay::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparlay::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    std::cerr << "Enter the lowering LoadOp " << std::endl;
    if (op.hasInserts()) {
      // Finalize any pending insertions.
      StringRef name = "endInsert";
      TypeRange noTp;
      createFuncCall(rewriter, op, name, noTp, adaptor.getOperands(), true);
    }
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse conversion rule for inserting in lexicographic index order.
class SparlayInsertConverter : public OpConversionPattern<sparlay::InsertOp> {
public:
  using OpConversionPattern<sparlay::InsertOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparlay::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    std::cerr << "Enter the lowering InsertOp " << std::endl;
//    Type elemTp = op.tensor().getType().cast<ShapedType>().getElementType();
    StringRef name = "lexInsert";
    TypeRange noTp;
    replaceOpWithFuncCall(rewriter, op, name, noTp, adaptor.getOperands(), true);
    return success();
  }
};

class SparlayExpandConverter : public OpConversionPattern<sparlay::ExpandOp> {
public:
  using OpConversionPattern<sparlay::ExpandOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparlay::ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
//    std::cerr << "Enter the lowering ExpandOp " << std::endl;
    Location loc = op->getLoc();
    ShapedType srcType = op.tensor().getType().cast<ShapedType>();
    Type eltType = srcType.getElementType();
    Type boolType = rewriter.getIntegerType(1);
    Type idxType = rewriter.getIndexType();
    // All initialization should be done on entry of the loop nest.
    rewriter.setInsertionPointAfter(op.tensor().getDefiningOp());
    // Determine the size for access expansion.
    auto enc = getSparlayEncoding(srcType);
    Value src = adaptor.getOperands()[0];
    Value sz = genSparlayDimSizeCall(rewriter, op, enc, src, srcType.getRank() - 1);
    // Allocate temporary buffers for values, filled-switch, and indices.
    // We do not use stack buffers for this, since the expanded size may
    // be rather large (as it envelops a single expanded dense dimension).
    Value values = genSparlayAlloc(rewriter, loc, sz, eltType);
    Value filled = genSparlayAlloc(rewriter, loc, sz, boolType);
    Value indices = genSparlayAlloc(rewriter, loc, sz, idxType);
    Value zero = mlir::sparse_tensor::constantZero(rewriter, loc, idxType);
    // Reset the values/filled-switch to all-zero/false. Note that this
    // introduces an O(N) operation into the computation, but this reset
    // operation is amortized over the innermost loops for the access
    // pattern expansion. As noted in the operation doc, we would like
    // to amortize this setup cost even between kernels.
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{mlir::sparse_tensor::constantZero(rewriter, loc, eltType)},
        ValueRange{values});
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{mlir::sparse_tensor::constantZero(rewriter, loc, boolType)},
        ValueRange{filled});
    // Replace expansion op with these buffers and initial index.
    assert(op.getNumResults() == 4);
    rewriter.replaceOp(op, {values, filled, indices, zero});
    return success();
  }
};

class SparlayCompressConverter : public OpConversionPattern<sparlay::CompressOp> {
public:
  using OpConversionPattern<sparlay::CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparlay::CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // Note that this method call resets the values/filled-switch back to
    // all-zero/false by only iterating over the set elements, so the
    // complexity remains proportional to the sparsity of the expanded
    // access pattern.
//    Type elemTp = op.tensor().getType().cast<ShapedType>().getElementType();
//    std::cerr << "Enter the lowering CompressOp " << std::endl;
    StringRef name = "expInsert";
    TypeRange noTp;
    replaceOpWithFuncCall(rewriter, op, name, noTp, adaptor.getOperands(), true);
    // Deallocate the buffers on exit of the loop nest.
    Operation *parent = op;
    for (; isa<scf::ForOp>(parent->getParentOp()) ||
           isa<scf::WhileOp>(parent->getParentOp()) ||
           isa<scf::ParallelOp>(parent->getParentOp()) ||
           isa<scf::IfOp>(parent->getParentOp());
         parent = parent->getParentOp())
      ;
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[2]);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[3]);
    rewriter.create<memref::DeallocOp>(loc, adaptor.getOperands()[4]);
    return success();
  }
};

class DiaSpmvOpLowering : public OpConversionPattern<sparlay::DiaSpmvOp> {
public:
    using OpConversionPattern<sparlay::DiaSpmvOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::DiaSpmvOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value output = op->getResult(0);
        auto outputType = output.getType();
        Value input_A = adaptor.getOperands()[0];
        Value input_B = adaptor.getOperands()[1];
        Value input_C = adaptor.getOperands()[2];

        std::vector<Value> params = {input_A, input_B, input_C};
        func::CallOp SpmvOp = rewriter.create<func::CallOp>(loc, outputType, 
            getFunc(op, "kernel_dia_spmv", outputType, params, /*emitCInterface=*/true), params);
        auto ret = SpmvOp.getResult(0);
        rewriter.replaceOp(op, ret);
        return success();
    }
};

class DiaSpmmOpLowering : public OpConversionPattern<sparlay::DiaSpmmOp> {
public:
    using OpConversionPattern<sparlay::DiaSpmmOp>::OpConversionPattern;

    LogicalResult 
        matchAndRewrite(sparlay::DiaSpmmOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
        Location loc = op->getLoc();
        Value output = op->getResult(0);
        auto outputType = output.getType();
        Value input_A = adaptor.getOperands()[0];
        Value input_B = adaptor.getOperands()[1];
        Value input_C = adaptor.getOperands()[2];
        
        std::vector<Value> params = {input_A, input_B, input_C};
        func::CallOp SpmmOp = rewriter.create<func::CallOp>(loc, outputType, 
            getFunc(op, "kernel_dia_spmm", outputType, params, /*emitCInterface=*/true), params);
        auto ret = SpmmOp.getResult(0);
        rewriter.replaceOp(op, ret);
        return success();
    }
};

class COOSpMVOpLowering: public OpConversionPattern<sparlay::COOSpMVOp> {
public:
  using OpConversionPattern<sparlay::COOSpMVOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::COOSpMVOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value vec = adaptor.getOperands()[1];
    Value out_vec = adaptor.getOperands()[2];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, vec, out_vec};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "calculateCOOSpMV", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class COOSpMMOpLowering: public OpConversionPattern<sparlay::COOSpMMOp> {
public:
  using OpConversionPattern<sparlay::COOSpMMOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::COOSpMMOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value vec = adaptor.getOperands()[1];
    Value out_vec = adaptor.getOperands()[2];
    Type outputType = op->getResult(0).getType();
    std::vector<Value> params = {inputTensor, vec, out_vec};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "calculateCOOSpMM", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class BDIASpMVOpLowering: public OpConversionPattern<sparlay::BDIASpMVOp> {
public:
  using OpConversionPattern<sparlay::BDIASpMVOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::BDIASpMVOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor_CSR = adaptor.getOperands()[0];
    Value inputTensor_BDIA = adaptor.getOperands()[1];
    Value input_B = adaptor.getOperands()[2];
    auto inputType_B = input_B.getType().dyn_cast<TensorType>();
    Value input_C = adaptor.getOperands()[3];
    auto inputType_C = input_C.getType().dyn_cast<TensorType>();
    Value output = op->getResult(0);
    auto outputType = output.getType();

    // auto dynShape = {ShapedType::kDynamicSize};
    auto mem_B_tp = MemRefType::get(inputType_B.getShape(), inputType_B.getElementType());
    auto mem_C_tp = MemRefType::get(inputType_C.getShape(), inputType_C.getElementType());
    Value mem_input_B = rewriter.create<bufferization::ToMemrefOp>(loc, mem_B_tp, input_B);
    Value mem_input_C = rewriter.create<bufferization::ToMemrefOp>(loc, mem_C_tp, input_C);

    std::vector<Value> params = {inputTensor_CSR, inputTensor_BDIA, mem_input_B, mem_input_C};
    // auto out_tp = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "kernel_hetero_bdia_spmv_iter", outputType, params, true),
        params
    );
    rewriter.replaceOp(op, callOp.getResult(0));
    // rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, outputType, 
    //         ValueRange({callOp.getResult(0)}));
    return success();
  }
};

class BDIASpMMOpLowering: public OpConversionPattern<sparlay::BDIASpMMOp> {
public:
  using OpConversionPattern<sparlay::BDIASpMMOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::BDIASpMMOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor_CSR = adaptor.getOperands()[0];
    Value inputTensor_BDIA = adaptor.getOperands()[1];
    Value input_B = adaptor.getOperands()[2];
    auto inputType_B = input_B.getType().dyn_cast<TensorType>();
    Value input_C = adaptor.getOperands()[3];
    auto inputType_C = input_C.getType().dyn_cast<TensorType>();
    Value output = op->getResult(0);
    auto outputType = output.getType();

    // auto dynShape = {ShapedType::kDynamicSize};
    auto mem_B_tp = MemRefType::get(inputType_B.getShape(), inputType_B.getElementType());
    auto mem_C_tp = MemRefType::get(inputType_C.getShape(), inputType_C.getElementType());
    Value mem_input_B = rewriter.create<bufferization::ToMemrefOp>(loc, mem_B_tp, input_B);
    Value mem_input_C = rewriter.create<bufferization::ToMemrefOp>(loc, mem_C_tp, input_C);

    std::vector<Value> params = {inputTensor_CSR, inputTensor_BDIA, mem_input_B, mem_input_C};
    // auto out_tp = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "kernel_bdia_spmm_iter", outputType, params, true),
        params
    );
    rewriter.replaceOp(op, callOp.getResult(0));
    // rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, outputType, 
    //         ValueRange({callOp.getResult(0)}));
    return success();
  }
};

class DecomposeBDIAOpLowering: public OpConversionPattern<sparlay::DecomposeBDIAOp> {
public:
  using OpConversionPattern<sparlay::DecomposeBDIAOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::DecomposeBDIAOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value blockSize = adaptor.getOperands()[1];
    Value thres = adaptor.getOperands()[2];
    Type outputType = inputTensor.getType();
    std::vector<Value> params = {inputTensor, blockSize, thres};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "decompose_BDIA", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class DecomposeBELLOpLowering: public OpConversionPattern<sparlay::DecomposeBELLOp> {
public:
  using OpConversionPattern<sparlay::DecomposeBELLOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::DecomposeBELLOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getOperands()[0];
    Value blockSize = adaptor.getOperands()[1];
    Value block_thres = adaptor.getOperands()[2];
    Value col_thres = adaptor.getOperands()[3];
    Type outputType = inputTensor.getType();
    std::vector<Value> params = {inputTensor, blockSize, block_thres, col_thres};
    auto callOp = rewriter.create<func::CallOp>(loc, outputType,
        getFunc(op, "decompose_BELL_COO", outputType, params, true),
        params
    );
    auto ret = callOp.getResult(0);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class ReleaseOpLowering: public OpConversionPattern<sparlay::ReleaseOp> {
public:
  using OpConversionPattern<sparlay::ReleaseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sparlay::ReleaseOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::CallOp>(op, llvm::None, 
            getFunc(op, "release", llvm::None, adaptor.getOperands(), false), adaptor.getOperands());
    return success();
  }
};

} // end anonymous namespace



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
    SparlayTensorTypeConverter converter;

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `MemRef` and `Standard` dialects.
    target.addLegalDialect<scf::SCFDialect, memref::MemRefDialect,
                           vector::VectorDialect, bufferization::BufferizationDialect,
                           arith::ArithmeticDialect, LLVM::LLVMDialect, func::FuncDialect>();
    target.addIllegalDialect<sparlay::SparlayDialect>();

    // We also define the Sparlay dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Sparlay operations that don't want
    // to lower as `legal`.
  
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::DimOp>([&](tensor::DimOp op) {
      return converter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<tensor::CastOp>([&](tensor::CastOp op) {
      return converter.isLegal(op.getSource().getType()) &&
             converter.isLegal(op.getDest().getType());
    });
    target.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [&](tensor::ExpandShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [&](tensor::CollapseShapeOp op) {
          return converter.isLegal(op.getSrc().getType()) &&
                 converter.isLegal(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
        [&](bufferization::AllocTensorOp op) {
          return converter.isLegal(op.getType());
        });

    target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
    [&](bufferization::DeallocTensorOp op) {
        return converter.isLegal(op.getTensor().getType());
    });

    //target.addLegalOp<bufferization::DeallocTensorOp>();    
    target.addLegalOp<linalg::FillOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Sparlay operations.
    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    //TODO: maybe add converter in this call? Need to figure out how converter works
    patterns.add<NewOpLowering, fromFileOpLowering, ConvertOpLowering, printStorageOpLowering,
                 checkOpLowering, copyOpLowering, ticOpLowering, tocOpLowering,
                 StructAccessOpLowering, DecompseOpLowering, ToCrdOpLowering, 
                 ToPtrOpLowering, ToValueOpLowering, ToSizeOpLowering, 
                 SparlayAllocConverter, SparlayDeallocConverter, SparlayToDimSizeConverter,
                 SparlayLoadConverter, SparlayInsertConverter, SparlayReturnConverter,
                 SparlayExpandConverter, SparlayCompressConverter, 
                 DiaSpmvOpLowering, DiaSpmmOpLowering, COOSpMVOpLowering, COOSpMMOpLowering,
                 DecomposeBDIAOpLowering, DecomposeBELLOpLowering, BDIASpMVOpLowering, BDIASpMMOpLowering, ReleaseOpLowering>(&getContext());
    // LLVM_DEBUG(llvm::dbgs() << "Has the pattern rewrite applied?\n");

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    auto curOp = getOperation();
    if (failed(applyPartialConversion(curOp, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::sparlay::createLowerFormatConversionPass() {
    return std::make_unique<LowerFormatConversionPass>();
}
