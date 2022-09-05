//===- SparlayCodegen.cpp - Implementation of code gen for sparse kernel using sparse tensor defined by Sparlay dialect--------------===//

#include "Transforms/Passes.h"
#include "IR/SparlayDialect.h"
#include "IR/SparlayTypes.h"
#include "IR/SparlayOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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
/*
static bool findAffine(Merger &merger, unsigned tensor, AffineExpr a, Dim dim,
                       bool isDense) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (!merger.isDim(tensor, idx, Dim::kUndef))
      return false; // used more than once
    merger.setDim(tensor, idx, dim);
    return true;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    if (!isDense)
      return false;
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return findAffine(merger, tensor, binOp.getLHS(), dim, isDense) &&
           findAffine(merger, tensor, binOp.getRHS(), dim, isDense);
  }
  case AffineExprKind::Constant:
    return isDense;
  default:
    return false;
  }
}

static bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  bool annotated = false;
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparlayEncoding(t->get().getType());
    if (enc)
      annotated = true;
    assert(map.getNumResults() == op.getRank(t));
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      unsigned tensor = t->getOperandNumber();
      AffineExpr a = map.getResult(perm(enc, d));
      if (!findAffine(merger, tensor, a, toDim(enc, d), !enc))
        return false; // inadmissable affine expression
    }
  }
  return annotated;
}

static bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort,
                                  unsigned mask) {
  // Set up an n x n from/to adjacency matrix of the iteration graph
  // for the implicit loop indices i_0 .. i_n-1.
  unsigned n = op.getNumLoops();
  std::vector<std::vector<bool>> adjM(n, std::vector<bool>(n, false));

  // Iterate over the indexing maps of every tensor in the tensor expression.
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparlayEncoding(t->get().getType());
    assert(map.getNumDims() == n);
    // Skip dense tensor constraints when not requested.
    if (!(mask & SortMask::kIncludeDense) && !enc)
      continue;
    // Each tensor expression and optional dimension ordering (row-major
    // by default) puts an ordering constraint on the loop indices. For
    // example, the tensor expresion A_ijk forces the ordering i < j < k
    // on the loop indices if no explicit dimension ordering is given.
    for (unsigned d = 1, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr f = map.getResult(perm(enc, d - 1));
      AffineExpr t = map.getResult(perm(enc, d));
      addAffineOrderings(adjM, f, t, 0);
    }
    // Push unrelated loops into sparse iteration space, so these
    // will be skipped more often.
    if (mask & SortMask::kIncludeUndef) {
      unsigned tensor = t->getOperandNumber();
      for (unsigned i = 0; i < n; i++)
        if (merger.isDim(tensor, i, Dim::kSparse))
          for (unsigned j = 0; j < n; j++)
            if (merger.isDim(tensor, j, Dim::kUndef))
              adjM[i][j] = true;
    }
  }

  // Topologically sort the iteration graph to determine loop order.
  // Report failure for a cyclic iteration graph.
  topSort.clear();
  topSort.reserve(n);
  std::vector<unsigned> visit(n, 0);
  for (unsigned i = 0; i < n; i++)
    if (visit[i] == 0)
      if (!topSortDFS(i, visit, topSort, adjM))
        return false; // cycle!
  std::reverse(std::begin(topSort), std::end(topSort));
  return true;
}
*/

SparlayEncodingAttr getSparlayEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<SparlayEncodingAttr>();
  return nullptr;
}

LogicalResult genBuffer(linalg::GenericOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  std::vector<Value> interStorage;
  for (OpOperand *t: op.getInputAndOutputOperands()) {
    // unsigned ts_id = t->getOperandNumber();
    auto enc = getSparlayEncoding(t->get().getType());
    if (enc == nullptr) continue;
    auto crd = enc.getCrdMap();
    auto compress = enc.getCompressMap();
    auto trim = compress.getTrimIndex();
    auto fuse = compress.getFuseIndex();
    // crd.Print();
    // compress.Print();
    int nLevel = crd.getNumResults();
    std::cerr << "nLevel = " << nLevel << std::endl;
    int mn_trim_level = 1000, mx_trim_level = -1;
    for (size_t i = 0; i < trim.size(); ++i) {
      mn_trim_level = std::min(mn_trim_level, trim[i]);
      mx_trim_level = std::max(mx_trim_level, trim[i]);
    }
    size_t pt = 0;
    auto i32Tp = rewriter.getI32Type();
    //TODO: integrate the following and create ops
    for (int i = 0; i < nLevel; ++i) {
      Value dim = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
      std::cerr << "== Level " << i << ": " << std::endl;
      std::cerr << "==== Have: ";
      if (i == mn_trim_level - 1) {
        std::cerr << "ptr ";
        interStorage.push_back(rewriter.create<sparlay::ToPtrOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim));
      } else {
        if (i >= mn_trim_level && i <= mx_trim_level) {
          std::cerr << "crd ";
          interStorage.push_back(rewriter.create<sparlay::ToCrdOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim));
        }
        else if (i < mn_trim_level) std::cerr << "only_size ";
        else if (i > mx_trim_level) std::cerr << "dense_tensor ";
        while (pt < fuse.size() && fuse[pt] < i) pt++;
        if (pt < fuse.size() && fuse[pt] == i) {
          std::cerr << "ptr ";
          interStorage.push_back(rewriter.create<sparlay::ToPtrOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim));
        }
      }
      std::cerr << std::endl;
    }
    Value dim0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    interStorage.push_back(rewriter.create<sparlay::ToValueOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim0));
  }
  /****** The following is only for the consistency of the code ******/
  Value c1 = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  for (OpOperand *t: op.getOutputOperands()) {
    auto enc = getSparlayEncoding(t->get().getType());
    if (enc != nullptr) {
      std::cerr << "Unable to generate the pseudo output tensor, please change the code directly." << std::endl;
    }
    std::vector<Value> dims;
    auto tType = t->get().getType().dyn_cast<RankedTensorType>();
    assert(tType != nullptr);
    int nDim = tType.getShape().size();
    for (int i = 0; i < nDim; ++i) dims.push_back(c1);
    interStorage.push_back(rewriter.create<bufferization::AllocTensorOp>(loc, tType, llvm::makeArrayRef(dims)));
  }
  std::vector<Type> eleTypes;
  std::vector<int64_t> dimSizes = {};
  std::vector<AffineMap> order = {};
  for (size_t i = 0; i < interStorage.size(); ++i) {
    eleTypes.push_back(interStorage[i].getType());
  }
  auto outputType = sparlay::StructType::get(llvm::makeArrayRef(dimSizes), llvm::makeArrayRef(eleTypes), "", llvm::makeArrayRef(order));
  Value outStruct = rewriter.create<sparlay::StructConstructOp>(loc, outputType, llvm::makeArrayRef(interStorage));
  Value outTensor = rewriter.create<sparlay::StructAccessOp>(loc, outStruct, interStorage.size()-1);
  rewriter.replaceOp(op, outTensor);
  /****** Redundant code end here ******/
  return success();
}

struct GenericOpSparlayCodegen : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparlayCodegen(MLIRContext *context) : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    return genBuffer(op, rewriter);
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
