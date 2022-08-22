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
