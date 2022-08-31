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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"
#include "IR/SparlayTypes.h"
#include "Transforms/Passes.h"


using namespace mlir;
using namespace sparlay;

namespace {
#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"


SparlayEncodingAttr getSparlayEncoding(Type type) {
  if (auto ttp = type.dyn_cast<RankedTensorType>())
    return ttp.getEncoding().dyn_cast_or_null<SparlayEncodingAttr>();
  return nullptr;
}

class LowerGenericOp : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
    // Location loc = op.getLoc();
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

      //TODO: integrate the following and create ops
      for (int i = 0; i < nLevel; ++i) {
        std::cerr << "== Level " << i << ": " << std::endl;
        std::cerr << "==== Have: ";
        if (i == mn_trim_level - 1) std::cerr << "ptr ";
        else {
          if (i >= mn_trim_level && i <= mx_trim_level) std::cerr << "crd ";
          else if (i < mn_trim_level) std::cerr << "only_size ";
          else if (i > mx_trim_level) std::cerr << "dense_tensor ";
          while (pt < fuse.size() && fuse[pt] < i) pt++;
          if (pt < fuse.size() && fuse[pt] == i) std::cerr << "ptr ";
        }
        std::cerr << std::endl;
      }
    }
    return success();
  }
};

struct TmpGenBufferPass: public TmpGenBufferBase<TmpGenBufferPass> {
  TmpGenBufferPass() = default;
  TmpGenBufferPass(const TmpGenBufferPass &pass) : TmpGenBufferBase<TmpGenBufferPass>() {}
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerGenericOp>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} //end of anonymus namespace

std::unique_ptr<Pass> mlir::sparlay::createTmpGenBuffer() {
    return std::make_unique<TmpGenBufferPass>();
}