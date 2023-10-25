#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "IR/UniSparseDialect.h"
#include "IR/UniSparseOps.h"
#include "IR/UniSparseTypes.h"
#include "Transforms/Passes.h"

#include <cstdio>
#include <cstring>

using namespace mlir;
using namespace unisparse;

namespace {
#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

class StructConvertOpLowering : public OpConversionPattern<unisparse::StructConvertOp> {
public:
  using OpConversionPattern<unisparse::StructConvertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(unisparse::StructConvertOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    //access all element
    Value input = op->getOperand(0);
    Value output = op->getResult(0);
    auto inputType = input.getType().dyn_cast<StructType>();
    auto outputType = output.getType().dyn_cast<StructType>();
    llvm::ArrayRef<mlir::Type> inputElmTypes = inputType.getElementTypes();
    llvm::ArrayRef<mlir::Type> outputElmTypes = outputType.getElementTypes();
    uint64_t inputSize = inputElmTypes.size();
    uint64_t outputSize = outputElmTypes.size();
    assert(inputSize == outputSize);
    std::vector<Value> inputValues;
    for (uint64_t i = 0; i < inputSize; ++i) {
      inputValues.push_back(rewriter.create<unisparse::StructAccessOp>(loc,inputElmTypes[i],input,i));
    }
    std::vector<Value> outputValues;
    //convert
    for (size_t i = 0; i < inputValues.size(); ++i) {
      outputValues.push_back(rewriter.create<unisparse::ConvertOp>(loc, outputElmTypes[i], inputValues[i]));
    }
    //construct a new struct
    Value outStruct = rewriter.create<unisparse::StructConstructOp>(loc, outputType, llvm::makeArrayRef(outputValues));
    rewriter.replaceOp(op, outStruct);
    return success();
  }

};

struct LowerStructConvertPass : 
public LowerStructConvertBase<LowerStructConvertPass> {
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<
      arith::ArithmeticDialect, LLVM::LLVMDialect, 
      func::FuncDialect, unisparse::UniSparseDialect
    >();
    target.addIllegalOp<unisparse::StructConvertOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<StructConvertOpLowering>(&getContext());
    func::FuncOp curOp = getOperation();
    if (
      failed(applyPartialConversion(curOp, target, std::move(patterns)))
    ) {
      signalPassFailure();
    }
  }
};

} //end of anonymus namespace

std::unique_ptr<Pass> mlir::unisparse::createLowerStructConvertPass() {
    return std::make_unique<LowerStructConvertPass>();
}
