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
#include "mlir/../../lib/Dialect/SparseTensor/Transforms/CodegenUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

#define DEBUG_TYPE "sparlay-codegen"

namespace mlir{
namespace sparlay {

#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

enum SortMask {
  kSparseOnly = 0x0,
  kIncludeDense = 0x1,
  kIncludeUndef = 0x2,
  kIncludeAll = 0x3
};

// Reduction kinds.
enum Reduction { kNoReduc, kSum, kProduct, kAnd, kOr, kXor };

struct CodeGen {
  CodeGen(unsigned numTensors, unsigned numLoops,
          OpOperand *op, unsigned nest)
      : loops(numLoops), sizes(numLoops), buffers(numTensors),
        pointers(numTensors, std::vector<Value>(numLoops)),
        indices(numTensors, std::vector<Value>(numLoops)),
        highs(numTensors, std::vector<Value>(numLoops)),
        pidxs(numTensors, std::vector<Value>(numLoops)),
        idxs(numTensors, std::vector<Value>(numLoops)), redVal(), sparseOut(op),
        outerParNest(nest), lexIdx(), lexVal(), expValues(), expFilled(),
        expAdded(), expCount(), curVecMask() {}
  /// Universal dense indices and upper bounds (by index). The loops array
  /// is updated with the value of the universal dense index in the current
  /// loop. The sizes array is set once with the inferred dimension sizes.
  std::vector<Value> loops;
  std::vector<Value> sizes;
  /// Buffers for storing dense and sparse numerical values (by tensor).
  /// This array is set once during bufferization of all tensors.
  std::vector<Value> buffers;
  /// Sparse storage schemes (1-D): pointers and indices (by tensor and index).
  /// This array is set once during bufferization of all sparse tensors.
  std::vector<std::vector<Value>> pointers;
  std::vector<std::vector<Value>> indices;
  /// Sparse iteration information (by tensor and index). These arrays
  /// are updated to remain current within the current loop.
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> idxs;
  /// Current reduction, updated during code generation. When indices of a
  /// reduction are exhausted, all inner loops can use a scalarized reduction.
  unsigned redExp = -1u;
  Value redVal;
  Reduction redKind = kNoReduc;
  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order (where indices are updated
  // in the temporary array `lexIdx`) or through access pattern expansion
  // in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value lexIdx;
  Value lexVal;
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;
  // Current vector length and mask.
  unsigned curVecLength = 1;
  Value curVecMask;
};

unsigned perm(const SparlayEncodingAttr &enc, unsigned d) {
  if (enc) {
    auto order = enc.getCrdMap();
    if (order) {
      assert(order.isPermutation());
      return order.getDimPosition(d);
    }
  }
  return d;
}

void createTrimVec(std::vector<int> trim, std::vector<bool> &trim_vec, unsigned rank) {
  int trim_to = trim[0];
  int trim_from = trim[1];
  assert(trim_to <= trim_from);
  for (int i = trim_to; i <= trim_from; i++) {
    trim_vec[i] = true;
  }  
}

void createMergeVec(std::vector<int> merge, std::vector<bool> &merge_vec, unsigned rank){
  int end = merge[0];
  for(int i = 0; i <= end; i++) {
    merge_vec[i] = true;
  } 
}

Dim toDim(const SparlayEncodingAttr &enc, unsigned d) {
  if(enc) { 
    auto crdmap = enc.getCrdMap();
    auto compress = enc.getCompressMap();
    auto trim = compress.getTrimIndex();
    auto merge = compress.getFuseIndex();
    unsigned rank = crdmap.getNumResults();
    std::vector<bool> trim_vec(rank, false); 
    createTrimVec(trim, trim_vec, rank);
    std::vector<bool> merge_vec(rank, false);
    createMergeVec(merge, merge_vec, rank);

    if(d == 0) {
      if(trim_vec[d]) {
        return Dim::kSparse;
      }
    } else if (trim_vec[d] && merge_vec[d-1]) {
      return Dim::kSparse;
    } else {
      return Dim::kSingle;
    }
  }
  return Dim::kDense;
}

 bool findAffine(Merger &merger, unsigned tensor, AffineExpr a, Dim dim,
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

  bool findSparseAnnotations(Merger &merger, linalg::GenericOp op) {
  std::cerr << "Enter findSparseAnnotations " << std::endl;
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

  bool topSortDFS(unsigned i, std::vector<unsigned> &visit,
                       std::vector<unsigned> &topSort,
                       std::vector<std::vector<bool>> &adjM) {
  if (visit[i] != 0)
    return visit[i] != 1; // 1 denotes cycle!
  visit[i] = 1;
  for (unsigned j = 0, e = visit.size(); j < e; j++)
    if (adjM[i][j])
      if (!topSortDFS(j, visit, topSort, adjM))
        return false;
  visit[i] = 2;
  topSort.push_back(i);
  return true;
}

  void addAffineOrderings(std::vector<std::vector<bool>> &adjM,
                               AffineExpr a, AffineExpr b, unsigned fidx) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (b)
      addAffineOrderings(adjM, b, AffineExpr(), idx);
    else
      adjM[fidx][idx] = true;
    break;
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    addAffineOrderings(adjM, binOp.getLHS(), b, fidx);
    addAffineOrderings(adjM, binOp.getRHS(), b, fidx);
    break;
  }
  default:
    break;
  }
}

bool computeIterationGraph(Merger &merger, linalg::GenericOp op,
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

bool isMaterializing(Value val) {
  return val.getDefiningOp<linalg::InitTensorOp>() ||
         val.getDefiningOp<bufferization::AllocTensorOp>();
}

bool isAdmissableTensorExp(Merger &merger, linalg::GenericOp op,
                                  std::vector<unsigned> &topSort, unsigned exp,
                                  OpOperand **sparseOut,
                                  unsigned &outerParNest) {
  OpOperand *lhs = op.getOutputOperand(0);
  unsigned tensor = lhs->getOperandNumber();
  auto enc = getSparlayEncoding(lhs->get().getType());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissable since insertions cannot occur.
  if (!enc)
    return true;
  // An all-dense annotated "sparse" output tensor becomes a linearized random
  // access 1-dim memref. Also admissable since insertions cannot occur.
  bool allDense = true;
  auto iteratorTypes = op.iterator_types().getValue();
  unsigned numLoops = iteratorTypes.size();
  for (unsigned i = 0; i < numLoops; i++)
    if (merger.isDim(tensor, i, Dim::kSparse)) {
      allDense = false;
      break;
    }
  if (allDense)
    return true;
  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissable without special codegen.
  if (merger.isSingleCondition(tensor, exp))
    return true;
  // Accept "truly dynamic" if the output tensor materializes uninitialized
  // into the computation and insertions occur in lexicographic index order.
  if (isMaterializing(lhs->get())) {
    unsigned nest = 0;
    for (unsigned i = 0; i < numLoops; i++) {
      if (isReductionIterator(iteratorTypes[topSort[i]]))
        break; // terminate at first reduction
      nest++;
    }
    // Determine admissable dynamic insertion situations:
    // (1) fully injective, since there are no reductions,
    // (2) admissable 1-d expansion in innermost dimension.
    if (nest >= op.getRank(lhs) - 1) {
      *sparseOut = lhs;
      outerParNest = nest;
      return true;
    }
  }
  return false;
}

/// Updates scalarized reduction value.
  void updateReduc(Merger &merger, CodeGen &codegen, Value reduc) {
  assert(codegen.redKind != kNoReduc);
  codegen.redVal = merger.exp(codegen.redExp).val = reduc;
}

  vector::CombiningKind getCombiningKind(Reduction kind) {
  switch (kind) {
  case kNoReduc:
    break;
  case kSum:
    return vector::CombiningKind::ADD;
  case kProduct:
    return vector::CombiningKind::MUL;
  case kAnd:
    return vector::CombiningKind::AND;
  case kOr:
    return vector::CombiningKind::OR;
  case kXor:
    return vector::CombiningKind::XOR;
  }
  llvm_unreachable("unknown reduction kind");
}

/// Maps operation to reduction.
  Reduction getReduction(Kind kind) {
  switch (kind) {
  case Kind::kAddF:
  case Kind::kAddC:
  case Kind::kAddI:
  case Kind::kSubF:
  case Kind::kSubC:
  case Kind::kSubI:
    return kSum;
  case Kind::kMulF:
  case Kind::kMulC:
  case Kind::kMulI:
    return kProduct;
  case Kind::kAndI:
    return kAnd;
  case Kind::kOrI:
    return kOr;
  case Kind::kXorI:
    return kXor;
  default:
    llvm_unreachable("unexpected reduction operator");
  }
}

static Value genOutputBuffer(CodeGen &codegen, OpBuilder &builder,
                             linalg::GenericOp op, MemRefType denseTp,
                             ArrayRef<Value> args) {
  Location loc = op.getLoc();
  OpOperand *lhs = op.getOutputOperand(0);
  Value tensor = lhs->get();
  bool isInit = op.isInitTensor(lhs);
  Value init = builder.create<bufferization::ToMemrefOp>(loc, denseTp, tensor);
  if (!isInit) {
    Value zero = constantZero(builder, loc, denseTp.getElementType());
    builder.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init});
  }
  return init;
}

void genBuffers(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op) {
  Location loc = op.getLoc();
  assert(op.getNumInputsAndOutputs() == op.getNumInputs() + 1);
  std::cerr << "Enter genBuffers" << std::endl;
  SmallVector<Value, 4> args;
  for (OpOperand *t : op.getInputAndOutputOperands()) {
    unsigned tensor = t->getOperandNumber();
    auto shape = op.getShape(t);
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparlayEncoding(t->get().getType());
    auto i32Tp = builder.getI32Type();
    // Scan all dimensions of current tensor.
    args.clear();
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      if (a.getKind() != AffineExprKind::DimId)
        continue; // compound
      unsigned idx = a.cast<AffineDimExpr>().getPosition();
      // Handle sparse storage schemes.
      if (merger.isDim(tensor, idx, Dim::kSparse)) {
      //  auto dynShape = {ShapedType::kDynamicSize};
        Value dim = constantIndex(builder, loc, d);
        Value dim_1 = constantIndex(builder, loc, d+1);
        // Generate sparse primitives to obtains pointer and indices.
        codegen.pointers[tensor][idx] =
            builder.create<sparlay::ToPtrOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim);
        codegen.indices[tensor][idx] =
            builder.create<sparlay::ToCrdOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, i32Tp), t->get(), dim_1);
      }
      // Find upper bound in current dimension.
      unsigned p = perm(enc, d);
      Value up = linalg::createOrFoldDimOp(builder, loc, t->get(), p);
      if (ShapedType::isDynamic(shape[p]))
        args.push_back(up);
      assert(codegen.highs[tensor][idx] == nullptr);
      codegen.sizes[idx] = codegen.highs[tensor][idx] = up;
    }
    // Perform the required bufferization. Dense inputs materialize
    // from the input tensors. Dense outputs need special handling.
    // Sparse inputs use sparse primitives to obtain the values.
    // We also accept in-place all-dense annotated "sparse" outputs.
    Type elementType = getElementTypeOrSelf(t->get().getType());
    if (!enc) {
      // Non-annotated dense tensors.
      auto denseTp = MemRefType::get(shape, elementType);
      if (tensor < op.getNumInputs())
        codegen.buffers[tensor] =
            builder.create<bufferization::ToMemrefOp>(loc, denseTp, t->get());
      else
        codegen.buffers[tensor] =
            genOutputBuffer(codegen, builder, op, denseTp, args);
    } else if (t == codegen.sparseOut) {
      std::cerr << "Does not support sparse output right now" << std::endl;
    } else {
      // Annotated sparse tensors.
    // auto dynShape = {ShapedType::kDynamicSize};
    //  auto sparseTp = MemRefType::get(dynShape, elementType);
      Value dim0 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
      codegen.buffers[tensor] =
          builder.create<sparlay::ToValueOp>(loc, MemRefType::get({ShapedType::kDynamicSize}, elementType), t->get(), dim0);
    }
  }
}

/*
LogicalResult genBuffer(linalg::GenericOp op, PatternRewriter &rewriter, CodeGen &codegen) {
  Location loc = op.getLoc();
  std::vector<Value> interStorage;
  for (OpOperand *t: op.getInputAndOutputOperands()) {
    unsigned tensor = t->getOperandNumber();
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
      std::cerr << "trim[" << i << "] is " << trim[i] << std::endl;
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
  return success();
}
*/

  Value genAffine(CodeGen &codegen, OpBuilder &builder, AffineExpr a,
                       Location loc) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    return codegen.loops[idx]; // universal dense index
  }
  case AffineExprKind::Add: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::AddIOp>(
        loc, genAffine(codegen, builder, binOp.getLHS(), loc),
        genAffine(codegen, builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return builder.create<arith::MulIOp>(
        loc, genAffine(codegen, builder, binOp.getLHS(), loc),
        genAffine(codegen, builder, binOp.getRHS(), loc));
  }
  case AffineExprKind::Constant: {
    int64_t c = a.cast<AffineConstantExpr>().getValue();
    return constantIndex(builder, loc, c);
  }
  default:
    llvm_unreachable("unexpected affine subscript");
  }
}

  Value genIndex(CodeGen &codegen, linalg::GenericOp op, OpOperand *t) {
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparlayEncoding(t->get().getType());
  AffineExpr a = map.getResult(perm(enc, map.getNumResults() - 1));
  assert(a.getKind() == AffineExprKind::DimId);
  unsigned idx = a.cast<AffineDimExpr>().getPosition();
  return codegen.loops[idx];
}

  Value genSubscript(CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, OpOperand *t,
                          SmallVector<Value, 4> &args) {
  unsigned tensor = t->getOperandNumber();
  auto map = op.getTiedIndexingMap(t);
  auto enc = getSparlayEncoding(t->get().getType());
  unsigned rank = map.getNumResults();
  if (enc) {
    // Note that currently, all sparse subscripts are simple.
    // TODO: accept affine too?
    AffineExpr a = map.getResult(perm(enc, rank - 1));
    assert(a.getKind() == AffineExprKind::DimId);
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    assert(codegen.pidxs[tensor][idx] != nullptr);
    args.push_back(codegen.pidxs[tensor][idx]); // position index
  } else {
    for (unsigned d = 0; d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      args.push_back(genAffine(codegen, builder, a, op.getLoc()));
    }
  }
  return codegen.buffers[tensor];
}


  Value genTensorLoad(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp) {
  // Test if the load was hoisted to a higher loop nest.
  Value val = merger.exp(exp).val;
  if (val) {
    /*
    if (codegen.curVecLength > 1 && !val.getType().isa<VectorType>())
      return genVectorInvariantValue(codegen, builder, val);
    */
    return val;
  }
  // Load during insertion.
  OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
  assert(t != codegen.sparseOut);
/*
  if (t == codegen.sparseOut)
    return genInsertionLoad(codegen, builder, op, t);
*/
  // Actual load.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, builder, op, t, args);
/*
  if (codegen.curVecLength > 1)
    return genVectorLoad(codegen, builder, ptr, args);
*/
  return builder.create<memref::LoadOp>(op.getLoc(), ptr, args);
}



  void genTensorStore(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned exp, Value rhs) {
  Location loc = op.getLoc();
  // Test if this is a scalarized reduction.
  if (codegen.redVal) {
/*
    if (codegen.curVecLength > 1)
      rhs = builder.create<arith::SelectOp>(loc, codegen.curVecMask, rhs,
                                            codegen.redVal);
*/
    updateReduc(merger, codegen, rhs);
    return;
  }
  // Store during insertion.
  OpOperand *t = op.getOutputOperand(0);
  assert(t != codegen.sparseOut);
/*
  if (t == codegen.sparseOut) {
    if (!rhs) {
      // Only unary and binary are allowed to return uninitialized rhs
      // to indicate missing output.
      assert(merger.exp(exp).kind == kUnary || merger.exp(exp).kind == kBinary);
    } else {
      genInsertionStore(codegen, builder, op, t, rhs);
    }  
    return;
  }
*/
  // Actual store.
  SmallVector<Value, 4> args;
  Value ptr = genSubscript(codegen, builder, op, t, args);
/*
  if (codegen.curVecLength > 1)
    genVectorStore(codegen, builder, rhs, ptr, args);
*/
//  else
    builder.create<memref::StoreOp>(loc, rhs, ptr, args);
}

  Value genLoad(CodeGen &codegen, OpBuilder &builder, Location loc,
                     Value ptr, Value s) {
  // See https://llvm.org/docs/GetElementPtr.html for some background on
  // the complications described below.
/*
  if (codegen.curVecLength > 1) {
    // Since the index vector is used in a subsequent gather/scatter operations,
    // which effectively defines an unsigned pointer + signed index, we must
    // zero extend the vector to an index width. For 8-bit and 16-bit values,
    // an 32-bit index width suffices. For 32-bit values, zero extending the
    // elements into 64-bit loses some performance since the 32-bit indexed
    // gather/scatter is more efficient than the 64-bit index variant (if the
    // negative 32-bit index space is unused, the enableSIMDIndex32 flag can
    // preserve this performance). For 64-bit values, there is no good way
    // to state that the indices are unsigned, with creates the potential of
    // incorrect address calculations in the unlikely case we need such
    // extremely large offsets.
    Type etp = ptr.getType().cast<MemRefType>().getElementType();
    Value vload = genVectorLoad(codegen, builder, ptr, {s});
    if (!etp.isa<IndexType>()) {
      if (etp.getIntOrFloatBitWidth() < 32)
        vload = builder.create<arith::ExtUIOp>(
            loc, vectorType(codegen, builder.getI32Type()), vload);
      else if (etp.getIntOrFloatBitWidth() < 64 &&
               !codegen.options.enableSIMDIndex32)
        vload = builder.create<arith::ExtUIOp>(
            loc, vectorType(codegen, builder.getI64Type()), vload);
    }
    return vload;
  }
*/

  Value load = builder.create<memref::LoadOp>(loc, ptr, s);
  if (!load.getType().isa<IndexType>()) {
    if (load.getType().getIntOrFloatBitWidth() < 64)
      load = builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), load);
    load =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), load);
  }
  return load;
}

  Value genInvariantValue(Merger &merger, CodeGen &codegen,
                               OpBuilder &builder, unsigned exp) {
  Value val = merger.exp(exp).val;
/*
  if (codegen.curVecLength > 1)
    return genVectorInvariantValue(codegen, builder, val);
*/
  return val;
}

  Value genAddress(CodeGen &codegen, OpBuilder &builder, Location loc,
                        Value size, Value p, Value i) {
  Value mul = builder.create<arith::MulIOp>(loc, size, p);
/*
  if (auto vtp = i.getType().dyn_cast<VectorType>()) {
    Value inv =
        builder.create<arith::IndexCastOp>(loc, vtp.getElementType(), mul);
    mul = genVectorInvariantValue(codegen, builder, inv);
  }
*/
  return builder.create<arith::AddIOp>(loc, mul, i);
}

  Value genIndexValue(CodeGen &codegen, OpBuilder &builder, unsigned idx,
                           unsigned ldx) {
  Value ival = codegen.loops[idx];
//  Type itype = ival.getType();
  // During vectorization, we either encounter:
  // (1) indices already in vector form, as in ... = ind[lo:hi], good to go, or
  // (2) single index, as in ... = i, must convert to [i, i+1, ...] for inner i.
/*
  unsigned vl = codegen.curVecLength;
  if (vl > 1 && !itype.isa<VectorType>()) {
    Location loc = ival.getLoc();
    VectorType vtp = vectorType(codegen, itype);
    ival = builder.create<vector::BroadcastOp>(loc, vtp, ival);
    if (idx == ldx) {
      Value incr;
      if (vtp.isScalable()) {
        Type stepvty = vectorType(codegen, builder.getI64Type());
        Value stepv = builder.create<LLVM::StepVectorOp>(loc, stepvty);
        incr = builder.create<arith::IndexCastOp>(loc, vtp, stepv);
      } else {
        SmallVector<APInt, 4> integers;
        for (unsigned i = 0; i < vl; i++)
          integers.push_back(APInt(64, i));
        auto values = DenseElementsAttr::get(vtp, integers);
        incr = builder.create<arith::ConstantOp>(loc, vtp, values);
      }
      ival = builder.create<arith::AddIOp>(loc, ival, incr);
    }
  }
*/
  return ival;
}

  Value relinkBranch(CodeGen &codegen, RewriterBase &rewriter,
                          Block *block, Value e, unsigned ldx) {
  if (Operation *def = e.getDefiningOp()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(def))
      return genIndexValue(codegen, rewriter, indexOp.dim(), ldx);
    if (def->getBlock() == block) {
      for (unsigned i = 0, n = def->getNumOperands(); i < n; i++)
        def->setOperand(
            i, relinkBranch(codegen, rewriter, block, def->getOperand(i), ldx));
    }
  }
  return e;
}

  Value genExp(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, unsigned exp, unsigned ldx) {
  Location loc = op.getLoc();
  if (exp == -1u)
    return Value();
  if (merger.exp(exp).kind == Kind::kTensor)
    return genTensorLoad(merger, codegen, rewriter, op, exp);
  if (merger.exp(exp).kind == Kind::kInvariant)
    return genInvariantValue(merger, codegen, rewriter, exp);
  if (merger.exp(exp).kind == Kind::kIndex)
    return genIndexValue(codegen, rewriter, merger.exp(exp).index, ldx);
  Value v0 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e0, ldx);
  Value v1 =
      genExp(merger, codegen, rewriter, op, merger.exp(exp).children.e1, ldx);
  Value ee = merger.buildExp(rewriter, loc, exp, v0, v1);
  if (ee && (merger.exp(exp).kind == Kind::kUnary ||
             merger.exp(exp).kind == Kind::kBinary ||
             merger.exp(exp).kind == Kind::kBinaryBranch))
    ee = relinkBranch(codegen, rewriter, ee.getParentBlock(), ee, ldx);
  return ee;
}

  bool isInvariantAffine(const CodeGen &codegen, AffineExpr a,
                              unsigned ldx, bool &atLevel) {
  switch (a.getKind()) {
  case AffineExprKind::DimId: {
    unsigned idx = a.cast<AffineDimExpr>().getPosition();
    if (idx == ldx)
      atLevel = true;
    return codegen.loops[idx] != nullptr; // no longer in play?
  }
  case AffineExprKind::Add:
  case AffineExprKind::Mul: {
    auto binOp = a.cast<AffineBinaryOpExpr>();
    return isInvariantAffine(codegen, binOp.getLHS(), ldx, atLevel) &&
           isInvariantAffine(codegen, binOp.getRHS(), ldx, atLevel);
  }
  default:
    return true;
  }
}

  void genInvariants(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, unsigned exp, unsigned ldx,
                          bool atStart, Kind last = Kind::kTensor) {
  if (exp == -1u)
    return;
  if (merger.exp(exp).kind == Kind::kTensor) {
    // Inspect tensor indices.
    bool atLevel = ldx == -1u;
    OpOperand *t = op.getInputAndOutputOperands()[merger.exp(exp).tensor];
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparlayEncoding(t->get().getType());
    for (unsigned d = 0, rank = map.getNumResults(); d < rank; d++) {
      AffineExpr a = map.getResult(perm(enc, d));
      if (!isInvariantAffine(codegen, a, ldx, atLevel))
        return; // still in play
    }
    // All exhausted at this level (atLevel denotes exactly at this level).
    if (!atLevel)
      return;
    OpOperand *lhs = op.getOutputOperand(0);
    if (lhs == t) {
      // Start or end a scalarized reduction
      if (atStart) {
        Value load = genTensorLoad(merger, codegen, builder, op, exp);
        codegen.redKind = getReduction(last);
        codegen.redExp = exp;
        updateReduc(merger, codegen, load);
      } else {
        Value redVal = codegen.redVal;
        updateReduc(merger, codegen, Value());
        codegen.redExp = -1u;
        codegen.redKind = kNoReduc;
        genTensorStore(merger, codegen, builder, op, exp, redVal);
      }
    } else {
      // Start or end loop invariant hoisting of a tensor load.
      merger.exp(exp).val =
          atStart ? genTensorLoad(merger, codegen, builder, op, exp) : Value();
    }
  } else if (merger.exp(exp).kind != Kind::kInvariant &&
             merger.exp(exp).kind != Kind::kIndex) {
    // Traverse into the binary operations. Note that we only hoist
    // tensor loads, since subsequent MLIR/LLVM passes know how to
    // deal with all other kinds of derived loop invariants.
    Kind last = merger.exp(exp).kind;
    unsigned e0 = merger.exp(exp).children.e0;
    unsigned e1 = merger.exp(exp).children.e1;
    genInvariants(merger, codegen, builder, op, e0, ldx, atStart, last);
    genInvariants(merger, codegen, builder, op, e1, ldx, atStart, last);
  }
}

  bool genInit(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned at, BitVector &inits) {
  bool needsUniv = false;
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse positions.
  for (unsigned b = 0, be = inits.size(); b < be; b++) {
    if (inits[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      if (merger.isDim(b, Dim::kSparse)) {
        // Initialize sparse index.
        unsigned pat = at;
        for (; pat != 0; pat--) {
          if (codegen.pidxs[tensor][topSort[pat - 1]])
            break;
        }
        Value ptr = codegen.pointers[tensor][idx];
        Value one = constantIndex(builder, loc, 1);
        Value p0 = (pat == 0) ? constantIndex(builder, loc, 0)
                              : codegen.pidxs[tensor][topSort[pat - 1]];
        codegen.pidxs[tensor][idx] = genLoad(codegen, builder, loc, ptr, p0);
        Value p1 = builder.create<arith::AddIOp>(loc, p0, one);
        codegen.highs[tensor][idx] = genLoad(codegen, builder, loc, ptr, p1);
      } else {
        // Dense index still in play.
        needsUniv = true;
      }
    }
  }

  // Initialize the universal dense index.
  codegen.loops[idx] = constantIndex(builder, loc, 0);
  return needsUniv;
}

/// Generates a for-loop on a single index.
  Operation *genFor(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, bool isOuter, bool isInner,
                         unsigned idx, BitVector &indices) {
  unsigned fb = indices.find_first();
  unsigned tensor = merger.tensor(fb);
  assert(idx == merger.index(fb));
//  auto iteratorTypes = op.iterator_types().getValue();
//  bool isReduction = isReductionIterator(iteratorTypes[idx]);
  bool isSparse = merger.isDim(fb, Dim::kSparse);
//  bool isVector = isVectorFor(codegen, isInner, isReduction, isSparse) && denseUnitStrides(merger, op, idx);
//  bool isParallel = isParallelFor(codegen, isOuter, isReduction, isSparse, isVector);

  // Prepare vector length.
//  if (isVector)
//    codegen.curVecLength = codegen.options.vectorLength;

  // Loop bounds and increment.
  Location loc = op.getLoc();
  Value lo = isSparse ? codegen.pidxs[tensor][idx] : codegen.loops[idx];
  Value hi = isSparse ? codegen.highs[tensor][idx] : codegen.sizes[idx];
  Value step = constantIndex(builder, loc, codegen.curVecLength);
/*  
  if (isVector && codegen.options.enableVLAVectorization) {
    Value vscale = builder.create<vector::VectorScaleOp>(
        loc, IndexType::get(builder.getContext()));
    step = builder.create<arith::MulIOp>(loc, vscale, step);
  }

  // Emit a parallel loop.
  if (isParallel) {
    assert(!isVector);
    scf::ParallelOp parOp = builder.create<scf::ParallelOp>(loc, lo, hi, step);
    if (isSparse)
      codegen.pidxs[tensor][idx] = parOp.getInductionVars()[0];
    else
      codegen.loops[idx] = parOp.getInductionVars()[0];
    builder.setInsertionPointToStart(parOp.getBody());
    return parOp;
  }
*/
  // Emit a sequential or vector loop.
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    // In a vector loop, bring reduction into SIMD form, if not already.
    /*
    if (isVector && !codegen.redVal.getType().isa<VectorType>()) {
      VectorType vtp = vectorType(codegen, codegen.redVal.getType());
      Value vred = genVectorReducInit(codegen, builder, loc, vtp);
      updateReduc(merger, codegen, vred);
    }
    */
    operands.push_back(codegen.redVal);
  }
  if (codegen.expValues)
    operands.push_back(codegen.expCount);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lo, hi, step, operands);
  if (codegen.redVal)
    updateReduc(merger, codegen, forOp.getRegionIterArgs().front());
  if (codegen.expValues)
    codegen.expCount = forOp.getRegionIterArgs().back();
  // Assign induction variable to sparse or dense index.
  Value iv = forOp.getInductionVar();
  if (isSparse)
    codegen.pidxs[tensor][idx] = iv;
  else
    codegen.loops[idx] = iv;
  builder.setInsertionPointToStart(forOp.getBody());
  // Share vector iteration mask between all subsequent loads/stores.
//  if (isVector)
//    codegen.curVecMask = genVectorMask(codegen, builder, iv, lo, hi, step);
  return forOp;
}

/// Emit a while-loop for co-iteration over multiple indices.
  Operation *genWhile(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                           linalg::GenericOp op, unsigned idx, bool needsUniv,
                           BitVector &indices) {
  SmallVector<Type, 4> types;
  SmallVector<Value, 4> operands;
  // Construct the while-loop with a parameter for each index.
  Type indexType = builder.getIndexType();
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      types.push_back(indexType);
      operands.push_back(codegen.pidxs[tensor][idx]);
    }
  }
  if (codegen.redVal) {
    types.push_back(codegen.redVal.getType());
    operands.push_back(codegen.redVal);
  }
  if (codegen.expValues) {
    types.push_back(indexType);
    operands.push_back(codegen.expCount);
  }
  if (needsUniv) {
    types.push_back(indexType);
    operands.push_back(codegen.loops[idx]);
  }
  assert(types.size() == operands.size());
  Location loc = op.getLoc();
  scf::WhileOp whileOp = builder.create<scf::WhileOp>(loc, types, operands);

  SmallVector<Location> locs(types.size(), loc);
  Block *before = builder.createBlock(&whileOp.getBefore(), {}, types, locs);
  Block *after = builder.createBlock(&whileOp.getAfter(), {}, types, locs);

  // Build the "before" region, which effectively consists
  // of a conjunction of "i < upper" tests on all induction.
  builder.setInsertionPointToStart(&whileOp.getBefore().front());
  Value cond;
  unsigned o = 0;
  for (unsigned b = 0, be = indices.size(); b < be; b++) {
    if (indices[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = before->getArgument(o);
      Value op2 = codegen.highs[tensor][idx];
      Value opc = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                op1, op2);
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, opc) : opc;
      codegen.pidxs[tensor][idx] = after->getArgument(o++);
    }
  }
  if (codegen.redVal)
    updateReduc(merger, codegen, after->getArgument(o++));
  if (codegen.expValues)
    codegen.expCount = after->getArgument(o++);
  if (needsUniv)
    codegen.loops[idx] = after->getArgument(o++);
  assert(o == operands.size());
  builder.create<scf::ConditionOp>(loc, cond, before->getArguments());
  builder.setInsertionPointToStart(&whileOp.getAfter().front());
  return whileOp;
}

/// Generates a for-loop or a while-loop, depending on whether it implements
/// singleton iteration or co-iteration over the given conjunction.
  Operation *genLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                          linalg::GenericOp op, std::vector<unsigned> &topSort,
                          unsigned at, bool needsUniv, BitVector &indices) {
  unsigned idx = topSort[at];
  if (indices.count() == 1) {
    bool isOuter = at == 0;
    bool isInner = at == topSort.size() - 1;
    return genFor(merger, codegen, builder, op, isOuter, isInner, idx, indices);
  }
  return genWhile(merger, codegen, builder, op, idx, needsUniv, indices);
}

/// Generates the local variables for this loop, consisting of the sparse
/// indices, restored universal dense index, and dense positions.
  void genLocals(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                      linalg::GenericOp op, std::vector<unsigned> &topSort,
                      unsigned at, bool needsUniv, BitVector &locals) {
  Location loc = op.getLoc();
  unsigned idx = topSort[at];

  // Initialize sparse indices.
  Value min;
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if (locals[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value ptr = codegen.indices[tensor][idx];
      Value s = codegen.pidxs[tensor][idx];
      Value load = genLoad(codegen, builder, loc, ptr, s);
      codegen.idxs[tensor][idx] = load;
      if (!needsUniv) {
        if (min) {
          Value cmp = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, load, min);
          min = builder.create<arith::SelectOp>(loc, cmp, load, min);
        } else {
          min = load;
        }
      }
    }
  }

  // Merge dense universal index over minimum.
  if (min) {
    assert(!needsUniv);
    codegen.loops[idx] = min;
  }

  // Initialize dense positions. Note that we generate dense indices of the
  // output tensor unconditionally, since they may not appear in the lattice,
  // but may be needed for linearized codegen.
  for (unsigned b = 0, be = locals.size(); b < be; b++) {
    if ((locals[b] || merger.isOutTensor(b, idx)) &&
        merger.isDim(b, Dim::kDense)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      unsigned pat = at;
      for (; pat != 0; pat--)
        if (codegen.pidxs[tensor][topSort[pat - 1]])
          break;
      Value p = (pat == 0) ? constantIndex(builder, loc, 0)
                           : codegen.pidxs[tensor][topSort[pat - 1]];
      codegen.pidxs[tensor][idx] = genAddress(
          codegen, builder, loc, codegen.sizes[idx], p, codegen.loops[idx]);
    }
  }

  // Move the insertion indices in lexicographic index order. During access
  // pattern expansion, we can skip setting the innermost dimension.
  if (codegen.sparseOut && !codegen.expValues) {
    Value pos = constantIndex(builder, loc, at);
    builder.create<memref::StoreOp>(loc, codegen.loops[idx], codegen.lexIdx,
                                    pos);
  }
}

/// Generates the induction structure for a while-loop.
  void genWhileInduction(Merger &merger, CodeGen &codegen,
                              OpBuilder &builder, linalg::GenericOp op,
                              unsigned idx, bool needsUniv,
                              BitVector &induction, scf::WhileOp whileOp) {
  Location loc = op.getLoc();
  // Finalize each else branch of all if statements.
  if (codegen.redVal || codegen.expValues) {
    while (auto ifOp = dyn_cast_or_null<scf::IfOp>(
               builder.getInsertionBlock()->getParentOp())) {
      unsigned y = 0;
      SmallVector<Value, 4> yields;
      if (codegen.redVal) {
        yields.push_back(codegen.redVal);
        updateReduc(merger, codegen, ifOp.getResult(y++));
      }
      if (codegen.expValues) {
        yields.push_back(codegen.expCount);
        codegen.expCount = ifOp->getResult(y++);
      }
      assert(y == yields.size());
      builder.create<scf::YieldOp>(loc, yields);
      builder.setInsertionPointAfter(ifOp);
    }
  }
  builder.setInsertionPointToEnd(&whileOp.getAfter().front());
  // Finalize the induction. Note that the induction could be performed
  // in the individual if-branches to avoid re-evaluating the conditions.
  // However, that would result in a rather elaborate forest of yield
  // instructions during code generation. Moreover, performing the induction
  // after the if-statements more closely resembles code generated by TACO.
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  Value one = constantIndex(builder, loc, 1);
  for (unsigned b = 0, be = induction.size(); b < be; b++) {
    if (induction[b] && merger.isDim(b, Dim::kSparse)) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value op1 = codegen.idxs[tensor][idx];
      Value op2 = codegen.loops[idx];
      Value op3 = codegen.pidxs[tensor][idx];
      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                op1, op2);
      Value add = builder.create<arith::AddIOp>(loc, op3, one);
      operands.push_back(builder.create<arith::SelectOp>(loc, cmp, add, op3));
      codegen.pidxs[tensor][idx] = whileOp->getResult(o++);
    }
  }
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, whileOp->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = whileOp->getResult(o++);
  }
  if (needsUniv) {
    operands.push_back(
        builder.create<arith::AddIOp>(loc, codegen.loops[idx], one));
    codegen.loops[idx] = whileOp->getResult(o++);
  }
  assert(o == operands.size());
  builder.create<scf::YieldOp>(loc, operands);
  builder.setInsertionPointAfter(whileOp);
}

/// Generates the induction structure for a for-loop.
  void genForInduction(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            Operation *loop) {
  Location loc = op.getLoc();
  unsigned o = 0;
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, loop->getResult(o++));
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = loop->getResult(o++);
  }
  assert(o == operands.size());
  if (o > 0)
    builder.create<scf::YieldOp>(loc, operands);
  builder.setInsertionPointAfter(loop);
}

/// Generates a single if-statement within a while-loop.
  scf::IfOp genIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned idx,
                       BitVector &conditions) {
  Location loc = op.getLoc();
  SmallVector<Type, 4> types;
  Value cond;
  for (unsigned b = 0, be = conditions.size(); b < be; b++) {
    if (conditions[b]) {
      unsigned tensor = merger.tensor(b);
      assert(idx == merger.index(b));
      Value clause;
      if (merger.isDim(b, Dim::kSparse)) {
        Value op1 = codegen.idxs[tensor][idx];
        Value op2 = codegen.loops[idx];
        clause = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               op1, op2);
      } else {
        clause = constantI1(builder, loc, true);
      }
      cond = cond ? builder.create<arith::AndIOp>(loc, cond, clause) : clause;
    }
  }
  if (codegen.redVal)
    types.push_back(codegen.redVal.getType());
  if (codegen.expValues)
    types.push_back(builder.getIndexType());
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, types, cond, /*else=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return ifOp;
}

/// Generates end of true branch of if-statement within a while-loop.
  void endIf(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                  linalg::GenericOp op, scf::IfOp ifOp, Operation *loop,
                  Value redInput, Value cntInput) {
  SmallVector<Value, 4> operands;
  if (codegen.redVal) {
    operands.push_back(codegen.redVal);
    updateReduc(merger, codegen, redInput);
  }
  if (codegen.expValues) {
    operands.push_back(codegen.expCount);
    codegen.expCount = cntInput;
  }
  if (!operands.empty())
    builder.create<scf::YieldOp>(op.getLoc(), operands);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
}

  bool startLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                         linalg::GenericOp op, std::vector<unsigned> &topSort,
                         unsigned exp, unsigned at, unsigned idx, unsigned ldx,
                         unsigned lts) {
  assert(codegen.curVecLength == 1);
  assert(!codegen.loops[idx]);
  // Emit invariants at this loop sequence level.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/true);
  // Emit access pattern expansion for sparse tensor output.
 // genExpansion(merger, codegen, builder, op, at, /*atStart=*/true);
  // Emit further intitialization at this loop sequence level.
  unsigned l0 = merger.set(lts)[0];
  bool needsUniv =
      genInit(merger, codegen, builder, op, topSort, at, merger.lat(l0).bits);
  // Maintain the universal index only if it is actually
  // consumed by a subsequent lattice point.
  if (needsUniv) {
    unsigned lsize = merger.set(lts).size();
    for (unsigned i = 1; i < lsize; i++) {
      unsigned li = merger.set(lts)[i];
      if (!merger.hasAnyDimOf(merger.lat(li).simple, Dim::kSparse))
        return true;
    }
  }
  return false;
}

  Operation *startLoop(Merger &merger, CodeGen &codegen,
                            OpBuilder &builder, linalg::GenericOp op,
                            std::vector<unsigned> &topSort, unsigned at,
                            unsigned li, bool needsUniv) {
  assert(codegen.curVecLength == 1);
  // Emit the for/while-loop control.
  Operation *loop = genLoop(merger, codegen, builder, op, topSort, at,
                            needsUniv, merger.lat(li).simple);
  // Emit the locals for this loop.
  genLocals(merger, codegen, builder, op, topSort, at, needsUniv,
            merger.lat(li).bits);
  return loop;
}

  bool endLoop(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                    linalg::GenericOp op, Operation *loop, unsigned idx,
                    unsigned li, bool needsUniv) {
  codegen.curVecLength = 1;
  // End a while-loop.
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    genWhileInduction(merger, codegen, builder, op, idx, needsUniv,
                      merger.lat(li).bits, whileOp);
    return needsUniv;
  }
  // End a for-loop.
  genForInduction(merger, codegen, builder, op, loop);
  return false;
}

  void endLoopSeq(Merger &merger, CodeGen &codegen, OpBuilder &builder,
                       linalg::GenericOp op, unsigned exp, unsigned at,
                       unsigned idx, unsigned ldx) {
  assert(codegen.curVecLength == 1);
  codegen.loops[idx] = Value();
  // Bring a pending reduction back from SIMD form when sequence ends.
  /*
  if (codegen.redVal)
    if (auto vtp = codegen.redVal.getType().dyn_cast<VectorType>())
      updateReduc(merger, codegen, genVectorReducEnd(codegen, builder, op.getLoc(), vtp));
  */
  // Unmark bookkeeping of invariants and loop index.
  genInvariants(merger, codegen, builder, op, exp, ldx, /*atStart=*/false);
  // Finalize access pattern expansion for sparse tensor output.
//  genExpansion(merger, codegen, builder, op, at, /*atStart=*/false);
}

  void genStmt(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                    linalg::GenericOp op, std::vector<unsigned> &topSort,
                    unsigned exp, unsigned at) {
  // At each leaf, assign remaining tensor (sub)expression to output tensor.
  std::cerr << "topSort.size is " << topSort.size() << std::endl;
  if (at == topSort.size()) {
    unsigned ldx = topSort[at - 1];
    Value rhs = genExp(merger, codegen, rewriter, op, exp, ldx);
    genTensorStore(merger, codegen, rewriter, op, exp, rhs);
    return;
  }

  // Construct iteration lattices for current loop index, with L0 at top.
  unsigned idx = topSort[at];
  unsigned ldx = at == 0 ? -1u : topSort[at - 1];
  unsigned lts = merger.optimizeSet(merger.buildLattices(exp, idx));
  std::cerr << "start loop seq" << std::endl;
  // Start a loop sequence.
  bool needsUniv = startLoopSeq(merger, codegen, rewriter, op, topSort, exp, at,
                                idx, ldx, lts);
  
  // Emit a loop for every lattice point L0 >= Li in this loop sequence.
  unsigned lsize = merger.set(lts).size();
  for (unsigned i = 0; i < lsize; i++) {
    // Start a loop.
    unsigned li = merger.set(lts)[i];
    std::cerr << "start loop of level " << i << std::endl;
    Operation *loop =
        startLoop(merger, codegen, rewriter, op, topSort, at, li, needsUniv);

    // Visit all lattices points with Li >= Lj to generate the
    // loop-body, possibly with if statements for coiteration.
    Value redInput = codegen.redVal;
    Value cntInput = codegen.expCount;
    std::cerr << "gen merge lattice inside loop level " << i << std::endl;
    bool isWhile = dyn_cast<scf::WhileOp>(loop) != nullptr;
    for (unsigned j = 0; j < lsize; j++) {
      unsigned lj = merger.set(lts)[j];
      unsigned ej = merger.lat(lj).exp;
      if (li == lj || merger.latGT(li, lj)) {
        // Recurse into body of each branch.
        if (isWhile) {
          scf::IfOp ifOp =
              genIf(merger, codegen, rewriter, op, idx, merger.lat(lj).simple);
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
          endIf(merger, codegen, rewriter, op, ifOp, loop, redInput, cntInput);
        } else {
          genStmt(merger, codegen, rewriter, op, topSort, ej, at + 1);
        }
      }
    }

    // End a loop.
    needsUniv =
        endLoop(merger, codegen, rewriter, op, loop, idx, li, needsUniv);
  }

  // End a loop sequence.
  endLoopSeq(merger, codegen, rewriter, op, exp, at, idx, ldx);
}

  void genResult(Merger &merger, CodeGen &codegen, RewriterBase &rewriter,
                      linalg::GenericOp op) {
  std::cerr << "Enter genResult() " << std::endl;
  OpOperand *lhs = op.getOutputOperand(0);
  Type resType = lhs->get().getType();
  assert(!getSparlayEncoding(resType));

    // To rematerialize an non-annotated tensor, simply load it
    // from the bufferized value.
  Value val = codegen.buffers.back(); // value array
  rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, resType, val);
}

void testAffineMap(linalg::GenericOp op) {
  int count = 0;
  for(OpOperand *t : op.getInputAndOutputOperands()) {
    std::cerr << "The " << count << "th operand. " << std::endl;
    count++; 
    auto map = op.getTiedIndexingMap(t);
    auto enc = getSparlayEncoding(t->get().getType());
    if (enc == nullptr) {
      continue;
    }
    auto crd = enc.getCrdMap();
    unsigned rank = map.getNumResults();
    std::cerr << "rank of current map is : " << rank << std::endl;
    for(unsigned d = 0; d < rank; d++) {
      unsigned d1 = perm(enc, d);
      std::cerr << "perm(end, " << d << ") is "<< d1 << std::endl;
      AffineExpr a = map.getResult(d1);
      unsigned idx = a.cast<AffineDimExpr>().getPosition();
      std::cerr << "position is: " << idx << std::endl;
    }
  }
}

struct GenericOpSparlayCodegen : public OpRewritePattern<linalg::GenericOp> {
public:
  GenericOpSparlayCodegen(MLIRContext *context) : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
 
    assert(op.getNumOutputs() == 1);
    unsigned numTensors = op.getNumInputsAndOutputs();
    unsigned numLoops = op.iterator_types().getValue().size();
    Merger merger(numTensors, numLoops);
    if(!findSparseAnnotations(merger, op)) {
      return failure();
    }

    std::vector<unsigned> topSort;

    if (!computeIterationGraph(merger, op, topSort, SortMask::kIncludeAll) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kIncludeUndef) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kIncludeDense) &&
        !computeIterationGraph(merger, op, topSort, SortMask::kSparseOnly)) {
      return failure();
    }

    Optional<unsigned> optExp = merger.buildTensorExpFromLinalg(op);
    if (!optExp.has_value())
      return failure();
    unsigned exp = optExp.value(); 

    OpOperand *sparseOut = nullptr;
    unsigned outerParNest = 0;
    if (!isAdmissableTensorExp(merger, op, topSort, exp, &sparseOut,
                               outerParNest))
      return failure();
    merger.setHasSparseOut(sparseOut != nullptr);
    CodeGen codegen(numTensors, numLoops, sparseOut, outerParNest);
//    std::cout << "Start executing testAffineMap " << std::endl;
//    testAffineMap(op);
    genBuffers(merger, codegen, rewriter, op);
    genStmt(merger, codegen, rewriter, op, topSort, exp, 0);
    genResult(merger, codegen, rewriter, op);
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
} //namespace sparlay
} //namespace mlir

std::unique_ptr<Pass> mlir::sparlay::createSparlayCodegenPass() {
  // std::cerr << "Create SparlayCodegenPass pointer" << std::endl;
  return std::make_unique<SparlayCodegenPass>();
}

void mlir::sparlay::populateSparlayCodegenPatterns(RewritePatternSet &patterns) {
  patterns.add<GenericOpSparlayCodegen>(patterns.getContext());
}
