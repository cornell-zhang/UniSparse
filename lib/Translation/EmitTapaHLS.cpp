//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The UniSparse Authors.
//
//===----------------------------------------------------------------------===//

#include "Translation/EmitTapaHLS.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"

#include "IR/UniSparseDialect.h"
#include "IR/UniSparseOps.h"
#include "IR/UniSparseTypes.h"

#define DEBUG_TYPE "emit-tapa-hls"

using namespace mlir;
using namespace unisparse;
using namespace std;

//===----------------------------------------------------------------------===//
// Some Base Classes
//===----------------------------------------------------------------------===//

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct HLSEmitter {
  explicit HLSEmitter(raw_ostream &os);

  /// Emits operation 'op'.
  LogicalResult emitOperation(Operation &op);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Fixed module code.
  FixModules fix_modules;

  /// Function argument names -> top module fifo argument names.
  std::map<std::string, std::string> nameTable;

  /// funcName -> {input operand names}.
  std::map<std::string, std::vector<std::string>> taskList;

  /// Shared size consts and fifos.
  std::map<string, string> topIOs, topFifos;
  std::vector<std::string> topSizes;

  /// fifoName, fifoSize -> stage
  std::map<tuple<string, string, StreamAttr>, int> fifoStg;

  /// Top module fifo names.
  SmallVector<std::string, 8> fifoList;

  int64_t read_operand_cnt = 0;
private:
  raw_indented_ostream os;
};
} // namespace

static LogicalResult printOperation(HLSEmitter &emitter, ModuleOp moduleOp) {
  raw_indented_ostream &os = emitter.ostream();
  // os << "visit moduleOp\n";
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(HLSEmitter &emitter, func::FuncOp funcOp) {
  raw_indented_ostream &os = emitter.ostream();
  // os << "visit funcOp\n";
  Region::BlockListType &blocks = funcOp.getBlocks();
  for (Block &block : blocks) {
    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op)))
        return failure();
    }
  }
  return success();
}

std::string generateTopModule(HLSEmitter &emitter) {
  std::string topModule;
  topModule += R"(
void top()";
  for (auto pair = emitter.topIOs.begin(); pair != emitter.topIOs.end(); pair++) {
    topModule += R"(
      tapa::mmap<)"+pair->second+R"(> )"+pair->first+R"(,)";
  }
  for (auto const &size: emitter.topSizes) {
    topModule += R"(
      const int )"+size+R"(,)";
  }
  /// remove the last `,`
  topModule.pop_back();
  topModule += R"(
){)";
  for (auto pair = emitter.topFifos.begin(); pair != emitter.topFifos.end(); pair++) {
    topModule += R"(
      tapa::stream<)"+pair->second+R"(, FIFO_DEPTH> )"+pair->first+R"((")"+pair->first+R"(");)";
  }
  topModule += R"(
  tapa::task())";
  for (auto pair = emitter.taskList.begin(); pair != emitter.taskList.end(); pair++) {
    topModule += R"(
      .invoke<tapa::)"+pair->second[0]+R"(>()"+R"(
        )"+pair->first+R"(,
        )"+pair->second[1]+R"(,
        )"+pair->second[2]+R"(,
        )"+pair->second[3]+R"(
        ))";
  }
  topModule += R"(
  ;
}
)";
  return topModule;
}

void stream_read_args(HLSEmitter &emitter, std::string &funcName, std::string &dataType, std::string &mmapIn,
    std::string &fifoOut, const std::string dense_sparse, const std::string dstruct) {
  funcName = dense_sparse+std::to_string(emitter.read_operand_cnt)+dstruct;
  dataType = funcName+"_T";
  std::transform(dataType.begin(), dataType.end(), 
    dataType.begin(), std::ptr_fun<int, int>(std::toupper));
  mmapIn = funcName + "_mmapIn";
  fifoOut = funcName + "_fifoOut";
}

void printS1SparseRead(linalg::GenericOp &op, HLSEmitter &emitter, UniSparseEncodingAttr &encoding, AffineMap &indMap) {
  /// read in sparse data structures, data preprocessing
  raw_indented_ostream &os = emitter.ostream();
  CrdMap crdMap = encoding.getCrdMap();
  CompressMap compressMap = encoding.getCompressMap();
  assert(crdMap.isPermutation());
  assert(crdMap.getNumResults() == indMap.getNumResults());
  unsigned rank = indMap.getNumResults();
  std::vector<int> trimInd = compressMap.getTrimIndex();
  std::vector<int> mergeInd = compressMap.getFuseIndex();
  std::vector<bool> trimVec(rank, false), mergeVec(rank, false);
  for (int i: trimInd) {
    trimVec[i] = true;
  }
  for (int i: mergeInd) {
    mergeVec[i] = true; 
  }
  for (unsigned d = 0; d < rank; d++) {
    /// gets the dim position of unisparse crdMap encoding and linalg index_map
    unsigned dim = indMap.getDimPosition(crdMap.getDimPosition(d)); 
    std::string funcName, dataType, mmapIn, fifoOut, size;
    size = "size_d"+std::to_string(dim);
    if ( std::find(emitter.topSizes.begin(), emitter.topSizes.end(), size) 
      == emitter.topSizes.end() )
      emitter.topSizes.push_back(size);
    if (trimVec[d] && mergeVec[d]) {
      stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "sparse", "_ptr"+std::to_string(d));
      os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
      emitter.topIOs[mmapIn]=dataType;
      emitter.topFifos[fifoOut]=dataType;
      emitter.taskList[funcName].push_back("join");
      emitter.taskList[funcName].push_back(mmapIn);
      emitter.taskList[funcName].push_back(fifoOut);
      emitter.taskList[funcName].push_back(size);
      StreamAttr streamAttr(indMap, crdMap, compressMap, d, false);
      emitter.fifoStg[{fifoOut, size, streamAttr}]=2;
      stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "sparse", "_crd"+std::to_string(d));
      size = "size_crd"+std::to_string(d);
      os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
      emitter.topIOs[mmapIn]=dataType;
      emitter.topFifos[fifoOut]=dataType;
      emitter.taskList[funcName].push_back("join");
      emitter.taskList[funcName].push_back(mmapIn);
      emitter.taskList[funcName].push_back(fifoOut);
      emitter.taskList[funcName].push_back(size);
      emitter.fifoStg[{fifoOut, size, streamAttr}]=3;
    } else if (trimVec[d]) {
      stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "sparse", "_crd"+std::to_string(d));
      size = "size_crd"+std::to_string(d);
      os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
      emitter.topIOs[mmapIn]=dataType;
      emitter.topFifos[fifoOut]=dataType;
      emitter.taskList[funcName].push_back("join");
      emitter.taskList[funcName].push_back(mmapIn);
      emitter.taskList[funcName].push_back(fifoOut);
      emitter.taskList[funcName].push_back(size);
      StreamAttr streamAttr(indMap, crdMap, compressMap, d, false);
      emitter.fifoStg[{fifoOut, size, streamAttr}]=3;
    } else if (mergeVec[d]) {
      stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "sparse", "_ptr"+std::to_string(d));
      os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
      emitter.topIOs[mmapIn]=dataType;
      emitter.topFifos[fifoOut]=dataType;
      emitter.taskList[funcName].push_back("join");
      emitter.taskList[funcName].push_back(mmapIn);
      emitter.taskList[funcName].push_back(fifoOut);
      emitter.taskList[funcName].push_back(size);
      StreamAttr streamAttr(indMap, crdMap, compressMap, d, false);
      emitter.fifoStg[{fifoOut, size, streamAttr}]=2;
    } else {
      emitError(op.getLoc(), "Unsupported sparse tensor encoding.");
    }
  } 
  /// emit Value read in stream
  std::string funcName, dataType, mmapIn, fifoOut, size;
  stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "sparse", "_val");
  size = "size_val";
  os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
  emitter.topIOs[mmapIn]=dataType;
  emitter.topFifos[fifoOut]=dataType;
  emitter.taskList[funcName].push_back("join");
  emitter.taskList[funcName].push_back(mmapIn);
  emitter.taskList[funcName].push_back(fifoOut);
  emitter.taskList[funcName].push_back(size);
  StreamAttr streamAttr(indMap, crdMap, compressMap, -1, true);
  emitter.fifoStg[{fifoOut, size, streamAttr}]=4;
}

void printS1DenseRead(linalg::GenericOp &op, HLSEmitter &emitter, Type &type, AffineMap &indMap) {
  raw_indented_ostream &os = emitter.ostream();
  unsigned rank = indMap.getNumResults();
  /// we only support SpMV now.
  assert(rank == 1);
  unsigned dim = indMap.getDimPosition(0);
  std::string funcName, dataType, mmapIn, fifoOut, size;
  size = "size_d"+std::to_string(dim);
  if ( std::find(emitter.topSizes.begin(), emitter.topSizes.end(), size) 
    == emitter.topSizes.end() )
    emitter.topSizes.push_back(size);
  stream_read_args(emitter, funcName, dataType, mmapIn, fifoOut, "dense", "");
  os << emitter.fix_modules.stream_read(funcName, dataType, mmapIn, fifoOut, size);
  emitter.topIOs[mmapIn]=dataType;
  emitter.topFifos[fifoOut]=dataType;
  emitter.taskList[funcName].push_back("join");
  emitter.taskList[funcName].push_back(mmapIn);
  emitter.taskList[funcName].push_back(fifoOut);
  emitter.taskList[funcName].push_back(size);
  StreamAttr streamAttr(indMap, nullptr, nullptr, -1, true);
  emitter.fifoStg[{fifoOut, size, streamAttr}]=4;
}

void printS2Decompression(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  for(auto iter = emitter.fifoStg.begin(); iter != emitter.fifoStg.end(); iter++) {
    if (iter->second == 2) {
      string inFifoName = get<0>(iter->first);
      string size = get<1>(iter->first);
      StreamAttr streamAttr = get<2>(iter->first);
      string funcName = "repeater_"+inFifoName;
      string inFifoType = emitter.topFifos[inFifoName];
      string outFifoName = inFifoName+"_crd";
      os << emitter.fix_modules.repeater(funcName, inFifoType, inFifoName, inFifoType, outFifoName, size);
      string outFifoSize = outFifoName+"_size";
      emitter.topSizes.push_back(outFifoSize);
      emitter.topFifos[outFifoName]=inFifoType;
      emitter.taskList[funcName].push_back("join");
      emitter.taskList[funcName].push_back(inFifoName);
      emitter.taskList[funcName].push_back(outFifoName);
      emitter.taskList[funcName].push_back(size);
      emitter.fifoStg[{outFifoName, outFifoSize, streamAttr}]=3;
    }
  }
}

void printS3IndexCalculation(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  string funcName = "index_calc";
  emitter.taskList[funcName].push_back("detach");
  std::vector<pair</*name*/string, /*type*/string>> fifos;
  for(auto iter = emitter.fifoStg.begin(); iter != emitter.fifoStg.end(); iter++) {
    if (iter->second == 3) {
      string inFifoName = get<0>(iter->first);
      string size = get<1>(iter->first);
      StreamAttr streamAttr = get<2>(iter->first);
      string inFifoType = emitter.topFifos[inFifoName];
      fifos.push_back({inFifoName, inFifoType});
      emitter.fifoStg[{"out_"+inFifoName, size, streamAttr}]=4;
      emitter.topFifos["out_"+inFifoName]=inFifoType;
      emitter.taskList[funcName].push_back(inFifoName);
    }
  }
  os << emitter.fix_modules.index_calc(funcName, fifos);
  for (auto &fifo: fifos) {
    string outFifoName = "out_"+fifo.first;
    emitter.taskList[funcName].push_back(outFifoName);
  }
}

LogicalResult printS4MulOp(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  string funcName = "PEMul";
  emitter.taskList[funcName].push_back("detach");
  std::vector<tuple</*name*/string, /*type*/string, StreamAttr>> inFifos;
  std::vector<string> sizes;
  for(auto iter = emitter.fifoStg.begin(); iter != emitter.fifoStg.end(); iter++) {
    if (iter->second == 4) {
      string inFifoName = get<0>(iter->first);
      string size = get<1>(iter->first);
      StreamAttr streamAttr = get<2>(iter->first);
      string inFifoType = emitter.topFifos[inFifoName];
      inFifos.push_back({inFifoName, inFifoType, streamAttr});
      sizes.push_back(size);
    }
  }
  emitter.fifoStg[{"out_"+inFifoName, size, streamAttr}]=5;
  emitter.topFifos["out_"+inFifoName]=inFifoType;
  emitter.taskList[funcName].push_back(inFifoName);
}

LogicalResult printS5AddOp(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  string outputType;

  string funcName = "PEAdd";
}

static LogicalResult printOperation(HLSEmitter &emitter, unisparse::DeviceOp deviceOp) {
  raw_indented_ostream &os = emitter.ostream();
  // os << "visit deviceOp\n";
  if (deviceOp.target().str() != "HLS")
    emitError(deviceOp.getLoc(), "Only support HLS target now.");
  /// Emit device program header.
  os << emitter.fix_modules.device_header;
  os << emitter.fix_modules.async_read;
  os << emitter.fix_modules.async_write;
  /// Walk through device op body, find linalg::generic op.
  for (Operation &op: deviceOp.body().front().getOperations()) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      /// stage I - Emit input streams and data preprocessing modules.
      // SmallVector<Type, 3> inputTypes;
      for (OpOperand *input: genericOp.getInputOperands()) {
        AffineMap map = genericOp.getTiedIndexingMap(input);
        assert(map.getNumResults() == genericOp.getRank(input));
        Value iVal = input->get();
        Type iType = iVal.getType();
        // inputTypes.push_back(iType);
        if (UniSparseEncodingAttr iEncoding = getUniSparseEncoding(iType)) {
          // sparse tensor type, decode data structures
          printS1SparseRead(genericOp, emitter, iEncoding, map);
        } else {
          // dense tensor type
          printS1DenseRead(genericOp, emitter, iType, map);
        }
        emitter.read_operand_cnt++;
      }
      /// Stage II - Decompression
      printS2Decompression(genericOp, emitter);

      /// Stage III - Index Calculation
      printS3IndexCalculation(genericOp, emitter);

      /// Stage IV - Computation
      for (auto &op: genericOp.getRegion().front().getOperations()) {
        LogicalResult status =
          llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<arith::MulFOp, arith::MulIOp>([&](auto op) { 
              return printS4MulOp(genericOp, emitter); })
            .Case<arith::AddFOp, arith::AddIOp>([&](auto op) { 
              return printS5AddOp(genericOp, emitter); })
            .Default([&](Operation *) {
              return op.emitOpError("Unsupported arithmetic operation.");
            });
      }

      /// Stage V - Emit output streams.
      SmallVector<Type, 1> outputTypes;
      for (OpOperand *output: genericOp.getOutputOperands()) {
        Value oVal = output->get();
        Type oType = oVal.getType();
        outputTypes.push_back(oType);
      }
      os << generateTopModule(emitter);
    }
  }

  return success();
}

HLSEmitter::HLSEmitter(raw_ostream &in_os) : os(in_os) {
}

/// Top-level MLIR module emitter.
LogicalResult HLSEmitter::emitOperation(Operation &op) {
  
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { 
            return printOperation(*this, op); })
          // Func ops.
          .Case<func::FuncOp>([&](auto op) { 
            return printOperation(*this, op); })
          // Linalg ops.
          .Case<unisparse::DeviceOp>([&](auto op) { 
            return printOperation(*this, op); })
          // Default case.
          .Default([&](Operation *) {
            return success();
          });

  if (failed(status))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Entry of unisparse-translate
//===----------------------------------------------------------------------===//

LogicalResult unisparse::emitTapaHLS(Operation *op, llvm::raw_ostream &os) {
  HLSEmitter emitter(os);
  return emitter.emitOperation(*op);
}

void unisparse::registerEmitTapaHLSTranslation() {
  TranslateFromMLIRRegistration toTapaHLS(
      "emit-tapa-hls", 
      [](ModuleOp module, raw_ostream &output) {
        return unisparse::emitTapaHLS(module, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::unisparse::UniSparseDialect,
          mlir::tensor::TensorDialect,
          mlir::scf::SCFDialect,
          mlir::AffineDialect,
          mlir::math::MathDialect,
          mlir::memref::MemRefDialect,
          mlir::linalg::LinalgDialect,
          mlir::func::FuncDialect,
          mlir::vector::VectorDialect,
          mlir::bufferization::BufferizationDialect,
          mlir::LLVM::LLVMDialect>();
        // clang-format on
      });
}