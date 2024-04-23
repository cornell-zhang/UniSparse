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

  /// funcName -> {input operand names}.
  std::vector<Module> moduleTable;

  /// fifoName, fifoSize -> stage
  std::vector<Stream> streamTable;

  /// The counters start from 1 and increment by 1.
  /// The tensor dims starts from 1. The root dim is 0.
  unsigned read_operand_cnt = 1;
  unsigned S1_cnt = 1, S2_cnt = 1, S3_cnt = 1, S4_cnt = 1, S5_cnt = 1, S6_cnt = 1;
  unsigned write_operand_cnt = 1;
  std::string nnz_length;
private:
  raw_indented_ostream os;
};
} // namespace

static LogicalResult printOperation(HLSEmitter &emitter, ModuleOp moduleOp) {
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(HLSEmitter &emitter, func::FuncOp funcOp) {
  Region::BlockListType &blocks = funcOp.getBlocks();
  for (Block &block : blocks) {
    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op)))
        return failure();
    }
  }
  return success();
}

std::string genTopModule(HLSEmitter &emitter) {
  std::string topModule;
  topModule += R"(
void top()";

  for (Stream &stream: emitter.streamTable) {
    if (stream.IO)
      topModule += R"(
    tapa::mmap<)"+stream.dataType+R"(> )"+stream.name+R"(,)";
  }
  for (Stream &stream: emitter.streamTable) {
    if (stream.IO)
      topModule += R"(
    const int )"+stream.size+R"(,)";
  }

  /// remove the last `,`
  topModule.pop_back();
  topModule += R"(
){)";

  for (Stream &stream: emitter.streamTable) {
    if (!stream.IO)
      topModule += R"(
    tapa::stream<)"+stream.dataType+R"(, FIFO_DEPTH> )"+stream.name+R"((")"+stream.name+R"(");)";
  }

  topModule += R"(
  tapa::task())";

  for (Module &module: emitter.moduleTable) {
    string join = module.join?"join":"detach";
    topModule += R"(
      .invoke<tapa::)"+join+R"(>()"+R"(
        )"+module.name;
    for (Stream &inStream: module.inStreams) 
      topModule += R"(,
        )"+inStream.name;
    for (Stream &outStream: module.outStreams) 
      topModule += R"(,
        )"+outStream.name;
    for (string &size: module.sizes)
      topModule += R"(,
        )"+size;
    topModule += R"(
        ))";
  }
  topModule += R"(
    ;
}
)";
  return topModule;
}

LogicalResult genInitMemReadSparse(linalg::GenericOp &op, HLSEmitter &emitter, UniSparseEncodingAttr &encoding, AffineMap &indMap) {
  /// Generate initial memory read for sparse tensor.
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
  Stream sparse_val, sparse_metadata;
  sparse_val.name = "sparse"+to_string(emitter.read_operand_cnt)+"_val";
  sparse_val.dataType = sparse_val.name + "_T";
  std::transform(sparse_val.dataType.begin(), sparse_val.dataType.end(), 
    sparse_val.dataType.begin(), std::ptr_fun<int, int>(std::toupper));
  emitter.nnz_length = sparse_val.size = sparse_val.name + "_size";
  sparse_val.indMap = indMap;
  sparse_val.crdMap = crdMap;
  sparse_val.compressMap = compressMap;
  sparse_val.val = true;
  sparse_val.stage = 1;
  sparse_val.IO = true;
  emitter.streamTable.push_back(sparse_val);
  string sparse_metadata_datatype = "sparse"+to_string(emitter.read_operand_cnt)+"_meta_T";
  std::transform(sparse_metadata_datatype.begin(), sparse_metadata_datatype.end(), 
    sparse_metadata_datatype.begin(), std::ptr_fun<int, int>(std::toupper));
  for (unsigned d = 0; d < rank; d++) {
    /// gets the dim position of unisparse crdMap encoding and linalg index_map
    unsigned dim = indMap.getDimPosition(crdMap.getDimPosition(d)); 
    sparse_metadata.name = "sparse"+to_string(emitter.read_operand_cnt)+"_dim"+to_string(dim);
    sparse_metadata.dataType = sparse_metadata_datatype;
    sparse_metadata.size = sparse_metadata.name + "_size";
    sparse_metadata.indMap = indMap;
    sparse_metadata.crdMap = crdMap;
    sparse_metadata.compressMap = compressMap;
    sparse_metadata.level = dim;
    sparse_metadata.val = false;
    sparse_metadata.stage = 1;
    sparse_metadata.IO = true;
    if (trimVec[d] && mergeVec[d]) {
      /// Pointer and Crd
      sparse_metadata.name += "_ptr";
      Stream sparse_metadata_aux;
      sparse_metadata_aux.name = sparse_metadata.name + "_crd";
      sparse_metadata_aux.dataType = sparse_metadata_datatype;
      sparse_metadata_aux.size = sparse_metadata_aux.name + "_size";
      sparse_metadata_aux.indMap = indMap;
      sparse_metadata_aux.crdMap = crdMap;
      sparse_metadata_aux.compressMap = compressMap;
      sparse_metadata_aux.level = dim;
      sparse_metadata_aux.val = false;
      sparse_metadata_aux.stage = 1;
      sparse_metadata_aux.IO = true;
      emitter.streamTable.push_back(sparse_metadata_aux);
    } else if (trimVec[d]) {
      /// Crd
      sparse_metadata.name += "_crd";
    } else if (mergeVec[d]) {
      /// Pointer
      sparse_metadata.name += "_ptr";
    } else {
      /// Size
      emitError(op.getLoc(), "Unsupported sparse tensor encoding.");
    }
    emitter.streamTable.push_back(sparse_metadata);
  }
  return success();
}

LogicalResult genInitMemReadDense(linalg::GenericOp &op, HLSEmitter &emitter, Type &type, AffineMap &indMap) {
  /// Assumption: SpMV support only. May extend to SpMM with caveats.
  /// Generate initial memory read for flattened values of a dense tensor.
  unsigned rank = indMap.getNumResults();
  // assert(rank == 1); // if SpMV, assert the input is a vector
  Stream dense_val;
  dense_val.name = "dense"+to_string(emitter.read_operand_cnt)+"_val";
  dense_val.dataType = dense_val.name + "_T";
  std::transform(dense_val.dataType.begin(), dense_val.dataType.end(), 
    dense_val.dataType.begin(), std::ptr_fun<int, int>(std::toupper));
  dense_val.indMap = indMap;
  dense_val.val = true;
  dense_val.stage = 1;
  dense_val.IO = true;
  for (unsigned d = 0; d < rank; d++) {
    unsigned dim = indMap.getDimPosition(d);
    dense_val.size = "tensor_size_d"+std::to_string(dim); // should be a vector for SpMM
  }
  emitter.streamTable.push_back(dense_val);
  return success();
}

LogicalResult genReadInputStreams(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  std::vector<Stream> outStreams;
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 1) {
      /// Create a ReadIn module.
      Module readInModule;
      readInModule.name = "readIn"+to_string(emitter.S1_cnt++);
      readInModule.inStreams.push_back(stream);
      readInModule.sizes.push_back(stream.size);
      readInModule.join = true;
      /// Create the output stream.
      Stream readInOutStream(stream), readInOutStreamCopy(stream);
      readInOutStream.name = readInOutStreamCopy.name = readInModule.name+"_out";
      readInOutStream.IO = readInOutStreamCopy.IO = false;
      if (stream.val) {
        if (stream.compressMap == CompressMap()) // Dense Operand Val
          readInOutStream.stage = readInOutStreamCopy.stage = 4;
        else // Sparse Operand Value
          readInOutStream.stage = readInOutStreamCopy.stage = 3;
      } else {
        CrdMap crdMap = stream.crdMap;
        CompressMap compressMap = stream.compressMap;
        int level = stream.level;
        std::vector<int> trimInd = compressMap.getTrimIndex();
        std::vector<int> mergeInd = compressMap.getFuseIndex();
        bool hasTrim = std::find(trimInd.begin(), trimInd.end(), level) != trimInd.end();
        bool hasMerge = std::find(mergeInd.begin(), mergeInd.end(), level) != mergeInd.end();
        if (hasTrim && hasMerge) { // Pointer and Crd
          if (stream.name.substr(stream.name.size()-3) == "ptr")
            readInOutStream.stage = readInOutStreamCopy.stage = 2;
          else if (stream.name.substr(stream.name.size()-3) == "crd")
            readInOutStream.stage = readInOutStreamCopy.stage = 3;
        } else if (hasTrim) { // Crd
          readInOutStream.stage = readInOutStreamCopy.stage = 3;
        } else if (hasMerge) { // Pointer
          readInOutStream.stage = readInOutStreamCopy.stage = 2;
        } else {
          emitError(op.getLoc(), "Unsupported sparse tensor encoding.");
        }
      }
      readInModule.outStreams.push_back(readInOutStream);
      outStreams.push_back(readInOutStream);
      emitter.moduleTable.push_back(readInModule);
      os << emitter.fix_modules.stream_read(readInModule);
    }
  }
  for (auto &stream: outStreams) {
    emitter.streamTable.push_back(stream);
  }
  return success();
}

LogicalResult genDecompression(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  std::vector<Stream> outStreams;
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 2) {
      /// Create a Decompression module.
      Module decompModule;
      decompModule.name = "decompression"+to_string(emitter.S2_cnt++);
      decompModule.inStreams.push_back(stream);
      decompModule.sizes.push_back(stream.size);
      decompModule.join = true;
      /// Create the output stream.
      Stream decompOutStream(stream);
      decompOutStream.name = decompModule.name+"_out";
      decompOutStream.size = emitter.nnz_length;
      decompOutStream.IO = false;
      decompOutStream.stage = 3;
      outStreams.push_back(decompOutStream);
      decompModule.outStreams.push_back(decompOutStream);
      emitter.moduleTable.push_back(decompModule);
      os << emitter.fix_modules.repeater(decompModule);
    }
  }
  for (auto &stream: outStreams) {
    emitter.streamTable.push_back(stream);
  }
  return success();
}

LogicalResult genIndexCalculation(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  // Generate one index calculation module for all streams.
  Module indexCalcModule;
  indexCalcModule.name = "index_calc"+to_string(emitter.S3_cnt++);
  indexCalcModule.join = false; // all input streams of the same length
  unsigned rank;
  Stream valStream, metadataStream;
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 3) {
      indexCalcModule.inStreams.push_back(stream);
      if (stream.val) {
        valStream = stream;
      } else {
        metadataStream = stream;
        rank = stream.indMap.getNumResults();
      }
    }
  }
  unsigned outStreamNum = 1;
  for (unsigned i = 0; i < rank; i++) {
    Stream indexCalcOutStream(metadataStream);
    indexCalcOutStream.name = indexCalcModule.name+"_out"+to_string(outStreamNum++);
    indexCalcOutStream.size = emitter.nnz_length;
    // indexCalcOutStream.crdMap.setAffineMap(indexCalcOutStream.indMap);
    indexCalcOutStream.level = i;
    indexCalcOutStream.val = false;
    indexCalcOutStream.stage = 4;
    indexCalcModule.outStreams.push_back(indexCalcOutStream);
    emitter.streamTable.push_back(indexCalcOutStream);
  }
  Stream outValStream(valStream);
  outValStream.name = indexCalcModule.name+"_out"+to_string(outStreamNum++);
  outValStream.stage = 4;
  indexCalcModule.outStreams.push_back(outValStream);
  emitter.streamTable.push_back(outValStream);
  emitter.moduleTable.push_back(indexCalcModule);
  os << emitter.fix_modules.index_calc(indexCalcModule);
  return success();
}

LogicalResult genPEMul(linalg::GenericOp &op, HLSEmitter &emitter, AffineMap &indMap) {
  raw_indented_ostream &os = emitter.ostream();
  string funcName = "PE_Mul";
  Module PEMulModule(funcName, {}, {}, {}, true);
  Stream outValStream, outCrdStream;
  int outDim = indMap.getDimPosition(0); // get the source level for the output dim
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 4) {
      PEMulModule.inStreams.push_back(stream);
      if (stream.level == outDim)
        outCrdStream = stream;
      if (stream.val && stream.compressMap != CompressMap()) // sparse val stream
        outValStream = stream;
    }
  }
  outCrdStream.name = funcName+"_out1";
  outCrdStream.stage = 5;
  outValStream.name = funcName+"_out2";
  outValStream.stage = 5;
  PEMulModule.outStreams.push_back(outCrdStream);
  PEMulModule.outStreams.push_back(outValStream);
  emitter.streamTable.push_back(outCrdStream);
  emitter.streamTable.push_back(outValStream);
  emitter.moduleTable.push_back(PEMulModule);
  os << emitter.fix_modules.PE_Mul(PEMulModule);
  return success();
}

LogicalResult genPESum(linalg::GenericOp &op, HLSEmitter &emitter, AffineMap &indMap) {
  raw_indented_ostream &os = emitter.ostream();
  string funcName = "PE_Sum";
  Module PESumModule(funcName, {}, {}, {}, true);
  Stream outValStream;
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 5) {
      PESumModule.inStreams.push_back(stream);
      if (stream.val)
        outValStream = stream;
      if (find(PESumModule.sizes.begin(), PESumModule.sizes.end(), stream.size) 
              == PESumModule.sizes.end()) {
        PESumModule.sizes.push_back(stream.size);
      }
    }
  }
  unsigned rank = indMap.getNumResults();
  outValStream.name = funcName+"_out";
  outValStream.indMap = indMap;
  outValStream.val = true;
  outValStream.stage = 6;
  /// Assume the output is a dense tensor.
  for (unsigned d = 0; d < rank; d++) {
    unsigned dim = indMap.getDimPosition(d);
    outValStream.size = "tensor_size_d"+std::to_string(dim);
    PESumModule.sizes.push_back(outValStream.size);
  }
  emitter.streamTable.push_back(outValStream);
  PESumModule.outStreams.push_back(outValStream);
  emitter.moduleTable.push_back(PESumModule);
  os << emitter.fix_modules.PE_Sum(PESumModule);
  return success();
}

LogicalResult genWriteOutputStreams(linalg::GenericOp &op, HLSEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  std::vector<Stream> outStreams;
  for (auto &stream: emitter.streamTable) {
    if (stream.stage == 6) {
      /// Create a WriteOut module.
      Module writeOutModule;
      writeOutModule.name = "writeOut"+to_string(emitter.S6_cnt++);
      writeOutModule.inStreams.push_back(stream);
      writeOutModule.sizes.push_back(stream.size);
      writeOutModule.join = true;
      /// Create the output stream.
      Stream writeOutStream(stream);
      writeOutStream.name = writeOutModule.name+"_out";
      writeOutStream.IO = true;
      writeOutStream.stage = -1;
      outStreams.push_back(writeOutStream);
      writeOutModule.outStreams.push_back(writeOutStream);
      emitter.moduleTable.push_back(writeOutModule);
      os << emitter.fix_modules.stream_write(writeOutModule);
    }
  }
  for (auto &stream: outStreams) {
    emitter.streamTable.push_back(stream);
  }
  return success();
}

static LogicalResult printOperation(HLSEmitter &emitter, unisparse::DeviceOp deviceOp) {
  raw_indented_ostream &os = emitter.ostream();
  if (deviceOp.target().str() != "HLS")
    emitError(deviceOp.getLoc(), "Only support HLS target now.");
  /// Emit device program header.
  os << emitter.fix_modules.device_header;
  os << emitter.fix_modules.async_read;
  os << emitter.fix_modules.async_write;
  /// Walk through device op body, find linalg::generic op.
  for (Operation &op: deviceOp.body().front().getOperations()) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      // SmallVector<Type, 3> inputTypes;
      for (OpOperand *input: genericOp.getInputOperands()) {
        AffineMap map = genericOp.getTiedIndexingMap(input);
        assert(map.getNumResults() == genericOp.getRank(input));
        Value iVal = input->get();
        Type iType = iVal.getType();
        // inputTypes.push_back(iType);
        if (UniSparseEncodingAttr iEncoding = getUniSparseEncoding(iType)) {
          // sparse tensor type, decode data structures
          if (failed(genInitMemReadSparse(genericOp, emitter, iEncoding, map)))
            emitError(genericOp.getLoc(), "Failed to generate sparse tensor memory read.");
        } else {
          // dense tensor type
          if (failed(genInitMemReadDense(genericOp, emitter, iType, map)))
            emitError(genericOp.getLoc(), "Failed to generate dense tensor memory read.");
        }
        emitter.read_operand_cnt++;
      }

      /// stage I - Emit input streams and data preprocessing modules.
      if (failed(genReadInputStreams(genericOp, emitter)))
        emitError(genericOp.getLoc(), "Failed to generate input streams and data preprocessing modules.");

      /// Stage II - Decompression
      if (failed(genDecompression(genericOp, emitter)))
        emitError(genericOp.getLoc(), "Failed to generate decompression modules.");

      /// Stage III - Index Calculation
      if (failed(genIndexCalculation(genericOp, emitter)))
        emitError(genericOp.getLoc(), "Failed to generate index calculation modules.");

      /// Stage IV - Computation - Mul.
      /// Assumption: only support single output Linalg generic op.
      OpOperand *output = genericOp.getOutputOperand(0);
      AffineMap outMap = genericOp.getTiedIndexingMap(output);
      if (failed(genPEMul(genericOp, emitter, outMap)))
        emitError(genericOp.getLoc(), "Failed to generate PE Mul modules.");
      // for (auto &op: genericOp.getRegion().front().getOperations()) {
      //   LogicalResult status =
      //     llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      //       .Case<arith::MulFOp, arith::MulIOp>([&](auto op) { 
      //         return printS4MulOp(genericOp, emitter); })
      //       .Case<arith::AddFOp, arith::AddIOp>([&](auto op) { 
      //         return printS5AddOp(genericOp, emitter); })
      //       .Default([&](Operation *) {
      //         return op.emitOpError("Unsupported arithmetic operation.");
      //       });
      // }

      /// Stage V - Computation - Sum.
      if (failed(genPESum(genericOp, emitter, outMap)))
        emitError(genericOp.getLoc(), "Failed to generate PE Sum modules.");
      // /// Stage VI - Emit output streams.
      // // SmallVector<Type, 1> outputTypes;
      // for (OpOperand *output: genericOp.getOutputOperands()) {
      //   Value oVal = output->get();
      //   Type oType = oVal.getType();
      //   // outputTypes.push_back(oType);
      //   printS6DenseWrite(genericOp, emitter, oType);
      // }
      if (failed(genWriteOutputStreams(genericOp, emitter)))
        emitError(genericOp.getLoc(), "Failed to generate output streams and modules.");
      os << genTopModule(emitter);
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