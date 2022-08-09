//===- SparlayDialect.cpp - Sparlay dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/SparlayDialect.h"
#include "IR/SparlayOps.h"
#include "IR/SparlayTypes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

using namespace mlir;
using namespace mlir::sparlay;
using namespace mlir::detail;
using namespace llvm;

#include "IR/SparlayOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "IR/SparlayAttr.cpp.inc"

//===----------------------------------------------------------------------===//
// Sparlay Attributes
//===----------------------------------------------------------------------===//

enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

#define suc(A) succeeded(A)

static std::vector<std::string> mapDimtoID;

Attribute SparlayCompressAttr::parse(AsmParser &parser, Type type) {
  auto fuseIndex = std::vector<int>{};
  auto trimIndex = std::vector<int>{};
  std::vector<int>* curIndex = nullptr;
  SmallVector<AffineExpr> secondaryExprs;
  std::string tok_trim = "trim", tok_fuse = "fuse";
  SmallVector<StringRef, 2> op_token{tok_trim, tok_fuse};
  mapDimtoID.clear();
  auto parseInteger = [&]() -> ParseResult {
    int64_t val;
    assert(curIndex != nullptr);
    if (parser.parseOptionalInteger<int64_t>(val).hasValue()) {
      curIndex->push_back(val);
      return ParseResult(success());
    } else {
      return ParseResult(failure());
    }
  };
  bool fused = 0;
  bool trimmed = 0;
  if (failed(parser.parseLess())) {
    return {};
  }
  StringRef op_key_ref;
  auto parseSingleCompress = [&]() -> ParseResult {
    auto ret = parser.parseOptionalKeyword(&op_key_ref, ArrayRef<StringRef>(op_token));
    if (succeeded(ret)) {
      if (std::string(op_key_ref.str()) == "fuse") {
        if (fused) return failure();
        fused = 1;
        curIndex = &fuseIndex;
        ret = parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseInteger);
        if (failed(ret)) {
          return ret;
        } else {
          std::sort(fuseIndex.begin(), fuseIndex.end());
        }
      } else {
        if (trimmed) return failure();
        trimmed = 1;
        curIndex = &trimIndex;
        if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Paren, parseInteger
        ))) {
          return failure();
        } else {
          std::sort(trimIndex.begin(), trimIndex.end());
        }
      }
      return success();
    }
    return failure();
  };
  if (failed(parser.parseCommaSeparatedList(parseSingleCompress))) {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  SparlayCompressAttr ret = SparlayCompressAttr::get(
    parser.getContext(), CompressMap(trimIndex, fuseIndex)
  );
  return ret;
}

Attribute SparlayEncodingAttr::parse(AsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  // std::cerr << "Enter Encoding Parse" << std::endl;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};
  AffineMap crdMap = {};
  CompressMap compressMap = {};
  unsigned bitWidth = 8;
  for (const auto& attr: dict) {
    if (attr.getName() == "crdMap") {
      auto affineAttr = attr.getValue().dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        return {};
      }
      crdMap = affineAttr.getValue();
    } else if (attr.getName() == "compressMap") {
      auto sparlayCompressAttr = attr.getValue().dyn_cast<SparlayCompressAttr>();
      if (!sparlayCompressAttr) {
        return {};
      }
      compressMap = sparlayCompressAttr.getValue();
      auto trimIndex = compressMap.getTrimIndex();
      auto fuseIndex = compressMap.getFuseIndex();
    } else if (attr.getName() == "bitWidth") {
      auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
      if (!intAttr) return {};
      bitWidth = intAttr.getInt();
    } else {
      return {};
    }
  }
  return parser.getChecked<SparlayEncodingAttr>(parser.getContext(), crdMap, compressMap, bitWidth);
}

void SparlayEncodingAttr::print(AsmPrinter &printer) const {
  const CompressMap& compressMap = getCompressMap();
  const AffineMap& crdMap = getCrdMap();
  printer << "crdMap: " << crdMap << ", ";
  printer << "trim_level(";
  const auto& trimIndex = compressMap.getTrimIndex();
  for (size_t i = 0; i < trimIndex.size(); ++i) {
    printer << trimIndex[i];
    if (i != trimIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "fuse_level(";
  const auto& fuseIndex = compressMap.getFuseIndex();
  for (size_t i = 0; i < fuseIndex.size(); ++i) {
    printer << fuseIndex[i];
    if (i != fuseIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "bitWidth: " << getBitWidth();
}

void SparlayCompressAttr::print(AsmPrinter &printer) const {
  return;
}

LogicalResult SparlayEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    AffineMap primaryMap, CompressMap secondaryMap, unsigned bitWidth
) {
  return success();
}

LogicalResult SparlayEncodingAttr::verifyEncoding(
  ArrayRef<int64_t> shape, Type elementType, function_ref<InFlightDiagnostic()> emitError
) const {
  return success();
}

// SparlayEncodingAttr getSparlayEncoding(Type type) {
//   if (auto ttp = type.dyn_cast<RankedTensorType>())
//     return ttp.getEncoding().dyn_cast_or_null<SparlayEncodingAttr>();
//   return nullptr;
// }

//===----------------------------------------------------------------------===//
// Sparlay dialect.
//===----------------------------------------------------------------------===//

void SparlayDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/SparlayAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/SparlayOps.cpp.inc"
      >();
}