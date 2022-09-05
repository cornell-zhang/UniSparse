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
#include "IR/SparlayParser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"


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



Attribute SparlayCompressAttr::parse(AsmParser &parser, Type type) {
  auto fuseIndex = std::vector<int>{};
  auto trimIndex = std::vector<int>{};
  std::vector<int>* curIndex = nullptr;
  SmallVector<AffineExpr> secondaryExprs;
  std::string tok_trim = "trim", tok_fuse = "fuse";
  SmallVector<StringRef, 2> op_token{tok_trim, tok_fuse};
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
      } else if (std::string(op_key_ref.str()) == "trim") {
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
      } else {
        return parser.emitError(parser.getNameLoc(), "Expected \"trim\" or \"fuse\""), failure();
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

Attribute SparlayCrdAttr::parse(AsmParser &parser, Type type) {
  auto isIndirect = std::vector<bool>{};
  std::vector< std::vector<bool> > vis;
  if (failed(parser.parseLess())) {
    return {};
  }
  StringRef indirectTok = "indirect";
  std::vector<StringRef> opTokens = {indirectTok};
  auto amap = sparlay::parser::parseAffineMapWithKeyword(parser, opTokens, vis);
  assert(vis[0].size() == (size_t)1);
  isIndirect = vis[0];
  if (failed(parser.parseGreater())) {
    return {};
  }
  return SparlayCrdAttr::get(parser.getContext(), CrdMap(amap, isIndirect));
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
  CrdMap crdMap = {};
  CompressMap compressMap = {};
  unsigned bitWidth = 8;
  std::string sched_name = "";
  for (const auto& attr: dict) {
    if (attr.getName() == "crdMap") {
      auto crdAttr = attr.getValue().dyn_cast<SparlayCrdAttr>();
      if (!crdAttr) {
        return {};
      }
      crdMap = crdAttr.getValue();
    } else if (attr.getName() == "compressMap") {
      auto sparlayCompressAttr = attr.getValue().dyn_cast<SparlayCompressAttr>();
      if (!sparlayCompressAttr) {
        return {};
      }
      compressMap = sparlayCompressAttr.getValue();
    } else if (attr.getName() == "bitWidth") {
      auto intAttr = attr.getValue().dyn_cast<IntegerAttr>();
      if (!intAttr) return {};
      bitWidth = intAttr.getInt();
    }  else if (attr.getName() == "sched") {
      auto schedAttr = attr.getValue().dyn_cast<StringAttr>();
      sched_name = schedAttr.getValue();
    } else {
      return {};
    }
  }
  return parser.getChecked<SparlayEncodingAttr>(parser.getContext(), crdMap, compressMap, bitWidth, sched_name);
}

void SparlayEncodingAttr::print(AsmPrinter &printer) const {
  const CompressMap& compressMap = getCompressMap();
  const CrdMap& crdMap = getCrdMap();
  printer << " crdMap: { ";
  const auto& isIndirect = crdMap.getIsIndirect();
  printer << (AffineMap)(crdMap) << "; ";
  printer << "indirect_level(";
  for (size_t i = 0; i < isIndirect.size(); ++i) {
    if (isIndirect[i]) {
      printer << i;
      if (i != isIndirect.size()-1) printer << ',';
    }
  }
  printer << ") }, ";
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
  printer << ", schedule: " << getSched();
}

void SparlayCompressAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayCrdAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

LogicalResult SparlayEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    CrdMap primaryMap, CompressMap secondaryMap, unsigned bitWidth, std::string sched
) {
  return success();
}

LogicalResult SparlayEncodingAttr::verifyEncoding(
  ArrayRef<int64_t> shape, Type elementType, function_ref<InFlightDiagnostic()> emitError
) const {
  return success();
}

 SparlayEncodingAttr getSparlayEncoding(Type type) {
   if (auto ttp = type.dyn_cast<RankedTensorType>())
     return ttp.getEncoding().dyn_cast_or_null<SparlayEncodingAttr>();
   return nullptr;
 }

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