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
  // SmallVector<AffineExpr> secondaryExprs;
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
  std::vector< std::vector<AffineExpr> > indirectExpr;
  if (failed(parser.parseLess())) {
    return {};
  }
  StringRef indirectTok = "indirect";
  std::vector<StringRef> opTokens = {indirectTok};
  auto amap = sparlay::parser::parseAffineMapWithKeyword(parser, opTokens, vis, indirectExpr);
  for (size_t i = 0; i < vis.size(); i++) {
    assert(vis[i].size() == (size_t)1);
    isIndirect.push_back(vis[i][0]);
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return SparlayCrdAttr::get(parser.getContext(), CrdMap(amap, isIndirect, indirectExpr));
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
  SumPrim sumVal = {};
  EnumeratePrim enumVal = {};
  SchedulePrim schedVal = {};
  ReorderPrim reorderVal = {};
  IndirectFunc sched_name = {};
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
    }  else if (attr.getName() == "indirectFunc") {
      auto schedAttr = attr.getValue().dyn_cast<SparlayIndirectAttr>();
      sumVal = schedAttr.getSumVal();
      enumVal = schedAttr.getEnumVal();
      schedVal = schedAttr.getSchedVal();
      reorderVal = schedAttr.getReorderVal();
      sched_name.setIndirectFunc(sumVal, enumVal, schedVal, reorderVal);
    } else {
      return {};
    }
  }
  return parser.getChecked<SparlayEncodingAttr>(parser.getContext(), crdMap, compressMap, bitWidth, sched_name);
}

Attribute SparlayIndirectAttr::parse(AsmParser &parser, Type type) {
  std::cout << "in SparlayIndirectAttr::parse\n";
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  // std::cerr << "Enter Encoding Parse" << std::endl;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};
  SumPrim sumVal = {};
  EnumeratePrim enumVal = {};
  SchedulePrim schedVal = {};
  ReorderPrim reorderVal = {};
  for (const auto& attr: dict) {
    if (attr.getName() == "sumVal") {
      if (auto sumVal_attr = attr.getValue().dyn_cast<SparlaySumAttr>())
        sumVal = sumVal_attr.getValue();
    } else if (attr.getName() == "enumVal") {
      if (auto enumVal_attr = attr.getValue().dyn_cast<SparlayEnumerateAttr>())
        enumVal = enumVal_attr.getValue();
    } else if (attr.getName() == "schedVal") {
      if (auto schedVal_attr = attr.getValue().dyn_cast<SparlayScheduleAttr>())
        schedVal = schedVal_attr.getValue();
    }  else if (attr.getName() == "reorderVal") {
      if (auto reorderVal_attr = attr.getValue().dyn_cast<SparlayReorderAttr>())
        reorderVal = reorderVal_attr.getValue();
    } else {
      return {};
    }
  }
  return parser.getChecked<SparlayIndirectAttr>(parser.getContext(), sumVal, enumVal, schedVal, reorderVal);
}

Attribute SparlaySumAttr::parse(AsmParser &parser, Type type) {
  return SparlaySumAttr::get(parser.getContext(), {});
}

Attribute SparlayEnumerateAttr::parse(AsmParser &parser, Type type) {
  return SparlayEnumerateAttr::get(parser.getContext(), {});
}

Attribute SparlayScheduleAttr::parse(AsmParser &parser, Type type) {
  return SparlayScheduleAttr::get(parser.getContext(), {});
}

Attribute SparlayReorderAttr::parse(AsmParser &parser, Type type) {
  return SparlayReorderAttr::get(parser.getContext(), {});
}

void SparlayEncodingAttr::print(AsmPrinter &printer) const {
  const CompressMap& compressMap = getCompressMap();
  const CrdMap& crdMap = getCrdMap();
  printer << " crdMap: { ";
  const auto& isIndirect = crdMap.getIsIndirect();
  const auto& indirectExpr = crdMap.getIndirectExpr();
  // printer << (AffineMap)(crdMap) << "; ";
  // print AffineMap with indirect maps
  // Dimension identifiers.
  printer << '(';
  for (int i = 0; i < (int)crdMap.getNumDims() - 1; ++i)
    printer << 'd' << i << ", ";
  if (crdMap.getNumDims() >= 1)
    printer << 'd' << crdMap.getNumDims() - 1;
  printer << ')';

  // Symbolic identifiers.
  if (crdMap.getNumSymbols() != 0) {
    printer << '[';
    for (unsigned i = 0; i < crdMap.getNumSymbols() - 1; ++i)
      printer << 's' << i << ", ";
    if (crdMap.getNumSymbols() >= 1)
      printer << 's' << crdMap.getNumSymbols() - 1;
    printer << ']';
  }

  // Result affine expressions.
  printer << " -> (";
  // interleaveComma(map.getResults(),
  //                 [&](AffineExpr expr) { printAffineExpr(expr); });
  bool first_level = true;
  for (size_t i = 0; i < crdMap.getResults().size(); i++) {
    if (!first_level) printer << ", ";
    if (isIndirect[i]) {
      printer << "indirect (";
      bool first_indirect_level = true;
      for (size_t j = 0; j < indirectExpr[i].size(); j++) {
        if (!first_indirect_level) printer << ',';
        printer << (AffineExpr)(indirectExpr[i][j]);
        first_indirect_level = false;
      }
      printer << ")";
    } else {
      printer << (AffineExpr)(crdMap.getResults()[i]);
    }
    first_level = false;
  }
  printer << "); ";

  printer << "indirect_level(";
  bool first_indirect_level = true;
  for (size_t i = 0; i < isIndirect.size(); ++i) {
    if (isIndirect[i]) {
      if (!first_indirect_level) printer << ',';
      printer << i;
      first_indirect_level = false;
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
  printer << ", indirectFunc: " << getIndirectFunc();
}

void SparlayCompressAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayCrdAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayIndirectAttr::print(AsmPrinter &printer) const {
  const SumPrim& sumPrim = getSumVal();
  const EnumeratePrim& enumPrim = getEnumVal();
  const SchedulePrim& schedulePrim = getSchedVal();
  const ReorderPrim& reorderPrim = getReorderVal();
  printer << " indirectFunc: { ";
  if (!sumPrim.getIsEmpty()) {
    const auto& sumGroupBy = sumPrim.getGroupBy();
    const auto& sumValMap = sumPrim.getValMap();
    printer << "group-by (";
    for (size_t i = 0; i < sumGroupBy.size()-1; i++) {
      printer << sumGroupBy[i] << ", ";
    }
    printer << sumGroupBy[sumGroupBy.size()-1]<<"), ";
    printer << "valMap (";
    for (auto i = sumValMap.begin(); i != --sumValMap.end(); i++) {
      printer << i->first << " -> " << i->second << "; ";
    }
    printer << sumValMap.end()->first << " -> ";
    printer << sumValMap.end()->second << ")";
    printer << "}\n";
  }
  if (!enumPrim.getIsEmpty()) {
    const auto& enumGroupBy = enumPrim.getGroupBy();
    const auto& enumTraverseBy = enumPrim.getTraverseBy();
    const auto& enumValMap = enumPrim.getValMap();
    printer << "group-by (";
    for (size_t i = 0; i < enumGroupBy.size()-1; i++) {
      printer << enumGroupBy[i] << ", ";
    }
    printer << enumGroupBy[enumGroupBy.size()-1]<<"), ";
    printer << "traverse-by (";
    for (size_t i = 0; i < enumTraverseBy.size()-1; i++) {
      printer << enumTraverseBy[i] << ", ";
    }
    printer << enumTraverseBy[enumTraverseBy.size()-1]<<"), ";
    printer << "valMap (";
    for (auto i = enumValMap.begin(); i != --enumValMap.end(); i++) {
      printer << i->first << " -> " << i->second << "; ";
    }
    printer << enumValMap.end()->first << " -> ";
    printer << enumValMap.end()->second << ")";
    printer << "}\n";
  }
  if (!reorderPrim.getIsEmpty()) {
    const auto& reorderTraverseBy = reorderPrim.getTraverseBy();
    const auto& reorderWorkload = reorderPrim.getWorkload();
    const auto& reorderOrder = reorderPrim.getOrder();
    printer << "traverse-by (";
    for (size_t i = 0; i < reorderTraverseBy.size()-1; i++) {
      printer << reorderTraverseBy[i] << ", ";
    }
    printer << reorderTraverseBy[reorderTraverseBy.size()-1]<<"), ";
    printer << reorderWorkload << ", ";
    if (reorderOrder)
      printer << "order = ascend";
    else
      printer << "order = descend";
    printer << "}\n";
  }
  if (!schedulePrim.getIsEmpty()) {
    const auto& schedTraverseBy = schedulePrim.getTraverseBy();
    const auto& schedWorkload = schedulePrim.getWorkload();
    const auto& schedBucket = schedulePrim.getBucket();
    printer << "traverse-by (";
    for (size_t i = 0; i < schedTraverseBy.size()-1; i++) {
      printer << schedTraverseBy[i] << ", ";
    }
    printer << schedTraverseBy[schedTraverseBy.size()-1]<<"), ";
    printer << schedWorkload << ", " << schedBucket;
    printer << "}\n";
  }
}

void SparlaySumAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayEnumerateAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayScheduleAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

void SparlayReorderAttr::print(AsmPrinter &printer) const {
  printer << "HII";
  return;
}

LogicalResult SparlayEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    CrdMap primaryMap, CompressMap secondaryMap, unsigned bitWidth, IndirectFunc indirectFunc
) {
  return success();
}

LogicalResult SparlayEncodingAttr::verifyEncoding(
  ArrayRef<int64_t> shape, Type elementType, function_ref<InFlightDiagnostic()> emitError
) const {
  return success();
}

 SparlayEncodingAttr mlir::sparlay::getSparlayEncoding(Type type) {
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