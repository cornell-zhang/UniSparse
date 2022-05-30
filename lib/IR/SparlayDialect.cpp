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
#include "IR/SparlayAttr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

using namespace mlir;
using namespace mlir::sparlay;

#include "IR/SparlayOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "IR/SparlayAttr.cpp.inc"

//===----------------------------------------------------------------------===//
// Sparlay Attributes
//===----------------------------------------------------------------------===//

ParseResult parseAffineExprGetDim(
  DialectAsmParser& parser, AffineExpr& expr, const SmallVector<std::string>& mapDimtoID
) {
  StringRef key;
  auto ret = parser.parseKeyword(&key);
  if (succeeded(ret)) {
    bool mark = 0;
    for (size_t dim = 0; dim < mapDimtoID.size(); ++dim) {
      if (mapDimtoID[dim] == key.str()) {
        mark = 1;
        expr = getAffineDimExpr(dim, parser.getContext());
      }
    }
    if (!mark) ret = ParseResult(failure());
  }
  return ret;
}

Attribute SparlayAffineAttr::parse(DialectAsmParser &parser, Type type) {
  SmallVector<int> fuseIndex;
  SmallVector<int> trimIndex;
  SmallVector<AffineExpr> secondaryExprs;
  static std::string tok_trim = "trim", tok_fuse = "fuse";
  SmallVector<StringRef, 2> op_token{tok_trim, tok_fuse};
  SmallVector<std::string> mapDimtoID;
  int tot_dim = 0;
  auto parseElt = [&]() -> ParseResult {
    AffineExpr expr;
    StringRef key;
    auto ret = parser.parseKeyword(&key);
    if (succeeded(ret)) {
      expr = getAffineDimExpr(tot_dim++, parser.getContext());
      mapDimtoID.push_back(key.str());
    }
    return ret;
  };
  int image_idx = 0;
  auto parseElt2 = [&]() -> ParseResult {
    StringRef op_key_ref;
    bool fused = 0;
    bool trimmed = 0;
    while (1) {
      auto ret = parser.parseOptionalKeyword(&op_key_ref, ArrayRef<StringRef>(op_token));
      if (succeeded(ret)) {
        if (std::string(op_key_ref.str()) == "fuse") {
          if (!fused) fuseIndex.push_back(image_idx), fused = 1;
        } else {
          if (!trimmed) trimIndex.push_back(image_idx), trimmed = 1;
        }
      } else {
        break;
      }
    }
    AffineExpr expr;
    auto ret = parseAffineExprGetDim(parser, expr, mapDimtoID);
    if (succeeded(ret)) {
      image_idx++;
      secondaryExprs.push_back(expr);
    }
    return ret;
  };
  if (failed(parser.parseLess())) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt))) {
    return {};
  }
  if (failed(parser.parseArrow())) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseElt2))) {
    return {};
  }
  if (failed(parser.parseGreater())) {
    return {};
  }
  return SparlayAffineAttr::get(
    parser.getContext(), SparlayAffineMap(AffineMap::get(tot_dim, 0, secondaryExprs, parser.getContext()), trimIndex, fuseIndex)
  );
}

Attribute SparlayEncodingAttr::parse(DialectAsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  // Parse the data as a dictionary.
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  AffineMap primaryMap{};
  SparlayAffineMap secondaryMap{};
  unsigned bitWidth = 8;
  for (const auto& attr: dict) {
    if (attr.first == "primaryMap") {
      auto affineAttr = attr.second.dyn_cast<AffineMapAttr>();
      if (!affineAttr) {
        return {};
      }
      primaryMap = affineAttr.getValue();
    } else if (attr.first == "secondaryMap") {
      auto sparlayAffineAttr = attr.second.dyn_cast<SparlayAffineAttr>();
      if (!sparlayAffineAttr) {
        return {};
      }
      secondaryMap = sparlayAffineAttr.getValue();
    } else if (attr.first == "bitWidth") {
      auto intAttr = attr.second.dyn_cast<IntegerAttr>();
      if (!intAttr) return {};
      bitWidth = intAttr.getInt();
    } else {
      return {};
    }
  }
  if (failed(parser.parseGreater()))
    return {};
  return SparlayEncodingAttr::get(parser.getContext(), primaryMap, secondaryMap, bitWidth);
}

void SparlayEncodingAttr::print(DialectAsmPrinter &printer) const {
  const SparlayAffineMap& secondaryMap = getSecondaryMap();
  const AffineMap& primaryMap = getPrimaryMap();
  printer << "primaryMap: " << primaryMap << ", ";
  printer << "secondaryMap: " << AffineMap::get(secondaryMap.getNumDims(), secondaryMap.getNumSymbols(), secondaryMap.getResults(), secondaryMap.getContext()) << ", ";
  printer << "trim_level(";
  const auto& trimIndex = secondaryMap.getTrimIndex();
  for (size_t i = 0; i < trimIndex.size(); ++i) {
    printer << trimIndex[i];
    if (i != trimIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "fuse_level(";
  const auto& fuseIndex = secondaryMap.getFuseIndex();
  for (size_t i = 0; i < fuseIndex.size(); ++i) {
    printer << fuseIndex[i];
    if (i != fuseIndex.size()-1) printer << ',';
  }
  printer << "), ";
  printer << "bitWidth: " << getBitWidth();
}

void SparlayAffineAttr::print(DialectAsmPrinter &printer) const {
  return;
}

LogicalResult SparlayEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    AffineMap primaryMap, SparlayAffineMap secondaryMap, unsigned bitWidth
) {
  return success();
}

LogicalResult SparlayEncodingAttr::verifyEncoding(
  ArrayRef<int64_t> shape, Type elementType, function_ref<InFlightDiagnostic()> emitError
) const {
  return success();
}

Attribute SparlayDialect::parseAttribute(DialectAsmParser &parser,
                                              Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown sparse tensor layout attribute");
  return Attribute();
}

void SparlayDialect::printAttribute(Attribute attr,
                                         DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}

//===----------------------------------------------------------------------===//
// Sparlay dialect.
//===----------------------------------------------------------------------===//

void SparlayDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/SparlayAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/SparlayOps.cpp.inc"
      >();
}