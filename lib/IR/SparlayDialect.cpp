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

// AffineExpr parseAffineLowPrecOpExpr(DialectAsmParser& parser,
//                                     AffineExpr llhs,
//                                     AffineLowPrecOp llhsOp);
// AffineExpr parseAffineOperandExpr(DialectAsmParser& parser, AffineExpr lhs);
// AffineExpr parseAffineExpr(DialectAsmParser& parser);

// AffineLowPrecOp consumeIfLowPrecOp(DialectAsmParser& parser) {
//   if (suc(parser.parseOptionalPlus()))
//     return AffineLowPrecOp::Add;
//   else if (suc(parser.parseOptionalKeyword("-"))) {
//     return AffineLowPrecOp::Sub;
//   }
//   else {
//     return AffineLowPrecOp::LNoOp;
//   }
// }

// AffineHighPrecOp consumeIfHighPrecOp(DialectAsmParser& parser) {
//   if (suc(parser.parseOptionalStar()))
//     return AffineHighPrecOp::Mul;
//   else if (suc(parser.parseOptionalKeyword("floordiv")))
//     return AffineHighPrecOp::FloorDiv;
//   else if (suc(parser.parseOptionalKeyword("ceildiv")))
//     return AffineHighPrecOp::CeilDiv;
//   else if (suc(parser.parseOptionalKeyword("mod")))
//     return AffineHighPrecOp::Mod;
//   else
//     return AffineHighPrecOp::HNoOp;
// }

// AffineExprKind convertHighPrecOp(AffineHighPrecOp src) {
//   switch(src) {
//     case AffineHighPrecOp::Mul:
//       return AffineExprKind::Mul;
//     case AffineHighPrecOp::FloorDiv:
//       return AffineExprKind::FloorDiv;
//     case AffineHighPrecOp::CeilDiv:
//       return AffineExprKind::CeilDiv;
//     case AffineHighPrecOp::Mod:
//       return AffineExprKind::Mod;
//     default:
//       assert(0);
//       return AffineExprKind::LAST_AFFINE_BINARY_OP;
//   }
// }

// AffineExpr parseAffineHighPrecOpExpr(DialectAsmParser& parser,
//                                      AffineExpr llhs,
//                                      AffineHighPrecOp llhsOp,
//                                      SMLoc llhsOpLoc) {
//   AffineExpr lhs = parseAffineOperandExpr(parser, llhs);
//   if (!lhs)
//     return nullptr;

//   // Found an LHS. Parse the remaining expression.
//   auto opLoc = parser.getCurrentLocation();
//   if (AffineHighPrecOp op = consumeIfHighPrecOp(parser)) {
//     if (llhs) {
//       AffineExpr expr = getAffineBinaryOpExpr(convertHighPrecOp(llhsOp), llhs, lhs);
//       if (!expr)
//         return nullptr;
//       return parseAffineHighPrecOpExpr(parser, expr, op, opLoc);
//     }
//     // No LLHS, get RHS
//     return parseAffineHighPrecOpExpr(parser, lhs, op, opLoc);
//   }

//   // This is the last operand in this expression.
//   if (llhs)
//     return getAffineBinaryOpExpr(convertHighPrecOp(llhsOp), llhs, lhs);

//   // No llhs, 'lhs' itself is the expression.
//   return lhs;
// }

// AffineExpr parseParentheticalExpr(DialectAsmParser& parser) {
//   if (suc(parser.parseOptionalRParen()))
//     return parser.emitError(parser.getNameLoc(), "no expression inside parentheses"), nullptr;

//   auto expr = parseAffineExpr(parser);
//   if (!expr || failed(parser.parseOptionalRParen()))
//     return nullptr;

//   return expr;
// }

// AffineExpr parseBareIdExpr(DialectAsmParser& parser, const std::string& key) {
//   for (size_t dim = 0; dim < mapDimtoID.size(); ++dim) {
//     if (mapDimtoID[dim] == key) {
//       return getAffineDimExpr(dim, parser.getContext());
//     }
//   }
//   return parser.emitError(parser.getNameLoc(), "use of undeclared identifier"), nullptr;
// }

// AffineExpr parseNegateExpression(DialectAsmParser& parser, AffineExpr lhs) {
//   // if (parseToken(Token::minus, "expected '-'"))
//   //   return nullptr;
//   AffineExpr operand = parseAffineOperandExpr(parser, lhs);
//   // Since negation has the highest precedence of all ops (including high
//   // precedence ops) but lower than parentheses, we are only going to use
//   // parseAffineOperandExpr instead of parseAffineExpr here.
//   if (!operand)
//     // Extra error message although parseAffineOperandExpr would have
//     // complained. Leads to a better diagnostic.
//     return parser.emitError(parser.getNameLoc(), "missing operand of negation"), nullptr;
//   return (-1) * operand;
// }

// AffineExpr parseIntegerExpr(DialectAsmParser& parser, int64_t val) {
//   return getAffineConstantExpr(val, parser.getContext());
// }

// //TODO: FIXME:
// /// Parses an expression that can be a valid operand of an affine expression.
// /// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
// /// operator, the rhs of which is being parsed. This is used to determine
// /// whether an error should be emitted for a missing right operand.
// //  Eg: for an expression without parentheses (like i + j + k + l), each
// //  of the four identifiers is an operand. For i + j*k + l, j*k is not an
// //  operand expression, it's an op expression and will be parsed via
// //  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
// //  -l are valid operands that will be parsed by this function.
// AffineExpr parseAffineOperandExpr(DialectAsmParser& parser, AffineExpr lhs) {
//   StringRef key;
//   if (suc(parser.parseOptionalKeyword(&key))) {
//     return parseBareIdExpr(parser, key.str());
//   }
//   int64_t val;
//   if (!parser.parseOptionalInteger<int64_t>(val).hasValue()) {
//     return parseIntegerExpr(parser, val);
//   }
//   if (suc(parser.parseOptionalLParen())) {
//     return parseParentheticalExpr(parser);
//   }
//   StringRef a = "-";
//   if (suc(parser.parseOptionalKeyword(a))) {
//     return parseNegateExpression(parser, lhs);
//   }
//   if (lhs) {
//     parser.emitError(parser.getNameLoc(), "Missing right operand of binary operator");
//   } else {
//     parser.emitError(parser.getNameLoc(), "Expected affine expression");
//   }
//   return nullptr;
// }

// AffineExpr getAffineBinaryOpExprNeedConvert(AffineLowPrecOp llhsOp, AffineExpr llhs, AffineExpr lhs) {
//   if (llhsOp == AffineLowPrecOp::Sub) {
//     return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
//   } else if (llhsOp == AffineLowPrecOp::Add) {
//     return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
//   }
//   assert(0);
//   return nullptr;
// }

// AffineExpr parseAffineLowPrecOpExpr(DialectAsmParser& parser,
//                                     AffineExpr llhs,
//                                     AffineLowPrecOp llhsOp) {
//   AffineExpr lhs;
//   if (!(lhs = parseAffineOperandExpr(parser, llhs)))
//     return nullptr;

//   // Found an LHS. Deal with the ops.
//   if (AffineLowPrecOp lOp = consumeIfLowPrecOp(parser)) {
//     AffineExpr sum;
//     llhs.dump();
//     if (llhs) {
//       if (llhsOp == AffineLowPrecOp::Sub) {
//         sum = getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
//       } else if (llhsOp == AffineLowPrecOp::Add) {
//         sum = getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
//       } else {
//         assert(0);
//       }
//       return parseAffineLowPrecOpExpr(parser, sum, lOp);
//     }
//     // No LLHS, get RHS and form the expression.
//     return parseAffineLowPrecOpExpr(parser, lhs, lOp);
//   }
//   auto opLoc = parser.getCurrentLocation();
//   if (AffineHighPrecOp hOp = consumeIfHighPrecOp(parser)) {
//     // We have a higher precedence op here. Get the rhs operand for the llhs
//     // through parseAffineHighPrecOpExpr.
//     AffineExpr highRes = parseAffineHighPrecOpExpr(parser, lhs, hOp, opLoc);
//     if (!highRes)
//       return nullptr;

//     // If llhs is null, the product forms the first operand of the yet to be
//     // found expression. If non-null, the op to associate with llhs is llhsOp.
//     AffineExpr expr =
//         llhs ? getAffineBinaryOpExprNeedConvert(llhsOp, llhs, highRes) : highRes;

//     // Recurse for subsequent low prec op's after the affine high prec op
//     // expression.
//     if (AffineLowPrecOp nextOp = consumeIfLowPrecOp(parser))
//       return parseAffineLowPrecOpExpr(parser, expr, nextOp);
//     return expr;
//   }
//   // Last operand in the expression list.
//   if (llhs) {
//     if (llhsOp == AffineLowPrecOp::Sub) {
//       return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
//     } else if (llhsOp == AffineLowPrecOp::Add) {
//       return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
//     } else {
//       assert(0);
//     }
//   }
//   // No llhs, 'lhs' itself is the expression.
//   return lhs;
// }

// AffineExpr parseAffineExpr(DialectAsmParser& parser) {
//   return parseAffineLowPrecOpExpr(parser, nullptr, AffineLowPrecOp::LNoOp);
// }

// ParseResult parseAffineExprGetDim(
//   DialectAsmParser& parser, AffineExpr& expr, const std::vector<std::string>& mapDimtoID
// ) {
//   StringRef key;
//   auto ret = parser.parseKeyword(&key);
//   if (succeeded(ret)) {
//     bool mark = 0;
//     for (size_t dim = 0; dim < mapDimtoID.size(); ++dim) {
//       if (mapDimtoID[dim] == key.str()) {
//         mark = 1;
//         expr = getAffineDimExpr(dim, parser.getContext());
//       }
//     }
//     if (!mark) ret = ParseResult(failure());
//   }
//   return ret;
// }

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