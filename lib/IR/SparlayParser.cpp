#include "IR/SparlayParser.h"

using namespace mlir;

namespace mlir {
namespace sparlay {

namespace parser {

static std::vector<StringRef> mapDimtoID;
static std::vector<StringRef> mapSymtoID;

AffineExpr parseAffineLowPrecOpExpr(AsmParser& parser,
                                    AffineExpr llhs,
                                    AffineLowPrecOp llhsOp);
AffineExpr parseAffineOperandExpr(AsmParser& parser, AffineExpr lhs);
AffineExpr parseAffineExpr(AsmParser& parser);

AffineLowPrecOp consumeIfLowPrecOp(AsmParser& parser) {
  if (suc(parser.parseOptionalPlus())) {
    return AffineLowPrecOp::Add;
  }
  if (suc(parser.parseOptionalKeyword("minus"))) {
    return AffineLowPrecOp::Sub;
  }
  else {
    return AffineLowPrecOp::LNoOp;
  }
}

AffineHighPrecOp consumeIfHighPrecOp(AsmParser& parser) {
  if (suc(parser.parseOptionalStar()))
    return AffineHighPrecOp::Mul;
  else if (suc(parser.parseOptionalKeyword("floordiv")))
    return AffineHighPrecOp::FloorDiv;
  else if (suc(parser.parseOptionalKeyword("ceildiv")))
    return AffineHighPrecOp::CeilDiv;
  else if (suc(parser.parseOptionalKeyword("mod")))
    return AffineHighPrecOp::Mod;
  else
    return AffineHighPrecOp::HNoOp;
}

AffineExprKind convertHighPrecOp(AffineHighPrecOp src) {
  switch(src) {
    case AffineHighPrecOp::Mul:
      return AffineExprKind::Mul;
    case AffineHighPrecOp::FloorDiv:
      return AffineExprKind::FloorDiv;
    case AffineHighPrecOp::CeilDiv:
      return AffineExprKind::CeilDiv;
    case AffineHighPrecOp::Mod:
      return AffineExprKind::Mod;
    default:
      assert(0);
      return AffineExprKind::LAST_AFFINE_BINARY_OP;
  }
}

AffineExpr parseAffineHighPrecOpExpr(AsmParser& parser,
                                     AffineExpr llhs,
                                     AffineHighPrecOp llhsOp,
                                     SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(parser, llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = parser.getCurrentLocation();
  if (AffineHighPrecOp op = consumeIfHighPrecOp(parser)) {
    if (llhs) {
      AffineExpr expr = getAffineBinaryOpExpr(convertHighPrecOp(llhsOp), llhs, lhs);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(parser, expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(parser, lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(convertHighPrecOp(llhsOp), llhs, lhs);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

AffineExpr parseParentheticalExpr(AsmParser& parser) {
  if (suc(parser.parseOptionalRParen()))
    return parser.emitError(parser.getNameLoc(), "no expression inside parentheses"), nullptr;

  auto expr = parseAffineExpr(parser);
  if (!expr || failed(parser.parseOptionalRParen()))
    return nullptr;

  return expr;
}

AffineExpr parseBareIdExpr(AsmParser& parser, const std::string& key) {
  for (size_t dim = 0; dim < mapDimtoID.size(); ++dim) {
    if (mapDimtoID[dim] == key) {
      return getAffineDimExpr(dim, parser.getContext());
    }
  }
  for (size_t dim = 0; dim < mapSymtoID.size(); ++dim) {
    if (mapSymtoID[dim] == key) {
      return getAffineSymbolExpr(dim, parser.getContext());
    }
  }
  return parser.emitError(parser.getNameLoc(), "use of undeclared identifier"), nullptr;
}

AffineExpr parseNegateExpression(AsmParser& parser, AffineExpr lhs) {
  // if (parseToken(Token::minus, "expected '-'"))
  //   return nullptr;
  AffineExpr operand = parseAffineOperandExpr(parser, lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand)
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return parser.emitError(parser.getNameLoc(), "missing operand of negation"), nullptr;
  return (-1) * operand;
}

AffineExpr parseIntegerExpr(AsmParser& parser, int64_t val) {
  return getAffineConstantExpr(val, parser.getContext());
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
AffineExpr parseAffineOperandExpr(AsmParser& parser, AffineExpr lhs) {
  StringRef key;
  if (suc(parser.parseOptionalKeyword(&key, ArrayRef<StringRef>(mapDimtoID)))) {
    return parseBareIdExpr(parser, key.str());
  }
  if (suc(parser.parseOptionalKeyword(&key, ArrayRef<StringRef>(mapSymtoID)))) {
    return parseBareIdExpr(parser, key.str());
  }
  int64_t val = 0;
  if (parser.parseOptionalInteger<int64_t>(val).hasValue()) {
    return parseIntegerExpr(parser, val);
  }
  if (suc(parser.parseOptionalLParen())) {
    return parseParentheticalExpr(parser);
  }
  if (suc(parser.parseOptionalKeyword("minus"))) {
    return parseNegateExpression(parser, lhs);
  }
  if (lhs) {
    parser.emitError(parser.getNameLoc(), "Missing right operand of binary operator");
  } else {
    parser.emitError(parser.getNameLoc(), "Expected affine expression");
  }
  return nullptr;
}

AffineExpr getAffineBinaryOpExprNeedConvert(AffineLowPrecOp llhsOp, AffineExpr llhs, AffineExpr lhs) {
  if (llhsOp == AffineLowPrecOp::Sub) {
    return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
  } else if (llhsOp == AffineLowPrecOp::Add) {
    return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
  }
  assert(0);
  return nullptr;
}

AffineExpr parseAffineLowPrecOpExpr(AsmParser& parser,
                                    AffineExpr llhs,
                                    AffineLowPrecOp llhsOp) {
  AffineExpr lhs;
  if (!(lhs = parseAffineOperandExpr(parser, llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp(parser)) {
    AffineExpr sum;
    if (llhs) {
      if (llhsOp == AffineLowPrecOp::Sub) {
        sum = getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
      } else if (llhsOp == AffineLowPrecOp::Add) {
        sum = getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
      } else {
        assert(0);
      }
      return parseAffineLowPrecOpExpr(parser, sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(parser, lhs, lOp);
  }
  auto opLoc = parser.getCurrentLocation();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp(parser)) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes = parseAffineHighPrecOpExpr(parser, lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr =
        llhs ? getAffineBinaryOpExprNeedConvert(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp(parser))
      return parseAffineLowPrecOpExpr(parser, expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs) {
    if (llhsOp == AffineLowPrecOp::Sub) {
      return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, -lhs);
    } else if (llhsOp == AffineLowPrecOp::Add) {
      return getAffineBinaryOpExpr(AffineExprKind::Add, llhs, lhs);
    } else {
      assert(0);
    }
  }
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

AffineExpr parseAffineExpr(AsmParser& parser) {
  return parseAffineLowPrecOpExpr(parser, nullptr, AffineLowPrecOp::LNoOp);
}

//parse affine map in the form (*)[*]->(*)
AffineMap parseAffineMapWithKeyword(AsmParser& parser, const ArrayRef<StringRef>& opTokens, std::vector< std::vector<bool> >& vis) {

  mapDimtoID.clear();
  mapSymtoID.clear();
  std::vector<StringRef>* curMap = nullptr;
  
  auto parseName = [&]() -> ParseResult {
    assert(curMap != nullptr);
    StringRef key;
    if (failed(parser.parseOptionalKeyword(&key))) return failure();
    curMap->push_back(key);
    return success();
  };
  //parse dimension
  curMap = &mapDimtoID;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseName))) {
    return parser.emitError(parser.getNameLoc(), "Expected dimensions"), AffineMap();
  }
  //parse symbols (TODO: support ssaId)
  if (suc(parser.parseOptionalLSquare())) {
    curMap = &mapSymtoID;
    if (failed(parser.parseCommaSeparatedList(parseName))) {
      return parser.emitError(parser.getNameLoc(), "Expected symbols"), AffineMap();
    }
    if (failed(parser.parseRSquare())) {
      return parser.emitError(parser.getNameLoc(), "Expected \']\'"), AffineMap();
    }
  }
  //parse results
  if (failed(parser.parseArrow())) {
    return parser.emitError(parser.getNameLoc(), "Expected arrow"), AffineMap();
  }

  std::vector<AffineExpr> results;

  auto parseAffineExprHandler = [&]() -> ParseResult {
    StringRef opKeyRef;
    vis.push_back(std::vector<bool>(opTokens.size()));
    size_t id = vis.size()-1;
    for (size_t i = 0; i < opTokens.size(); ++i) {
      vis[id][i] = 0;
    }
    while (suc(parser.parseOptionalKeyword(&opKeyRef, ArrayRef<StringRef>(opTokens)))) {
      for (size_t i = 0; i < opTokens.size(); ++i) {
        if (opTokens[i] == opKeyRef && !vis[id][i]) {
          vis[id][i] = 1;
        }
      }
    }
    auto expr = parseAffineExpr(parser);
    if (expr == nullptr) {
      return failure();
    }
    results.push_back(expr);
    return success();
  };
  assert(vis.size() == (size_t)0);
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseAffineExprHandler))) {
    return parser.emitError(parser.getNameLoc(), "Expected results"), AffineMap();
  }

  return AffineMap::get(mapDimtoID.size(), mapSymtoID.size(), results, parser.getContext());
}

} //end of namespace parser


} //end of namespace sparlay
} //end of namespace mlir