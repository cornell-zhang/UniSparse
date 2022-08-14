#pragma once

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "AsmParser/Parser.h"
#include "AsmParser/Token.h"
#include <iostream>

using namespace mlir;

namespace mlir {

namespace sparlay {

namespace parser {

#define suc(A) succeeded(A)

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

AffineExpr parseAffineExpr(AsmParser& parser);

AffineMap parseAffineMapWithKeyword(AsmParser& parser, const ArrayRef<StringRef>& opTokens, std::vector< std::vector<bool> >& vis);


}

} //end of namespace sparlay
} //end of namespace mlir