#ifndef SPARLAY_SPARLAYAFFINEMAP_H
#define SPARLAY_SPARLAYAFFINEMAP_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include <vector>
#include <iostream>

namespace mlir {
namespace sparlay {
class CompressMap: public AffineMap {
public:

  CompressMap(): AffineMap(), trimIndex({}), fuseIndex({}) {}

  explicit CompressMap(const std::vector<int>& _trimIndex, const std::vector<int>& _fuseIndex): 
    AffineMap(), trimIndex(_trimIndex), fuseIndex(_fuseIndex) {}
  
  bool operator == (const CompressMap& A) const {
    auto dstTrim = A.getTrimIndex();
    auto dstFuse = A.getFuseIndex();
    if (trimIndex.size() != dstTrim.size()) return 0;
    if (fuseIndex.size() != dstFuse.size()) return 0;
    for (size_t i = 0; i < trimIndex.size(); ++i) {
      if (trimIndex[i] != dstTrim[i]) return 0;
    }
    for (size_t i = 0; i < fuseIndex.size(); ++i) {
      if (fuseIndex[i] != dstFuse[i]) return 0;
    }
    return 1;
  }

  std::vector<int> getTrimIndex() const { return trimIndex; }
  std::vector<int> getFuseIndex() const { return fuseIndex; }

  void Print() {
    std::cerr << "TrimLevel: ";
    for (auto ele: trimIndex) {
      std::cerr << ele << ' ';
    }
    std::cerr << std::endl;
    std::cerr << "FuseLevel: ";
    for (auto ele: fuseIndex) {
      std::cerr << ele << ' ';
    }
    std::cerr << std::endl;
  }

private:
  std::vector<int> trimIndex;
  std::vector<int> fuseIndex;
};

class CrdMap: public AffineMap {
  public:

  CrdMap(): AffineMap(), isIndirect({}), indirectExpr({}) {}

  explicit CrdMap(const AffineMap& amap, 
                  const std::vector<bool>& _isIndirect,
                  const std::vector< std::vector<AffineExpr> >& _indirectExpr): 
    AffineMap(amap), isIndirect(_isIndirect), indirectExpr(_indirectExpr) {}
  
  bool operator == (const CrdMap& A) const {
    if ((AffineMap)(*this) != (AffineMap)(A)) return 0;
    auto dstIndirect = A.getIsIndirect();
    auto dstIndirectExpr = A.getIndirectExpr();
    if (isIndirect.size() != dstIndirect.size()) return 0;
    if (indirectExpr.size() != dstIndirectExpr.size()) return 0;
    for (size_t i = 0; i < isIndirect.size(); ++i) {
      if (isIndirect[i] != dstIndirect[i]) return 0;
    }
    for (size_t i = 0; i < indirectExpr.size(); i++) {
      for (size_t j = 0; j < indirectExpr[i].size(); j++)
        if (indirectExpr[i][j] != dstIndirectExpr[i][j]) return 0;
    }
    return 1;
  }

  std::vector<bool> getIsIndirect() const { return this->isIndirect; }
  std::vector< std::vector<AffineExpr> > getIndirectExpr() const { return this->indirectExpr; }

  void Print() {
    this->dump();
    std::cerr << "isIndirect: (";
    for (auto ele: isIndirect) {
      std::cerr << ele << ',';
    }
    std::cerr << ')' << std::endl;
  }

private:
  std::vector<bool> isIndirect;
  std::vector< std::vector<AffineExpr> > indirectExpr;
};

} //endof sparlay
} //endof mlir

#endif