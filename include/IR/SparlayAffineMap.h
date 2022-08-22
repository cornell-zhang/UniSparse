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

  CrdMap(): AffineMap(), isIndirect({}) {}

  explicit CrdMap(const AffineMap& amap, const std::vector<bool>& _isIndirect): 
    AffineMap(amap), isIndirect(_isIndirect) {}
  
  bool operator == (const CrdMap& A) const {
    if ((AffineMap)(*this) != (AffineMap)(A)) return 0;
    auto dstIndirect = A.getIsIndirect();
    if (isIndirect.size() != dstIndirect.size()) return 0;
    for (size_t i = 0; i < isIndirect.size(); ++i) {
      if (isIndirect[i] != dstIndirect[i]) return 0;
    }
    return 1;
  }

  std::vector<bool> getIsIndirect() const { return this->isIndirect; }

  void Print() {
    std::cerr << "isIndirect: (";
    for (auto ele: isIndirect) {
      std::cerr << ele << ',';
    }
    std::cerr << ')' << std::endl;
  }

private:
  std::vector<bool> isIndirect;
};

} //endof sparlay
} //endof mlir

#endif