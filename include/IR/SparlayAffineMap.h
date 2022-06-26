#ifndef SPARLAY_SPARLAYAFFINEMAP_H
#define SPARLAY_SPARLAYAFFINEMAP_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include <vector>
#include <iostream>

namespace mlir {
namespace sparlay {
class SparlayAffineMap : public AffineMap {
public:

  SparlayAffineMap(): AffineMap(), trimIndex({}), fuseIndex({}) {}

  explicit SparlayAffineMap(AffineMap map, std::vector<int> _trimIndex, std::vector<int> _fuseIndex): 
    AffineMap(map), trimIndex(_trimIndex), fuseIndex(_fuseIndex) {
    }
  
  bool operator == (const SparlayAffineMap& A) const {
    if ((AffineMap)(*this) != (AffineMap)A) {
      return 0;
    }
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
    this->dump();
    std::cerr << "TrimIndex: ";
    for (auto ele: trimIndex) {
      std::cerr << ele << ' ';
    }
    std::cerr << std::endl;
  }

private:
  std::vector<int> trimIndex;
  std::vector<int> fuseIndex;
};
} //endof sparlay
} //endof mlir

#endif