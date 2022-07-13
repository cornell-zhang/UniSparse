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

  explicit CompressMap(std::vector<int> _trimIndex, std::vector<int> _fuseIndex): 
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
} //endof sparlay
} //endof mlir

#endif