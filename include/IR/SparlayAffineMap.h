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
      // std::cerr << (void*)this << std::endl;
    }

  std::vector<int> getTrimIndex() const { return trimIndex; }
  std::vector<int> getFuseIndex() const { return fuseIndex; }

private:
  std::vector<int> trimIndex;
  std::vector<int> fuseIndex;
};
} //endof sparlay
} //endof mlir

#endif