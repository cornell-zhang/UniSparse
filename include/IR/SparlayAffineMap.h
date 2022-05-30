#ifndef SPARLAY_SPARLAYAFFINEMAP_H
#define SPARLAY_SPARLAYAFFINEMAP_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace sparlay {
class SparlayAffineMap : public AffineMap {
public:

  SparlayAffineMap(): AffineMap(), trimIndex({}), fuseIndex({}) {}

  explicit SparlayAffineMap(AffineMap map, llvm::SmallVector<int>& _trimIndex, llvm::SmallVector<int>& _fuseIndex): 
    AffineMap(map), trimIndex(_trimIndex), fuseIndex(_fuseIndex) {}

  llvm::SmallVector<int> getTrimIndex() const { return trimIndex; }
  llvm::SmallVector<int> getFuseIndex() const { return fuseIndex; }

private:
  llvm::SmallVector<int> trimIndex;
  llvm::SmallVector<int> fuseIndex;
};
} //endof sparlay
} //endof mlir

#endif