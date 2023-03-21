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

class SumPrim: public AffineMap {
public:

  SumPrim(): AffineMap(), groupBy({}), valMap({}), is_empty(true) {}

  explicit SumPrim(const std::vector<unsigned>& _groupBy, const std::map<std::string, int>& _valMap): 
    AffineMap(), groupBy(_groupBy), valMap(_valMap) {
      is_empty = false;
    }

  std::vector<unsigned> getGroupBy() const { return groupBy; }
  std::map<std::string, int> getValMap() const { return valMap; }
  bool getIsEmpty() const { return is_empty; }
  void setSumPrim(const std::vector<unsigned>& _groupBy, 
                  const std::map<std::string, int>& _valMap,
                  const bool& _isEmpty ) {
    this->groupBy = _groupBy;
    this->valMap = _valMap;
    this->is_empty = _isEmpty;
  }

private:
  std::vector<unsigned> groupBy;
  std::map<std::string, int> valMap;
  bool is_empty;
};

class EnumeratePrim: public AffineMap {
public:

  EnumeratePrim(): AffineMap(), groupBy({}), traverseBy({}), valMap({}), is_empty(true) {}

  explicit EnumeratePrim(const std::vector<unsigned>& _groupBy, const std::vector<unsigned>& _traverseBy, const std::map<std::string, std::string>& _valMap): 
    AffineMap(), groupBy(_groupBy), traverseBy(_traverseBy), valMap(_valMap) {
      is_empty = false;
    }

  std::vector<unsigned> getGroupBy() const { return groupBy; }
  std::vector<unsigned> getTraverseBy() const { return traverseBy; }
  std::map<std::string, std::string> getValMap() const { return valMap; }
  bool getIsEmpty() const { return is_empty; }
  void setEnumPrim(const std::vector<unsigned>& _groupBy, 
                  const std::vector<unsigned>& _traverseBy, 
                  const std::map<std::string, std::string>& _valMap,
                  const bool& _isEmpty ) {
    this->groupBy = _groupBy;
    this->traverseBy = _traverseBy;
    this->valMap = _valMap;
    this->is_empty = _isEmpty;
  }

private:
  std::vector<unsigned> groupBy;
  std::vector<unsigned> traverseBy;
  std::map<std::string, std::string> valMap;
  bool is_empty;
};

class SchedulePrim: public AffineMap {
public:

  SchedulePrim(): AffineMap(), traverseBy({}), workload({}), bucket(), is_empty(true) {}

  explicit SchedulePrim(const std::vector<unsigned>& _traverseBy, const std::string& _workload, const unsigned& _bucket): 
    AffineMap(), traverseBy(_traverseBy), workload(_workload), bucket(_bucket) {
      is_empty = false;
    }

  std::vector<unsigned> getTraverseBy() const { return traverseBy; }
  std::string getWorkload() const { return workload; }
  unsigned getBucket() const { return bucket; }
  bool getIsEmpty() const { return is_empty; }
  void setSchedPrim(
                  const std::vector<unsigned>& _traverseBy, 
                  const std::string& _workload,
                  const unsigned& _bucket,
                  const bool& _isEmpty ) {
    this->traverseBy = _traverseBy;
    this->workload = _workload;
    this->bucket = _bucket;
    this->is_empty = _isEmpty;
  }

private:
  std::vector<unsigned> traverseBy;
  std::string workload;
  unsigned bucket;
  bool is_empty;
};

class ReorderPrim: public AffineMap {
public:

  ReorderPrim(): AffineMap(), traverseBy(), workload({}), order(), is_empty(true) {}

  explicit ReorderPrim(const std::vector<unsigned>& _traverseBy, const std::string& _workload, const bool& _order): 
    AffineMap(), traverseBy(_traverseBy), workload(_workload), order(_order) {
      is_empty = false;
    }

  std::vector<unsigned> getTraverseBy() const { return traverseBy; }
  std::string getWorkload() const { return workload; }
  bool getOrder() const { return order; }
  bool getIsEmpty() const { return is_empty; }
  void setReorderPrim(
                  const std::vector<unsigned>& _traverseBy, 
                  const std::string& _workload,
                  const unsigned& _order,
                  const bool& _isEmpty ) {
    this->traverseBy = _traverseBy;
    this->workload = _workload;
    this->order = _order;
    this->is_empty = _isEmpty;
  }

private:
  std::vector<unsigned> traverseBy;
  std::string workload;
  bool order;
  bool is_empty;
};

class IndirectFunc: public AffineMap {
  public:

  IndirectFunc(): AffineMap(), sumPrim({}), enumeratePrim({}), schedulePrim({}), reorderPrim({}) {}

  explicit IndirectFunc(const AffineMap& amap, 
                  const SumPrim& _sumPrim,
                  const EnumeratePrim& _enumeratePrim,
                  const SchedulePrim& _schedulePrim,
                  const ReorderPrim& _reorderPrim
                  ): 
    AffineMap(amap), sumPrim(_sumPrim), enumeratePrim(_enumeratePrim), 
    schedulePrim(_schedulePrim), reorderPrim(_reorderPrim) {}

  void setIndirectFunc(SumPrim _sumPrim, EnumeratePrim _enumeratePrim, SchedulePrim _schedulePrim, ReorderPrim _reorderPrim) {
    sumPrim.setSumPrim(_sumPrim.getGroupBy(), _sumPrim.getValMap(), _sumPrim.getIsEmpty());
    enumeratePrim.setEnumPrim(_enumeratePrim.getGroupBy(), _enumeratePrim.getTraverseBy(), 
                              _enumeratePrim.getValMap(), _enumeratePrim.getIsEmpty());
    schedulePrim.setSchedPrim(_schedulePrim.getTraverseBy(), _schedulePrim.getWorkload(),
                              _schedulePrim.getBucket(), _schedulePrim.getIsEmpty());
    reorderPrim.setReorderPrim(_reorderPrim.getTraverseBy(), _reorderPrim.getWorkload(),
                               _reorderPrim.getOrder(), _reorderPrim.getIsEmpty());
  }
  SumPrim getSumPrim() const { return this->sumPrim; }
  EnumeratePrim getEnumeratePrim() const { return this->enumeratePrim; }
  SchedulePrim getSchedulePrim() const { return this->schedulePrim; }
  ReorderPrim getReorderPrim() const { return this->reorderPrim; }

private:
  SumPrim sumPrim;
  EnumeratePrim enumeratePrim;
  SchedulePrim schedulePrim;
  ReorderPrim reorderPrim;
};


} //endof sparlay
} //endof mlir

#endif