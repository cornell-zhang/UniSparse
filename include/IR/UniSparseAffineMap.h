#ifndef UNISPARSE_UNISPARSEAFFINEMAP_H
#define UNISPARSE_UNISPARSEAFFINEMAP_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include <vector>
#include <iostream>

namespace mlir {
namespace unisparse {
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
    std::cout << "TrimLevel: ";
    for (auto ele: trimIndex) {
      std::cout << ele << ' ';
    }
    std::cout << std::endl;
    std::cout << "FuseLevel: ";
    for (auto ele: fuseIndex) {
      std::cout << ele << ' ';
    }
    std::cout << std::endl;
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
    std::cout << "isIndirect: (";
    for (auto ele: isIndirect) {
      std::cout << ele << ',';
    }
    std::cout << ')' << std::endl;
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
  ~SumPrim() {
    this->groupBy.clear();
    this->valMap.clear();
    this->is_empty = true;
  }

  bool operator == (const SumPrim& A) const {
    auto dstGroupBy = A.getGroupBy();
    auto dstValMap = A.getValMap();
    auto dstIsEmpty = A.getIsEmpty();
    if (this->is_empty != dstIsEmpty) return 0;
    if (this->groupBy.size() != dstGroupBy.size()) return 0;
    if (this->valMap.size() != dstValMap.size()) return 0;
    for (size_t i = 0; i < this->groupBy.size(); ++i) {
      if (this->groupBy[i] != dstGroupBy[i]) return 0;
    }
    auto valMapStart = dstValMap.begin();
    for (auto i = this->valMap.begin(); i != this->valMap.end(); i++) {
      if (i->first != valMapStart->first || i->second != valMapStart->second) return 0;
      valMapStart++;
    }
    return 1;
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
  void Print() {
    std::cout << "SumPrim GroupBy: (";
    for (size_t i = 0; i < this->groupBy.size()-1; i++) {
      std::cout << this->groupBy[i] << ", ";
    }
    std::cout << this->groupBy[this->groupBy.size()-1]<<")\n";
    std::cout << "SumPrim valMap: (";
    for (auto i = this->valMap.begin(); i != --(this->valMap.end()); i++) {
      std::cout << i->first << " -> " << i->second << "; ";
    }
    std::cout << (--this->valMap.end())->first << " -> ";
    std::cout << (--this->valMap.end())->second << ")\n";
    std::cout << "SumPrim is_empty = " << this->is_empty << "\n";
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
  ~EnumeratePrim() {
    this->groupBy.clear();
    this->traverseBy.clear();
    this->valMap.clear();
    this->is_empty = true;
  }

  bool operator == (const EnumeratePrim& A) const {
    auto dstGroupBy = A.getGroupBy();
    auto dstTraverseBy = A.getTraverseBy();
    auto dstValMap = A.getValMap();
    auto dstIsEmpty = A.getIsEmpty();
    if (this->groupBy.size() != dstGroupBy.size()) return 0;
    if (this->is_empty != dstIsEmpty) return 0;
    if (this->traverseBy.size() != dstTraverseBy.size()) return 0;
    if (this->valMap.size() != dstValMap.size()) return 0;
    for (size_t i = 0; i < this->groupBy.size(); ++i) {
      if (this->groupBy[i] != dstGroupBy[i]) return 0;
    }
    for (size_t i = 0; i < this->traverseBy.size(); ++i) {
      if (this->traverseBy[i] != dstTraverseBy[i]) return 0;
    }
    auto valMapStart = dstValMap.begin();
    for (auto i = this->valMap.begin(); i != this->valMap.end(); i++) {
      if (i->first != valMapStart->first || i->second != valMapStart->second) return 0;
      valMapStart++;
    }
    return 1;
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
  void Print() {
    if (!this->is_empty && this->groupBy.size() > 0) {
      std::cout << "EnumeratePrim GroupBy: (";
      for (size_t i = 0; i < this->groupBy.size()-1; i++) {
        std::cout << this->groupBy[i] << ", ";
      }
      std::cout << this->groupBy[this->groupBy.size()-1]<<")\n";
    }
    if (!this->is_empty && this->traverseBy.size() > 0) {
      std::cout << "EnumeratePrim traverseBy: (";
      for (size_t i = 0; i < this->traverseBy.size()-1; i++) {
        std::cout << this->traverseBy[i] << ", ";
      }
      std::cout << this->traverseBy[this->traverseBy.size()-1]<<")\n";
    }
    if (!this->is_empty && this->valMap.size() > 0) {
      std::cout << "EnumeratePrim valMap: (";
      for (auto i = this->valMap.begin(); i != --(this->valMap.end()); i++) {
        std::cout << i->first << " -> " << i->second << "; ";
      }
      std::cout << (--this->valMap.end())->first << " -> ";
      std::cout << (--this->valMap.end())->second << ")\n";
    }
    std::cout << "EnumeratePrim is_empty = " << this->is_empty << "\n";
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
  ~SchedulePrim() {
    this->traverseBy.clear();
    this->workload.clear();
    this->is_empty = true;
  }
  
  bool operator == (const SchedulePrim& A) const {
    auto dstTraverseBy = A.getTraverseBy();
    auto dstWorkload = A.getWorkload();
    auto dstBucket = A.getBucket();
    auto dstIsEmpty = A.getIsEmpty();
    if (this->is_empty != dstIsEmpty) return 0;
    if (this->traverseBy.size() != dstTraverseBy.size()) return 0;
    if (this->workload != dstWorkload) return 0;
    if (this->bucket != dstBucket) return 0;
    for (size_t i = 0; i < this->traverseBy.size(); ++i) {
      if (this->traverseBy[i] != dstTraverseBy[i]) return 0;
    }
    return 1;
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
  void Print() {
    if (!this->is_empty && this->traverseBy.size() > 0) {
      std::cout << "SchedulePrim traverseBy: (";
      for (size_t i = 0; i < this->traverseBy.size()-1; i++) {
        std::cout << this->traverseBy[i] << ", ";
      }
      std::cout << this->traverseBy[this->traverseBy.size()-1]<<")\n";
    }
    std::cout << "SchedulePrim workload = " << this->workload << "\n";
    std::cout << "SchedulePrim bucket = " << this->bucket << "\n";
    std::cout << "SchedulePrim is_empty = " << this->is_empty << "\n";
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
  ~ReorderPrim() {
    this->traverseBy.clear();
    this->workload.clear();
    this->is_empty = true;
  }

  bool operator == (const ReorderPrim& A) const {
    auto dstTraverseBy = A.getTraverseBy();
    auto dstWorkload = A.getWorkload();
    auto dstOrder = A.getOrder();
    auto dstIsEmpty = A.getIsEmpty();
    if (this->is_empty != dstIsEmpty) return 0;
    if (this->traverseBy.size() != dstTraverseBy.size()) return 0;
    if (this->workload != dstWorkload) return 0;
    if (this->order != dstOrder) return 0;
    for (size_t i = 0; i < this->traverseBy.size(); ++i) {
      if (this->traverseBy[i] != dstTraverseBy[i]) return 0;
    }
    return 1;
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
  void Print() {
    if (!this->is_empty && this->traverseBy.size() > 0) {
      std::cout << "ReorderPrim traverseBy: (";
      for (size_t i = 0; i < this->traverseBy.size()-1; i++) {
        std::cout << this->traverseBy[i] << ", ";
      }
      std::cout << this->traverseBy[this->traverseBy.size()-1]<<")\n";
    }
    std::cout << "ReorderPrim workload = " << this->workload << "\n";
    std::cout << "ReorderPrim order = " << this->order << "\n";
    std::cout << "ReorderPrim is_empty = " << this->is_empty << "\n";
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

  bool operator == (const IndirectFunc& A) const {
    auto dstSumPrim = A.getSumPrim();
    auto dstEnumPrim = A.getEnumeratePrim();
    auto dstSchedPrim = A.getSchedulePrim();
    auto dstReorderPrim = A.getReorderPrim();
    if (!(this->sumPrim == dstSumPrim)) return 0;
    if (!(this->enumeratePrim == dstEnumPrim)) return 0;
    if (!(this->schedulePrim == dstSchedPrim)) return 0;
    if (!(this->reorderPrim == dstReorderPrim)) return 0;
    return 1;
  }

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

class LayoutPrim: public AffineMap {
public:

  LayoutPrim(): AffineMap(), packIndex({}), partitionIndex({}) {}

  explicit LayoutPrim(const std::vector<int>& _packIndex, const std::vector<int>& _partitionIndex): 
    AffineMap(), packIndex(_packIndex), partitionIndex(_partitionIndex) {}
  
  bool operator == (const LayoutPrim& A) const {
    auto dstPack = A.getPackIndex();
    auto dstPartition = A.getPartitionIndex();
    if (packIndex.size() != dstPack.size()) return 0;
    if (partitionIndex.size() != dstPartition.size()) return 0;
    for (size_t i = 0; i < packIndex.size(); ++i) {
      if (packIndex[i] != dstPack[i]) return 0;
    }
    for (size_t i = 0; i < partitionIndex.size(); ++i) {
      if (partitionIndex[i] != dstPartition[i]) return 0;
    }
    return 1;
  }

  std::vector<int> getPackIndex() const { return packIndex; }
  std::vector<int> getPartitionIndex() const { return partitionIndex; }

  void Print() {
    std::cout << "PackLevel: ";
    for (auto ele: packIndex) {
      std::cout << ele << ' ';
    }
    std::cout << std::endl;
    std::cout << "PartitionLevel: ";
    for (auto ele: partitionIndex) {
      std::cout << ele << ' ';
    }
    std::cout << std::endl;
  }

private:
  std::vector<int> packIndex;
  std::vector<int> partitionIndex;
};

} //endof unisparse
} //endof mlir

#endif