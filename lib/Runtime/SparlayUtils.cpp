//===- SparlayUtils.cpp - Sparlay runtime lib -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a light-weight runtime support library that is useful
// for COO format sparse tensor read in. 
//
//===----------------------------------------------------------------------===//
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_PARALLELIZE
#include "mlir/ExecutionEngine/CRunnerUtils.h"

// #ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
// #define DEBUG

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <chrono>
#include "Eigen/Dense"
#define DataType float

using namespace mlir;
extern "C" {

enum class SparlayDimLevelType : uint8_t {
  kDense = 0,
  kCompressed = 1,
  kSingleton = 2
};
}

template <typename indexTp, typename valueTp>
class SparseCoordinate {
public:

  SparseCoordinate(uint64_t rank) {
      indices.reserve(rank);
      for (unsigned i = 0; i < rank; i++) {
          std::vector<indexTp> tmp;
          indices.push_back(tmp);
      }
  }

  ~SparseCoordinate() {}

  void insert(const std::vector<indexTp> &indices_read, const valueTp value_read) {
      // printf("indices_read size = %zu, getRank = %lu\n", indices_read.size() , getRank());
      // assert(getRank() == indices_read.size());
      for (unsigned i = 0; i < indices_read.size(); i++) {
          indices[i].push_back(indices_read[i]);
      }
      values.push_back(value_read);
  }

  uint64_t getRank() {
      return indices.size();
  }

  void getIndex(std::vector<indexTp> **output, uint64_t dim) {
      assert(dim < getRank());
      *output = &indices[dim];
  }

  void getValue(std::vector<valueTp> **output) {
      *output = &values;
  }

  void print() {
      printf("SparseCoordinate: \n");
      assert(indices.size() == values.size());
      for (unsigned i = 0; i < indices.size(); i++) {
          for (unsigned j = 0; j < indices[i].size(); j++) {
              printf("%d  ", indices[i][j]);
          }
          printf("%f  \n", values[i]);
      }
  }

private:
    std::vector<std::vector<indexTp>> indices;
    std::vector<valueTp> values;
};

static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

class SparlayStruct {
public:
  std::vector<void*> vec;
  SparlayStruct() { vec.clear(); }
  SparlayStruct(std::vector<void*>& _vec) { vec = _vec; }
  void* get(size_t index) { return vec[index]; }
};

class SparlayWindow {
public:
  int M[2][2];
  int T[2][2];
  SparlayWindow() { memset(M, 0, sizeof(M)); memset(T, 0, sizeof(T)); }
  void assign(int i, int j, int v) { M[i][j] = v; }
  void tile(int i, bool type, int size) { T[i][type] = size; }
};

// template<typename valueTp>
// class SparseComputeOutput {

// public:
//     SparseComputeOutput(std::vector<uint64_t> sizes) {
//         uint64_t total_size = 1;
//         for (uint64_t i = 0; i < sizes.size(); i++) {
//             total_size *= sizes[i];
//         }
//         output = new valueTp[total_size];
//     }

//     ~SparseComputeOutput() {}

//     valueTp *output;
// }

#define TI (double)clock()/CLOCKS_PER_SEC

static void readMTXHeader(FILE* file, char* fileName, uint64_t* metaData, char* field, char* symmetry) {
    char line[1025];
    char header[64];
    char object[64];
    char format[64];
    
    // Read header line.
    printf("read MTX filename %s\n", fileName);                                                       
    if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
                symmetry) != 5) {
        fprintf(stderr, "Corrupt header in %s\n", fileName);
        exit(1);
    }
    // Make sure this is a general sparse matrix.
    if (strcmp(toLower(header), "%%matrixmarket") ||
        strcmp(toLower(object), "matrix") ||
        strcmp(toLower(format), "coordinate")) {
        fprintf(stderr,
                "Cannot find a coordinate sparse matrix in %s\n", fileName);
        exit(1);
    }
    // Skip comments.
    while (1) {
        if (!fgets(line, 1025, file)) {
        fprintf(stderr, "Cannot find data in %s\n", fileName);
        exit(1);
        }
        if (line[0] != '%')
        break;
    }
    // Next line contains M N NNZ.
    metaData[0] = 2; // rank
    if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", metaData + 2, metaData + 3,
                metaData + 1) != 3) {
        fprintf(stderr, "Cannot find size in %s\n", fileName);
        exit(1);
    }
}

static void readFROSTTHeader(FILE* file, char* fileName, uint64_t* metaData) {

}

typedef Eigen::Matrix<int, 2, 1> Vector2i;

typedef long long u64;

namespace Perf {
  static std::chrono::steady_clock::time_point _tic;
  void tic() { _tic = std::chrono::steady_clock::now(); }
  void toc() {
    std::chrono::nanoseconds diff = std::chrono::steady_clock::now()-_tic;
    std::cerr << "Time = " << (double)diff.count()/1000000 << "(ms)" << std::endl; 
  }
}

class LevelStorage {
public:
  int type; //0: none, 1: trimmed, 2: fused, 4: info
  int size; //dense iteration bound
  std::vector<int> crd;
  std::vector<int> ptr;
  std::vector<bool> same_path;
  LevelStorage() {
    type = 0;
    size = 0;
    crd.clear(), ptr.clear(), same_path.clear();
  }
  LevelStorage(
    int _type, int _size, std::vector<int>& _crd, std::vector<int>& _ptr, std::vector<bool>& _same_path
  ) {
    type = _type, size = _size;
    crd = _crd;
    ptr = _ptr;
    same_path = _same_path;
  }
  bool operator == (const LevelStorage& A) {
    if (type != A.type || size != A.size) return 0;
    if (crd.size() != A.crd.size() || ptr.size() != A.ptr.size() || same_path.size() != A.same_path.size()) return 0;
    for (size_t i = 0; i < crd.size(); ++i) if (crd[i] != A.crd[i]) return 0;
    for (size_t i = 0; i < ptr.size(); ++i) if (ptr[i] != A.ptr[i]) return 0;
    for (size_t i = 0; i < same_path.size(); ++i) if (same_path[i] != A.same_path[i]) return 0;
    return 1;
  }
  
};

/*!
 * \brief assume that Level 0 is root
 */
class SparlayStorage {
public:

  std::vector<uint64_t> dimSizes;
  std::vector< std::shared_ptr<LevelStorage> > vLevel;
  std::vector< std::shared_ptr<Vector2i> > exprs;
  std::vector<float> valueArray;
  std::vector< std::vector<float> > vectorArray;
  std::vector<float> vector_1d;
  int singleVectorSize;

  #define LVINFO 4
  #define LVTRIM 1
  #define LVFUSE 2

  SparlayStorage() {singleVectorSize=0;}
  SparlayStorage(std::vector<uint64_t> &dimSizes, uint64_t *perm, const SparlayDimLevelType *sparsity)
        :dimSizes(dimSizes), rev(getRank()), dimTypes(sparsity, sparsity + getRank()), idx(getRank())  {
    assert(perm && sparsity);
    const uint64_t rank = getRank();
    // Validate parameters.
    assert(rank > 0 && "Trivial shape is unsupported");
    for (uint64_t r = 0; r < rank; r++) {
      assert(dimSizes[r] > 0 && "Dimension size zero has trivial storage");
      assert((dimTypes[r] == SparlayDimLevelType::kDense ||
              dimTypes[r] == SparlayDimLevelType::kCompressed) &&
             "Unsupported DimLevelType");
    }
    // Construct the "reverse" (i.e., inverse) permutation.
    for (uint64_t r = 0; r < rank; r++)
      rev[perm[r]] = r;
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    vLevel[0]->type = LVFUSE | LVINFO;
    vLevel[0]->size = 1;
    vLevel[0]->ptr.push_back(0);
    for(uint64_t i = 1; i <= rank; i++) {
      vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
      vLevel[1]->size = dimSizes[rev[i-1]];
    }  
  }

  void initCOO(int sizeI, int sizeJ) {
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    vLevel[0]->type = LVFUSE | LVINFO;
    vLevel[0]->size = 1;
    vLevel[0]->ptr.push_back(0);
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    vLevel[1]->size = sizeI;
    vLevel[1]->type = LVTRIM;
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    vLevel[2]->size = sizeJ;
    vLevel[2]->type = LVTRIM;
    valueArray.clear();
    vectorArray.clear();
//    dimSizes.push_back(0);
    dimSizes.push_back(sizeI), dimSizes.push_back(sizeJ);
    exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,0)));
    exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(1,0)));
    exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,1)));
  }

  void finalizeCOO() {
    vLevel[0]->ptr.push_back(vLevel[1]->crd.size());
    assert(vLevel[1]->crd.size() == vLevel[2]->crd.size());
    assert(vLevel[1]->crd.size() == valueArray.size());
  }

  void dfsLowerPtr(int cur_lv, int id, int pos, int target_lv, std::vector<int>& ret);
  std::vector<int> lowerPtr(int st_lv, int ed_lv);

  bool trim(const int level);
  bool fuse(const int level);
  bool grow(const int level);
  bool separate(const int level);
  bool swap(const int LU, const int LD);
  bool add(const int Ltarget, const int Lsrc);
  bool sub(const int Ltarget, const int Lsrc);
  bool vectorize(const int lv);
  bool devectorize();
  bool neg(const int level);
  bool moveLv(const int srcLevel, const int targetLevel);
  bool tile_merge(const int lv, const int factor);
  bool tile_split(const int lv, const int factor);

  void getSize(size_t lv) {
    assert(lv < exprs.size());
    // assert(dimSizes.size() == 3);
    
    Vector2i t0(0,0), t1(0,dimSizes[1]), t2(dimSizes[0],0), t3(dimSizes[0],dimSizes[1]);
    const auto& expr = exprs[lv];
    int mn = std::min(0, std::min(expr->dot(t1), std::min(expr->dot(t2), expr->dot(t3))));
    int mx = std::max(0, std::max(expr->dot(t1), std::max(expr->dot(t2), expr->dot(t3))));
    vLevel[lv]->size = (mx - mn);
  }

  bool isCompressedDim(uint64_t d) const {
    assert(d < getRank());
    return (dimTypes[d] == SparlayDimLevelType::kCompressed);
  }

  void swapStorage(int srcLv, int targetLv) {
    std::swap(vLevel[srcLv], vLevel[targetLv]);
    std::swap(exprs[srcLv], exprs[targetLv]);
  }

  void moveStorage(int srcLv, int dstLv) {
    assert(dstLv <= srcLv);
    auto svLS = vLevel[srcLv];
    for (int i = srcLv; i >= dstLv+1; --i) {
      vLevel[i] = vLevel[i-1];
    }
    vLevel[dstLv] = svLS;
  }

  void newStorage(int lv, std::shared_ptr<LevelStorage> LS, std::shared_ptr<Vector2i> expr) {
    vLevel.insert(vLevel.begin()+lv, LS);
    exprs.insert(exprs.begin()+lv, expr);
  }

  void removeStorage(size_t lv) {
    assert(lv < vLevel.size());
    vLevel.erase(vLevel.begin()+lv);
    exprs.erase(exprs.begin()+lv);
  }

  void applyPerm(std::vector<int>& perm) {
    if (valueArray.size()) {
      static std::vector<float> new_val_array;
      assert(perm.size() == valueArray.size());
      new_val_array.clear();
      new_val_array.resize(perm.size(),0);
      for (size_t i = 0; i < valueArray.size(); ++i) {
        new_val_array[i] = std::move(valueArray[perm[i]]);
      }
      valueArray = std::move(new_val_array);
    } else if (vectorArray.size()) {
      static std::vector< std::vector<float> > new_vector_array;
      assert(perm.size() == vectorArray.size());
      new_vector_array.clear();
      new_vector_array.resize(perm.size(), {});
      for (size_t i = 0; i < vectorArray.size(); ++i) {
        new_vector_array[i] = std::move(vectorArray[perm[i]]);
      }
      vectorArray = std::move(new_vector_array);
    } else {
      std::cerr << "where is the value?" << std::endl;
      assert(0);
    }
  }

  void clearVector() {
    vectorArray.clear();
    singleVectorSize = 0;
  }

  void Print(std::ostream& fout, bool verbose=0);
  SparlayStorage copy() {
    SparlayStorage ret;
    for (size_t i = 0; i < vLevel.size(); ++i) {
      ret.vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage(*vLevel[i])));
      ret.exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(*exprs[i])));
    }
    if (valueArray.size()) {
      for (size_t i = 0; i < valueArray.size(); ++i) {
        ret.valueArray.push_back(valueArray[i]);
      }
    }
    ret.dimSizes = dimSizes;
    if (vectorArray.size()) { 
      assert(0);
    }
    return ret;
  }
  bool operator == (const SparlayStorage& A) {
    if (A.vLevel.size() != vLevel.size()) return 0;
    if (valueArray.size() != A.valueArray.size()) return 0;
    if (vectorArray.size() != A.vectorArray.size()) return 0;
    for (size_t i = 0; i < vLevel.size(); ++i) if (!((*vLevel[i]) == (*A.vLevel[i]))) return 0;
    for (size_t i = 0; i < valueArray.size(); ++i) {
      if (valueArray[i] != A.valueArray[i]) return 0;
    }
    for (size_t i = 0; i < vectorArray.size(); ++i) {
      if (vectorArray[i].size() != A.vectorArray[i].size()) return 0;
      for (size_t j = 0; j < vectorArray[i].size(); ++j) {
        if (vectorArray[i][j] != A.vectorArray[i][j]) return 0;
      }
    }
    return 1;
  }

  uint64_t getRank() const { return dimSizes.size(); }

  uint64_t getDimSize(uint64_t d) const {
    assert(d < getRank());
    return dimSizes[d];
  }
  
  void lexInsert(const uint64_t *cursor, float val) {
    // First, wrap up pending insertion path.
//    std::cerr << "Enter lexInsert " << std::endl;
    uint64_t diff = 0;
    uint64_t top = 0;
    if (!valueArray.empty()) {
      diff = lexDiff(cursor);
      endPath(diff + 1);
      top = idx[diff] + 1;
    }
    // Then continue with insertion path.
    insPath(cursor, diff, top, val);
  }

  void expInsert(uint64_t *cursor, float *values, bool *filled, uint64_t *added,
                 uint64_t count) {
    if (count == 0)
      return;
    // Sort.
    std::sort(added, added + count);
    // Restore insertion path for first insert.
    const uint64_t lastDim = getRank() - 1;
    uint64_t index = added[0];
    cursor[lastDim] = index;
    lexInsert(cursor, values[index]);
    assert(filled[index]);
    values[index] = 0;
    filled[index] = false;
    // Subsequent insertions are quick.
    for (uint64_t i = 1; i < count; i++) {
      assert(index < added[i] && "non-lexicographic insertion");
      index = added[i];
      cursor[lastDim] = index;
      insPath(cursor, lastDim, added[i - 1] + 1, values[index]);
      assert(filled[index]);
      values[index] = 0;
      filled[index] = false;
    }
  }

  void endInsert() {
    if (valueArray.empty())
      finalizeSegment(0);
    else
      endPath(0);
  }

private:

  void appendPointer(uint64_t d, uint64_t pos, uint64_t count = 1) {
    assert(isCompressedDim(d));
    assert(pos <= std::numeric_limits<int32_t>::max() &&
           "Pointer value is too large for the P-type");
    vLevel[d]->ptr.insert(vLevel[d]->ptr.end(), count, static_cast<int32_t>(pos));
  }

  void finalizeSegment(uint64_t d, uint64_t full = 0, uint64_t count = 1) {
    if (count == 0)
      return; // Short-circuit, since it'll be a nop.
    if (isCompressedDim(d)) {
      appendPointer(d, vLevel[d+1]->crd.size(), count);
    } else { // Dense dimension.
      const uint64_t sz = getDimSize(d);
      assert(sz >= full && "Segment is overfull");
      assert((count == 0 || (sz - full) <= std::numeric_limits<uint64_t>::max() / count) &&
         "Integer overflow");
      count = count * (sz - full);
      if (d + 1 == getRank())
        valueArray.insert(valueArray.end(), count, 0);
      else
        finalizeSegment(d + 1, 0, count);
    }
  }

  void appendIndex(uint64_t d, uint64_t full, uint64_t i) {
    if (isCompressedDim(d)) {
      assert(i <= std::numeric_limits<int32_t>::max() && "Index value is too large for the I-type");
      vLevel[d+1]->crd.push_back(static_cast<int32_t>(i));
    } else { // Dense dimension.
      assert(i >= full && "Index was already filled");
      if (i == full)
        return; // Short-circuit, since it'll be a nop.
      if (d + 1 == getRank())
        valueArray.insert(valueArray.end(), i - full, 0);
      else
        finalizeSegment(d + 1, 0, i - full);
    }
  }
  
  void insPath(const uint64_t *cursor, uint64_t diff, uint64_t top, float val) {
    uint64_t rank = getRank();
    assert(diff < rank);
    for (uint64_t d = diff; d < rank; d++) {
      uint64_t i = cursor[d];
      appendIndex(d, top, i);
      top = 0;
      idx[d] = i;
    }
    valueArray.push_back(val);
  }

  uint64_t lexDiff(const uint64_t *cursor) const {
//    std::cerr << "Enter lexDiff " << std::endl;
    for (uint64_t r = 0, rank = getRank(); r < rank; r++)
      if (cursor[r] > idx[r])
        return r;
      else
        assert(cursor[r] == idx[r] && "non-lexicographic insertion");
    assert(0 && "duplication insertion");
    return -1u;
  }

  void endPath(uint64_t diff) {
    uint64_t rank = getRank();
    assert(diff <= rank);
    for (uint64_t i = 0; i < rank - diff; i++) {
      const uint64_t d = rank - i - 1;
      finalizeSegment(d, idx[d] + 1);
    }
  }

  std::vector<uint64_t> rev;
  std::vector<SparlayDimLevelType> dimTypes;
  std::vector<uint64_t> idx;
};

SparlayStorage* readFromFile(std::istream& fin);

void SparlayStorage::Print(std::ostream& fout, bool verbose) {
  fout << "==============================================" << std::endl;
  for (size_t i = 0; i < vLevel.size(); ++i) {
    fout << "crd: ";
    for (const auto ele: vLevel[i]->crd) {
      fout << std::setw(8) << ele;
    }
    fout << "      (Type:" << vLevel[i]->type << ")";
    fout << " [Size:" << vLevel[i]->size << "]";
    fout << std::endl;
    if (verbose && vLevel[i]->same_path.size()) {
      assert(vLevel[i]->same_path.size() == vLevel[i]->crd.size());
      fout << "smp: ";
      for (const auto ele: vLevel[i]->same_path) {
        fout << std::setw(8) << ele;
      }
      fout << std::endl;
    }
    if (vLevel[i]->ptr.size()) {
      fout << "ptr: ";
      for (const auto ele: vLevel[i]->ptr) {
        fout << std::setw(8) << ele;
      }
      fout << std::endl;
    }
  }
  if (valueArray.size()) {
    fout << "val: ";
    for (size_t i = 0; i < valueArray.size(); ++i) {
      fout << std::setw(8) << valueArray[i];
    }
    fout << std::endl;
  } else if (vectorArray.size()) {
    size_t Size = vectorArray[0].size();
    for (size_t j = 0; j < Size; ++j) {
      fout << "val: ";
      for (size_t i = 0; i < vectorArray.size(); ++i) {
        assert(vectorArray[i].size() == Size);
        fout << std::setw(8) << vectorArray[i][j];
      }
      fout << std::endl;
    }
  }
  fout << "==============================================" << std::endl;
}

//Read a COO file
SparlayStorage* readFromFile(std::istream& fin) {
  std::string tmp;
  while (1) {
    getline(fin, tmp);
    if (tmp[0] != '%') break;
  }
  std::stringstream SS(tmp);
  int H, W, N_ele;
  SS >> H >> W >> N_ele;

  std::cerr << "Size of the matrix: " << H << ' ' << W << ' ' << N_ele << std::endl;

  LevelStorage* rowStore = new LevelStorage();
  LevelStorage* colStore = new LevelStorage();
  LevelStorage* rootStore = new LevelStorage();
  static std::vector<float> valStore;
  valStore.clear();

  //Must be row major, otherwise the fuse operation will be incorrect
  //trim(0), fuse(None), (d0, ... dn)

  rowStore->type = colStore->type = 1;
  rowStore->size = H, colStore->size = W;
  rowStore->crd.reserve(N_ele), colStore->crd.reserve(N_ele);
  rowStore->same_path.push_back(0), colStore->same_path.push_back(0);

  rootStore->size = 1;
  rootStore->type = LVFUSE | LVINFO;
  rootStore->ptr.push_back(0);

  for (int row, col, i = 0; i < N_ele; ++i) {
    float v;
    fin >> row >> col >> v;
    --row, --col;
    // std::cerr << row << ' ' << col << ' ' << v << std::endl;
    rowStore->crd.push_back(row);
    colStore->crd.push_back(col);
    valStore.push_back(v);
  }

  rootStore->ptr.push_back(rowStore->crd.size());

  auto ret = new SparlayStorage();
  ret->vLevel.push_back(std::shared_ptr<LevelStorage>(rootStore));

  std::vector< std::vector<int> > bucket;
  std::vector<int> pos, oriID;
  bucket.resize(H, {});
  pos.resize(rowStore->crd.size(), 0);
  oriID.resize(pos.size(), 0);
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    
    assert((size_t)rowStore->crd[i] < bucket.size());
    bucket[rowStore->crd[i]].push_back(i);
    pos[i] = i;
    oriID[i] = i;
  }
  int ptr = 0;
  for (size_t i = 0; i < bucket.size(); ++i) {
    for (size_t j = 0; j < bucket[i].size(); ++j) {
      int cur_pos = pos[bucket[i][j]];
      assert(cur_pos >= ptr);
      if (cur_pos != ptr) {
        std::swap(rowStore->crd[ptr], rowStore->crd[cur_pos]);
        std::swap(colStore->crd[ptr], colStore->crd[cur_pos]);
        std::swap(valStore[ptr], valStore[cur_pos]);
        pos[oriID[ptr]] = cur_pos;
        oriID[cur_pos] = oriID[ptr];
      }
      ptr++;
#ifdef DEBUG
      for (size_t k = 0; k < pos.size(); ++k) std::cerr << pos[k] << ' ';
      std::cerr << std::endl;
#endif
    }
  }

  for (size_t i = 1; i < rowStore->crd.size(); ++i) {
    rowStore->same_path.push_back(rowStore->crd[i] == rowStore->crd[i-1]);
    colStore->same_path.push_back(
      (rowStore->crd[i] == rowStore->crd[i-1]) && (colStore->crd[i] == colStore->crd[i-1])
    );
  }

  ret->vLevel.push_back(std::shared_ptr<LevelStorage>(rowStore));
  ret->vLevel.push_back(std::shared_ptr<LevelStorage>(colStore));
//  ret->dimSizes.push_back(0);
  ret->dimSizes.push_back(H), ret->dimSizes.push_back(W);
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,0)));
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(1,0)));
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,1)));
  ret->valueArray = valStore;
  return ret;
}

bool SparlayStorage::moveLv(const int srcLv, const int dstLv) {
  assert(dstLv <= srcLv);
  assert(dstLv >= 1);
  for (int curLv = dstLv; curLv <= srcLv; ++curLv) {
    assert(!(vLevel[curLv]->type & LVFUSE));
    assert(vLevel[curLv]->type & LVTRIM);
  }
  int upperLv = dstLv-1;
  static std::vector<int> count;
  static std::vector<int> bucket;
  static std::vector<int> pos;
  static std::vector<int> perm;

  count.clear(), bucket.clear(), pos.clear(), perm.clear();
  if (upperLv != 0) {
    bucket.resize(vLevel[srcLv]->crd.size(), 0);
    pos.resize(vLevel[srcLv]->crd.size(), 0);
  }
  count.resize(vLevel[srcLv]->size+1, 0);
  perm.resize(vLevel[srcLv]->crd.size(), 0);
  int cur_bucket = -1;
  int min_src_crd = 2147483633;
  for (size_t i = 0; i < vLevel[srcLv]->crd.size(); ++i) {
    min_src_crd = std::min(min_src_crd, vLevel[srcLv]->crd[i]);
  }
  for (size_t i = 0; i < vLevel[srcLv]->crd.size(); ++i) {
    count[vLevel[srcLv]->crd[i] + 1 - min_src_crd]++;
    if (upperLv != 0) {
      if (!vLevel[upperLv]->same_path[i]) {
        bucket[i] = (++cur_bucket);
        pos[cur_bucket] = i;
      } else {
        bucket[i] = cur_bucket;
      }
    }
  }
  for (int i = 1; i < vLevel[srcLv]->size; ++i) {
    count[i] += count[i-1];
  }
  for (size_t i = 0; i < vLevel[srcLv]->crd.size(); ++i) {
    int cur_crd = vLevel[srcLv]->crd[i] - min_src_crd;
    perm[count[cur_crd]++] = i;
  }
  if (upperLv != 0) {
    static std::vector<int> new_perm;
    new_perm.clear();
    new_perm.resize(vLevel[srcLv]->crd.size(), 0);
    for (size_t i = 0; i < vLevel[srcLv]->crd.size(); ++i) {
      new_perm[pos[bucket[perm[i]]]++] = std::move(perm[i]);
    }
    perm = std::move(new_perm);
  }
  this->moveStorage(srcLv, dstLv);

  static std::vector< std::vector<int> > new_crd;
  static std::vector< std::vector<bool> > new_same_path;
  new_crd.resize(vLevel.size()-dstLv);
  new_same_path.resize(vLevel.size()-dstLv);
  for (size_t i = dstLv; i < vLevel.size(); ++i) {
    new_crd[i-dstLv].resize(vLevel[dstLv]->crd.size(),0);
    new_same_path[i-dstLv].resize(vLevel[dstLv]->crd.size(),0);
  }
  int stLv = dstLv;
  if (!perm.size()) return 1;
  if (stLv == 1) {
    new_crd[0][0] = vLevel[1]->crd[perm[0]];
    for (size_t j = 1; j < vLevel[1]->crd.size(); ++j) {
      new_crd[0][j] = std::move(vLevel[1]->crd[perm[j]]);
      new_same_path[0][j] = (new_crd[0][j] == new_crd[0][j-1]);
    }
    vLevel[1]->crd = std::move(new_crd[0]);
    vLevel[1]->same_path = std::move(new_same_path[0]);
    stLv++;
  }
  for (size_t i = stLv; i < vLevel.size(); ++i) {
    int curLv = i-dstLv;
    new_crd[curLv][0] = std::move(vLevel[i]->crd[perm[0]]);
  }
  for (size_t i = stLv; i < vLevel.size(); ++i) {
    int curLv = i-dstLv;
    for (size_t j = 1; j < vLevel[dstLv]->crd.size(); ++j) {  
      new_crd[curLv][j] = std::move(vLevel[i]->crd[perm[j]]);
      new_same_path[curLv][j] = ((new_crd[curLv][j] == new_crd[curLv][j-1]) && (vLevel[i-1]->same_path[j]));
    }
    vLevel[i]->crd = std::move(new_crd[curLv]);
    vLevel[i]->same_path = std::move(new_same_path[curLv]);
  }
  this->applyPerm(perm);
  return 1;
}

bool SparlayStorage::tile_split(int lv, int factor) {
  assert(!(vLevel[lv]->type & LVFUSE));
  assert(vLevel[lv]->type & LVTRIM);
  std::vector<int> new_crd;
  std::vector<bool> new_same_path;
  new_crd.resize(vLevel[lv]->crd.size(),0);
  new_same_path.resize(vLevel[lv]->crd.size(),0);
  for (size_t i = 0; i < new_crd.size(); ++i) {
    new_crd[i] = vLevel[lv]->crd[i]%factor;
    vLevel[lv]->crd[i] /= factor;
    if (i == 0) {
      new_same_path[i] = 0;
      vLevel[lv]->same_path[i] = 0;
    }
    else {
      new_same_path[i] = (new_crd[i] == new_crd[i-1]);
      new_same_path[i] = new_same_path[i] && (vLevel[lv]->same_path[i]);
      vLevel[lv]->same_path[i] = (vLevel[lv]->crd[i] == vLevel[lv]->crd[i-1]);
      if (lv != 1) (vLevel[lv]->same_path[i]) = (vLevel[lv]->same_path[i]) && (vLevel[lv-1]->same_path[i]);
    }
  }
  vLevel[lv]->size = ceil((float)vLevel[lv]->size/factor);
  std::vector<int> empty_ptr = {};
  this->newStorage(
    lv+1, 
    std::shared_ptr<LevelStorage>(new LevelStorage(LVTRIM, factor, new_crd, empty_ptr, new_same_path)), 
    std::make_shared<Vector2i>(0,0)
  );
  return 1;
}

bool SparlayStorage::tile_merge(int lv, int factor) {
  assert(!(vLevel[lv]->type & LVFUSE));
  assert(vLevel[lv]->type & LVTRIM);
  assert(vLevel[lv+1]->type & LVTRIM);
  assert(!(vLevel[lv+1]->type&LVFUSE));
  assert(vLevel[lv+1]->size == factor);
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    (vLevel[lv]->crd[i] *= factor) += vLevel[lv+1]->crd[i];
    if (i != 0) {
      vLevel[lv]->same_path[i] = (vLevel[lv]->same_path[i] && (vLevel[lv]->crd[i]==vLevel[lv]->crd[i-1]));
    }
  }
  this->getSize(lv);
  this->removeStorage(lv+1);
  return 1;
}

/*!
 * \param pos current dense position
 */
void SparlayStorage::dfsLowerPtr(int cur_lv, int id, int pos, int target_lv, std::vector<int>& ret) {
  if (cur_lv == target_lv) {
    assert(ret.size() > (size_t)pos);
    assert(ret[pos+1] <= id);
    if (vLevel[cur_lv]->ptr.size()) {
      assert((size_t)id < vLevel[cur_lv]->ptr.size());
      ret[pos+1] = vLevel[cur_lv]->ptr[id+1];
    } else {
      ret[pos+1] = id+1;
    }
    return;
  }
  int nxtLevelSize = vLevel[cur_lv+1]->size;
  if (vLevel[cur_lv]->ptr.size()) {
    int idL = vLevel[cur_lv]->ptr[id], idR = vLevel[cur_lv]->ptr[id+1];
    assert(vLevel[cur_lv+1]->crd.size() >= (size_t)idR);
    //TODO: optimizable when current level is not fused
    for (int to = idL; to < idR; ++to) {
      int to_pos = pos * nxtLevelSize + vLevel[cur_lv+1]->crd[to];
      dfsLowerPtr(cur_lv+1, to, to_pos, target_lv, ret);
    }
  } else {
    assert(vLevel[cur_lv+1]->crd.size() > (size_t)id);
    dfsLowerPtr(cur_lv+1, id, pos * nxtLevelSize + vLevel[cur_lv+1]->crd[id], target_lv, ret);
  }
}

std::vector<int> SparlayStorage::lowerPtr(int st_lv, int ed_lv) {
  std::vector<int> ret;
  u64 ed_level_size = 1;
  for (int i = 0; i <= ed_lv; ++i) {
    ed_level_size *= vLevel[i]->size;
    if (ed_level_size > 1e7) {
      std::cerr << "The untrimmed level is too huge to be stored, or inefficient conversion sequence" << std::endl;
      assert(0);
    }
  }
  assert(!(vLevel[st_lv]->type&1));
  assert(vLevel[st_lv]->size);
  ret.resize(ed_level_size+1, -1);
  for (size_t i = 0; i < vLevel[st_lv]->ptr.size()-1; ++i) {
    if (vLevel[st_lv]->ptr[i] < vLevel[st_lv]->ptr[i+1])
      dfsLowerPtr(st_lv, i, i, ed_lv, ret);
  }
  ret[0] = 0;
  for (size_t i = 1; i < ret.size(); ++i) {
    if (ret[i] == -1) ret[i] = ret[i-1];
  }
#ifdef DEBUG
  Print(std::cerr, 1);
  std::cerr << "ret: ";
  for (size_t i = 0; i < ret.size(); ++i) std::cerr << std::setw(8) << ret[i];
  std::cerr << std::endl;
#endif
  return ret;
}

bool SparlayStorage::grow(const int lv) {
  if (!(vLevel[lv]->type & 1)) { //already not trimmed
    return 1;
  }
  if (lv == (int)(vLevel.size()-1)) {
    std::cerr << "Error: Attempt to **Grow** the last level" << std::endl;
    assert(0);
  }
  int st_lv = 0;
  for (size_t i = 1; i < vLevel.size(); ++i) {
    if (vLevel[i]->type & 1) { st_lv = i; break; }
  }
  st_lv--;
  assert(st_lv != -1);
  assert(st_lv <= lv);
  auto new_ptr = lowerPtr(st_lv, lv);
  for (int i = st_lv; i <= lv; ++i) {
    vLevel[i]->crd.clear();
    vLevel[i]->same_path.clear();
    if (i != lv) vLevel[i]->ptr.clear();
    if (vLevel[i]->type & LVTRIM) {
      assert(i != st_lv);
      vLevel[i]->type ^= 1;
    }
  }
  assert(vLevel[st_lv]->type & LVINFO);
  vLevel[st_lv]->type ^= LVINFO;
  assert((vLevel[lv]->type & LVINFO) == 0);
  vLevel[lv]->ptr = new_ptr;
  vLevel[lv]->type ^= LVINFO;
  return 1;
}

bool SparlayStorage::trim(const int lv) {
  if (vLevel[lv]->type & 1) return 1; //already trimmed
  int lower_lv = vLevel.size()-1;
  while (lower_lv != 0 && (vLevel[lower_lv-1]->type & LVTRIM)) lower_lv--;
  assert(lower_lv > lv);
  auto cur_ptr = vLevel[lower_lv-1]->ptr;
  vLevel[lower_lv-1]->ptr.clear();
  assert(vLevel[lower_lv-1]->type & LVINFO);
  vLevel[lower_lv-1]->type ^= LVINFO;
  /*
   * Task: 
   *  1) Gen crd, ptr(is fused), same_path for Level[lv, lower_lv-1]
   *  2) Change type for Level[lv, lower_lv-1]
   */
  for (int cur_lv = lower_lv - 1; cur_lv >= lv; --cur_lv) {
    assert(vLevel[cur_lv]->crd.size() == 0);
    assert(vLevel[cur_lv]->ptr.size() == 0);
    assert(vLevel[cur_lv]->same_path.size() == 0);
    assert(cur_lv > 0);
    // tagB: Build crd, same_path, ptr(if fused)
    if (vLevel[cur_lv]->type & LVFUSE) vLevel[cur_lv]->ptr.push_back(0);
    int cur_lv_size = vLevel[cur_lv]->size;
    if (vLevel[cur_lv]->type & 2) {
      for (size_t i = 0; i < cur_ptr.size()-1; ++i) {
        if (cur_ptr[i] != cur_ptr[i+1]) {
          vLevel[cur_lv]->crd.push_back(i % cur_lv_size);
          vLevel[cur_lv]->ptr.push_back(cur_ptr[i+1]);
          vLevel[cur_lv]->same_path.push_back(0);
        }
      }
    } else {
      for (size_t i = 0; i < cur_ptr.size()-1; ++i) {
        for (auto j = cur_ptr[i]; j < cur_ptr[i+1]; ++j) {
          vLevel[cur_lv]->crd.push_back(i);
          vLevel[cur_lv]->same_path.push_back((j!=cur_ptr[i]));
        }
      }
    }
    assert(!(vLevel[cur_lv]->type & LVTRIM));
    vLevel[cur_lv]->type ^= LVTRIM;
    // End of tagB
    //TagC: update cur_ptr
    std::vector<int> new_cur_ptr;
    new_cur_ptr.reserve((cur_ptr.size()-1)/cur_lv_size + 1);
    assert((cur_ptr.size()-1) % cur_lv_size == 0);
    new_cur_ptr.push_back(0);
    for (size_t i = 0; i < cur_ptr.size()-1; i += cur_lv_size) {
      if (vLevel[cur_lv]->type & LVFUSE) {
        int cnt = 0;
        for (int j = 0; j < cur_lv_size; ++j) {
          cnt += (cur_ptr[i+j] != cur_ptr[i+j+1]);
        }
        new_cur_ptr.push_back(cnt+ *(new_cur_ptr.end()-1));
      } else {
        new_cur_ptr.push_back(cur_ptr[i+cur_lv_size]);
      }
    }
    cur_ptr = std::move(new_cur_ptr);
  }
  assert(lv > 0);
  assert(vLevel[lv-1]->ptr.size() == 0);
  assert((vLevel[lv-1]->type & LVINFO) == 0);
  vLevel[lv-1]->ptr = cur_ptr;
  vLevel[lv-1]->type ^= LVINFO;
  return 1;
}

bool SparlayStorage::fuse(const int lv) {
  if (vLevel[lv]->type & 2) return 1;
  if (vLevel[lv]->type & 1) {
    int upper_lv = lv;
    while (upper_lv!=0 && (vLevel[upper_lv-1]->type&1) && !(vLevel[upper_lv-1]->type&2)) upper_lv--;
    //Note: crd_deleted == same_path
    auto crd_deleted = vLevel[lv]->same_path;
    bool no_work = 0;
    for (const auto ele: crd_deleted) no_work |= ele;
    if (!no_work) {
      //build a ptr so that the user is happy
      assert(vLevel[lv]->ptr.size() == 0);
      vLevel[lv]->ptr.reserve(vLevel[lv]->crd.size()+1);
      vLevel[lv]->ptr.push_back(0);
      for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
        vLevel[lv]->ptr.push_back(i+1);
      }
      vLevel[lv]->type |= 2;
      return 1;
    }
    //update possibly the ptr of a fused level
    if (upper_lv!=0) {
      assert(vLevel[upper_lv-1]->type&2); //it must be a fused level
      int cur_lv = upper_lv-1;
      if (!(vLevel[cur_lv]->type&LVINFO)) assert(vLevel[cur_lv]->ptr.size() == vLevel[cur_lv]->crd.size()+1);
      int saved_st_point = 0;
      assert(vLevel[cur_lv]->ptr[0] == 0);
      for (size_t i = 0; i < vLevel[cur_lv]->ptr.size()-1; ++i) {
        int cnt = vLevel[cur_lv]->ptr[i+1] - vLevel[cur_lv]->ptr[i];
        for (int j = saved_st_point; j < vLevel[cur_lv]->ptr[i+1]; ++j) {
          if (crd_deleted[j]) cnt--;
        }
        // assert(cnt>=0);
        saved_st_point = vLevel[cur_lv]->ptr[i+1];
        vLevel[cur_lv]->ptr[i+1] = vLevel[cur_lv]->ptr[i]+cnt;
      }
    }
    int crd_reserve_size = 0;
    for (size_t i = 0; i < crd_deleted.size(); ++i) {
      crd_reserve_size += (!crd_deleted[i]);
    }
    vLevel[lv]->ptr.resize(crd_reserve_size+1, 0);
    int cnt = 1;
    int prev_pushed = 0;
    for (auto cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      assert(vLevel[cur_lv]->crd.size() == vLevel[lv]->crd.size());
      if (cur_lv != lv) assert(vLevel[cur_lv]->ptr.size() == 0);
      assert(vLevel[cur_lv]->crd.size() == vLevel[cur_lv]->same_path.size());
      static std::vector<int> new_crd;
      static std::vector<bool> new_same_path;
      new_crd.clear();
      new_same_path.clear();
      new_crd.resize(crd_reserve_size, 0);
      new_same_path.resize(crd_reserve_size, 0);
      int pt = 0;
      bool is_same_path = 1; /** !!!!!!! **/
      for (size_t i = 0; i < vLevel[cur_lv]->crd.size(); ++i) {
        is_same_path &= vLevel[cur_lv]->same_path[i];
        if (!crd_deleted[i]) {
          new_crd[pt] = vLevel[cur_lv]->crd[i];
          new_same_path[pt] = is_same_path;
          if (cur_lv == lv && i != 0) {
            prev_pushed += cnt;
            vLevel[lv]->ptr[pt] = prev_pushed;
            cnt = 1;
          }
          pt++;
          is_same_path = 1;
        }
        else if (cur_lv == lv && i != 0) cnt++;
      }
      vLevel[cur_lv]->crd = std::move(new_crd);
      vLevel[cur_lv]->same_path = std::move(new_same_path);
    }
    vLevel[lv]->ptr[crd_reserve_size] = (prev_pushed + cnt);
    vLevel[lv]->type ^= LVFUSE;
  } else {
    vLevel[lv]->type ^= LVFUSE;
  }
  return 1;
}

bool SparlayStorage::separate(const int lv) {
  if (!(vLevel[lv]->type & LVFUSE)) return 1;
  if (vLevel[lv]->type & LVTRIM) {
    int upper_lv = lv;
    while (upper_lv!=0 && (vLevel[upper_lv-1]->type&1) && !(vLevel[upper_lv-1]->type&2)) upper_lv--;
    //update possibly the ptr of a fused level
    if (upper_lv != 0) {
      int cur_lv = upper_lv-1;
      for (size_t i = 0; i < vLevel[cur_lv]->ptr.size()-1; ++i) {
        int idR = vLevel[cur_lv]->ptr[i+1]-1;
        vLevel[cur_lv]->ptr[i+1] = vLevel[lv]->ptr[idR+1];
      }
    }
    for (int cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      static std::vector<int> new_crd;
      static std::vector<bool> new_same_path;
      new_crd.clear();
      new_same_path.clear();
      new_crd.resize(vLevel[lv]->ptr[vLevel[lv]->crd.size()], 0);
      new_same_path.resize(vLevel[lv]->ptr[vLevel[lv]->crd.size()], 0);
      int ptr = 0;
      for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
        new_same_path[ptr] = std::move(vLevel[cur_lv]->same_path[i]);
        new_crd[ptr] = vLevel[cur_lv]->crd[i];
        ptr++;
        for (auto j = vLevel[lv]->ptr[i]+1; j < vLevel[lv]->ptr[i+1]; ++j) {
          new_crd[ptr] = vLevel[cur_lv]->crd[i];
          new_same_path[ptr] = 1;
          ptr++;
        }
      }
      vLevel[cur_lv]->crd = std::move(new_crd);
      vLevel[cur_lv]->same_path = std::move(new_same_path);
    }
    vLevel[lv]->ptr.clear();
    vLevel[lv]->type ^= LVFUSE;
  } else {
    vLevel[lv]->type ^= LVFUSE;
  }
  return 1;
}

bool SparlayStorage::swap(const int LU, const int LD) {
  assert(LU < LD);
  for (int i = LU; i <= LD; ++i) {
    assert(!(vLevel[i]->type & LVFUSE));
    assert(vLevel[i]->type&LVTRIM);
  }
  this->swapStorage(LD, LU);
  return 1;
}

bool SparlayStorage::add(const int Ltarget, const int Lsrc) {
  assert(Lsrc != Ltarget);
  for (int i = Lsrc; i <= Ltarget; ++i) {
    assert(!(vLevel[i]->type & LVFUSE));
  }
  for (size_t i = 0; i < vLevel[Lsrc]->crd.size(); ++i) {
    vLevel[Ltarget]->crd[i] += vLevel[Lsrc]->crd[i];
  }
  (*exprs[Ltarget]) += (*exprs[Lsrc]);
  getSize(Ltarget);
  return 1;
}

bool SparlayStorage::sub(const int Ltarget, const int Lsrc) {
  assert(Lsrc != Ltarget);
  int ULv = (Lsrc < Ltarget ? Lsrc : Ltarget);
  int DLv = (ULv == Lsrc ? Ltarget : Lsrc);
  for (int i = ULv; i <= DLv; ++i) {
    assert(!(vLevel[i]->type & LVFUSE));
  }
  assert(vLevel[Lsrc]->crd.size() == vLevel[Ltarget]->crd.size());
  for (size_t i = 0; i < vLevel[Lsrc]->crd.size(); ++i) {
    vLevel[Ltarget]->crd[i] -= vLevel[Lsrc]->crd[i];
  }
  (*exprs[Ltarget]) -= (*exprs[Lsrc]);
  getSize(Ltarget);
  return 1;
}

bool SparlayStorage::vectorize(const int lv) {
  assert(lv == 2);
  assert((size_t)lv == vLevel.size()-1);
  bool mark = !(vLevel[lv]->type&LVFUSE);
  fuse(lv-1);
  auto& father_ptr = vLevel[lv-1]->ptr;
  auto& father_crd = vLevel[lv-1]->crd;
  int cur_lv_size = vLevel[lv]->size;
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vectorArray.size() == 0);
    vectorArray.resize(father_crd.size(),{});
    assert(valueArray.size() == vLevel[lv]->crd.size());
    static std::vector<float> V;
    for (size_t i = 0; i < father_crd.size(); ++i) {
      V.clear();
      V.resize(cur_lv_size, 0.0);
      assert(father_ptr[i+1] > prev_ptr);
      for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
        V[vLevel[lv]->crd[j]] = std::move(valueArray[j]);
      }
      vectorArray[i] = std::move(V);
      int add_one = (father_ptr[i+1]>prev_ptr);
      int new_ptr = father_ptr[i] + add_one;
      prev_ptr = father_ptr[i+1];
      father_ptr[i+1] = new_ptr;
    }
    this->singleVectorSize = cur_lv_size;
    valueArray.clear();
    vLevel.pop_back();
  } else {
    std::cerr << "should not happen" << std::endl;
    assert(0);
  }
  if (mark) separate(lv-1);
  return 1;
}

bool SparlayStorage::devectorize() {
  int lv = vLevel.size()-1;
  bool mark = !(vLevel[lv]->type&LVFUSE);
  fuse(lv);
  auto& father_ptr = vLevel[lv]->ptr;
  auto& father_crd = vLevel[lv]->crd;
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vLevel[lv]->crd.size() == vectorArray.size());
    std::vector<int> new_crd;
    std::vector<bool> new_same_path;
    static std::vector<float> new_value;
    new_value.clear();
    size_t Size = this->singleVectorSize;
    if (vectorArray.size() != 0) {
      for (size_t i = 0; i < father_crd.size(); ++i) {
        assert(father_ptr[i+1] == prev_ptr+1);
        int cnt = 0;
        for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
          for (size_t k = 0; k < vectorArray[j].size(); ++k) {
            if (vectorArray[j][k]) {
              cnt++;
              new_crd.push_back(k);
              new_same_path.push_back(0);
              new_value.push_back(vectorArray[j][k]);
            }
          }
        }
        int new_ptr = father_ptr[i] + cnt;
        prev_ptr = father_ptr[i+1];
        father_ptr[i+1] = new_ptr;
      }
    }
    std::vector<int> empty_ptr = {};
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage(1, Size, new_crd, empty_ptr, new_same_path)));
    valueArray = std::move(new_value);
    this->clearVector();
  } else {
    std::cerr << "Should not happen" << std::endl;
    assert(0);
  }
  if (mark) separate(lv);
  return 1;
}

bool SparlayStorage::neg(int lv) {
  assert(!(vLevel[lv]->type & LVFUSE));
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    vLevel[lv]->crd[i] = -vLevel[lv]->crd[i];
  }
  return 1;
}

namespace decompose {

static int curRow;
static int innz;
static std::vector<int> vnnz;
static std::map<int, int> mnnz;
static int nnz_type;

void genWindowBuffer(SparlayStorage* T, SparlayWindow* W) {
  assert(W->M[1][0] == 0);
  if (W->M[0][1]) {
    assert(W->M[0][0] == 0);
    assert(W->M[1][1] == 0);
  }
  curRow = -2147483633;
  if (W->M[1][1]) {
    assert(W->M[1][1] == 1);
  }
  if (W->M[0][0]) {
    assert(W->M[0][0] == 1);
  }
  if (W->M[0][1]) {
    assert(W->M[0][1] == 1);
  }
  assert(!W->T[0][1]);
  assert(!W->T[1][1]);
  vnnz.clear();
  mnnz.clear();
  innz = 0;
  if (W->M[0][0] && W->M[1][1]) {
    nnz_type = 2;
  } else if (W->M[0][0]) {
    nnz_type = 0;
  } else if (W->M[0][1]) {
    nnz_type = 1;
    vnnz.resize(T->vLevel[2]->size,0);
  }
}

float getWindowSize(SparlayStorage* T, SparlayWindow* W) {
  float n = 1.0, m = 1.0;
  if (W->M[0][0] && W->M[1][1]) {
    n = (W->T[0][0] ? (float)W->T[0][0] : 1.0);
    m = (W->T[1][0] ? (float)W->T[1][0] : 1.0);
  } else if (W->M[0][0]) {
    m = T->vLevel[2]->size;
    n = (W->T[0][0] ? (float)W->T[0][0] : 1.0);
  } else if (W->M[0][1]) {
    n = T->vLevel[1]->size;
    m = (W->T[1][0] ? (float)W->T[1][0] : 1.0);
  }
  return n * m;
}

bool needSwitchBuffer(int newRow, SparlayWindow* W) {
  if (nnz_type == 1) return 0;
  if (W->T[0][0]) newRow /= W->T[0][0];
  return (newRow != curRow && nnz_type != 1);
}

void updateMaxDensity(float &mx, float wsize) {
  if (nnz_type == 0) {
    mx = std::max(mx, (float)innz / wsize);
  } else if (nnz_type == 1) {
    int max_nnz = 0;
    for (size_t i = 0; i < vnnz.size(); ++i) {
      max_nnz = std::max(max_nnz, vnnz[i]);
    }
    mx = std::max(mx, (float)max_nnz / wsize);
  } else {
    int max_nnz = 0;
    for (auto iter = mnnz.begin(); iter != mnnz.end(); iter++) {
      max_nnz = std::max(max_nnz, iter->second);
    }
    mx = std::max(mx, (float)max_nnz / wsize);
  }
}

void pushNewCrd(int i, int j, SparlayWindow* W) {
  if (W->T[0][0]) i /= W->T[0][0];
  if (W->T[1][0]) j /= W->T[1][0];
  if (nnz_type == 0) {
    assert(i == curRow);
    innz += 1;
  } else if (nnz_type == 1) {
    vnnz[j] += 1;
  } else {
    assert(i == curRow);
    mnnz[j] += 1;
  }
}

void switchBuffer(int i, SparlayWindow* W) {
  if (W->T[0][0]) i /= W->T[0][0];
  #ifdef DEBUG
  std::cerr << i << ' ' << curRow << std::endl;
  #endif
  assert(i > curRow);
  assert(nnz_type != 1);
  curRow = i;
  if (nnz_type == 0) {
    innz = 0;
  } else {
    mnnz.clear();
  }
}

void clearBuffer() {
  if (nnz_type == 0) {
    innz = 0;
  } else if (nnz_type == 1) {
    vnnz.clear();
  } else {
    mnnz.clear();
  }
}

float getMaxDensity(SparlayStorage* T, SparlayWindow* W) {
  genWindowBuffer(T, W);
  assert(T->vLevel.size() == (size_t)3);
  assert(T->vLevel[1]->crd.size() == T->vLevel[2]->crd.size());
  float wsize = getWindowSize(T, W);
  std::cerr << wsize << std::endl;
  float mx_density = 0.0;
  for (size_t i = 0; i < T->vLevel[1]->crd.size(); ++i) {
    if (needSwitchBuffer(T->vLevel[1]->crd[i], W)) {
      updateMaxDensity(mx_density, wsize);
      switchBuffer(T->vLevel[1]->crd[i], W);
    }
    pushNewCrd(T->vLevel[1]->crd[i], T->vLevel[2]->crd[i], W);
  }
  updateMaxDensity(mx_density, wsize);
  clearBuffer();
  return mx_density;
}

void pushCrd(SparlayStorage* T, int i, int j, float val) {
  T->vLevel[1]->crd.push_back(i);
  if (T->vLevel[1]->crd.size() == (size_t)1) T->vLevel[1]->same_path.push_back(0);
  else T->vLevel[1]->same_path.push_back(i == (*(T->vLevel[1]->crd.end()-2)));
  T->vLevel[2]->crd.push_back(j);
  T->vLevel[2]->same_path.push_back(0);
  T->valueArray.push_back(val);
}

void emplaceCrd(SparlayStorage* T, SparlayWindow* W, int L, int R, std::vector<SparlayStorage*>& cand, const std::vector<float>& thres, const float win_size, const float mx_density) {
  if (L > R) {
    assert(L == 0 && R == -1);
    return;
  }
  for (int i = L; i <= R; ++i) {
    int win_i = T->vLevel[1]->crd[i];
    int win_j = T->vLevel[2]->crd[i];
    if (W->T[0][0]) win_i /= W->T[0][0];
    if (W->T[1][0]) win_j /= W->T[1][0];
    float curDens = -1.0;
    if (nnz_type == 0) {
      assert(win_i == curRow);
      curDens = (float)innz / win_size;
    } else if (nnz_type == 1) {
      assert(L == 0 && (size_t)R == T->vLevel[1]->crd.size()-(size_t)1);
      curDens = (float)vnnz[win_j] / win_size;
    } else {
      assert(win_i == curRow);
      curDens = (float)mnnz[win_j] / win_size;
    }
    assert(curDens != -1.0);
    curDens /= mx_density;
    size_t target = 0;
    for (target = 0; target < thres.size(); ++target) {
      if (curDens < thres[target]) {
        break;
      }
    }
    pushCrd(cand[target], T->vLevel[1]->crd[i], T->vLevel[2]->crd[i], T->valueArray[i]);
  }
}

// sortCrd(sparT, swin, thres_data, mx_density, cand)
void sortCrd(SparlayStorage* T, SparlayWindow* W, const std::vector<float>& thres, const float mx_density, std::vector<SparlayStorage*>& cand) {
  genWindowBuffer(T, W);
  assert(T->vLevel.size() == (size_t)3);
  assert(T->vLevel[1]->crd.size() == T->vLevel[2]->crd.size());
  int prev_id = 0;
  float wsize = getWindowSize(T, W);
  for (size_t i = 0; i < cand.size(); ++i) {
    cand[i]->initCOO(T->vLevel[1]->size, T->vLevel[2]->size);
  }
  for (int i = 0; i < (int)T->vLevel[1]->crd.size(); ++i) {
    if (needSwitchBuffer(T->vLevel[1]->crd[i], W)) {
      emplaceCrd(T, W, prev_id, i-1, cand, thres, wsize, mx_density);
      switchBuffer(T->vLevel[1]->crd[i], W);
      prev_id = i;
    }
    pushNewCrd(T->vLevel[1]->crd[i], T->vLevel[2]->crd[i], W);
  }
  emplaceCrd(T, W, prev_id, T->vLevel[1]->crd.size()-1, cand, thres, wsize, mx_density);
  clearBuffer();
  //finalize sparT
  for (size_t i = 0; i < cand.size(); ++i) {
    cand[i]->finalizeCOO();
  }
}

}

extern "C" {
    // refactor into a swiss army knife function in the future
  void* readSparseCoordinate(void* ptr) {
    char* fileName = static_cast<char *>(ptr);
    char field[64];
    char symmetry[64];                                               
                                                                                                
    FILE *file = fopen(fileName, "r");   
    printf("filename %s\n", fileName);                                                       
    if (!file) {                                                                                
        fprintf(stderr, "Cannot find %s\n", fileName);                                          
        exit(1);                                                                                
    }                                                                                           
                                                                                                
    uint64_t metaData[512];                                                                     
    if (strstr(fileName, ".mtx")) {                                                             
        readMTXHeader(file, fileName, metaData, field, symmetry);                                                
    } else if (strstr(fileName, ".tns")) {                                                      
        readFROSTTHeader(file, fileName, metaData);                                             
    } else {                                                                                    
        fprintf(stderr, "Unknown format %s\n", fileName);                                       
        exit(1);                                                                                
    } 

    // printf("in getTensorIndices  :\n");
    // for (unsigned i = 0; i < 4; i++)
    //     printf("metaData[%u] = %lu \n", i, metaData[i]);                                                                                          
                                                                                                
    uint64_t rank = metaData[0];    
    uint64_t nnz = metaData[1]; 

    bool notFieldPattern = strcmp(toLower(field), "pattern");
    if (!strcmp(toLower(field), "complex")) {
        fprintf(stderr, "Complex data type not yet supported.\n");                                       
        exit(1); 
    } 
    if (strcmp(toLower(symmetry), "general")) {
        fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
        exit(1); 
    }                                                               
    
    static SparseCoordinate<uint64_t, double> tensor(rank);
    // read data                                              
    for (unsigned i = 0; i < nnz; i++) {   
        std::vector<uint64_t> indices;                                                       
        uint64_t idx = -1;                                                                      
        for (uint64_t r = 0; r < rank; r++) {                                                   
            if (fscanf(file, "%" PRIu64, &idx) != 1) {                                          
                fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                exit(1);                                                                        
            }
            indices.push_back(idx - 1);
        }
        double val;
        if (!notFieldPattern) {
            // Field is pattern
            val = 1;
        } else {
            if (fscanf(file, "%lg\n", &val) != 1) {
                fprintf(stderr, "Cannot find next value in %s\n", fileName);
                exit(1);
            }
        }
        tensor.insert(indices, val);
    }

    fclose(file);
    return &tensor;
  }

  void* _mlir_ciface_sptFromFile(void* ptr) {
    std::ios::sync_with_stdio(0);
    #ifdef DEBUG
    auto tic = TI;
    #endif
    char* fileName = static_cast<char*>(ptr);
    std::ifstream fin(fileName);
    void* ret = readFromFile(fin);
    fin.close();
    #ifdef DEBUG
    std::cerr << std::endl << "Read from file done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    return ret;
  }

  void _mlir_ciface_sptTic() {
    Perf::tic();
  }

  void _mlir_ciface_sptToc() {
    Perf::toc();
  }

  void _mlir_ciface_sptCheck(void* A, void* B) {
    SparlayStorage* aa = (SparlayStorage*)(A);
    SparlayStorage* bb = (SparlayStorage*)(B);
    if ((*aa) == (*bb)) {
      std::cerr << "Check Success" << std::endl;
    } else {
      std::cerr << "Check Failed" << std::endl;
      assert(0);
    }
  }

  void* _mlir_ciface_sptCopy(void* A) {
    SparlayStorage* aa = (SparlayStorage*)(A);
    SparlayStorage* ret = new SparlayStorage();
    (*ret) = aa->copy();
    return (void*)ret;
  }

  void* _mlir_ciface_sptFuse(void* ptr, int lv) {
    #ifdef DEBUG
      auto tic = TI;
    #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->fuse(lv+1);
      // sparT->Print(std::cerr, 1);
      #ifdef DEBUG
      std::cerr << std::endl << "Fuse done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      return (void*)sparT;
  }

  void* _mlir_ciface_sptGrow(void* ptr, int lv) {
    #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->grow(lv+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Grow done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      return (void*)sparT;
  }

  void* _mlir_ciface_sptTrim(void* ptr, int lv) {
    #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->trim(lv+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Trim done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptSeparate(void* ptr, int lv) {
    #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->separate(lv+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Separate done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptSwap(void* ptr, int LU, int LD) {
    #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->swap(LU+1, LD+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Swap done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptSub(void* ptr, int Ltarget, int Lsrc) {
      #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->sub(Ltarget+1, Lsrc+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Sub done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr, 1);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptAdd(void* ptr, int Ltarget, int Lsrc) {
      #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->add(Ltarget+1, Lsrc+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Add done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr, 1);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptNeg(void* ptr, int lv) {
    #ifdef DEBUG
      auto tic = TI;
      #endif
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->neg(lv+1);
      #ifdef DEBUG
      std::cerr << std::endl << "Neg done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      // sparT->Print(std::cerr);
      return (void*)sparT;
  }

  void* _mlir_ciface_sptTileSplit(void* ptr, int lv, int factor) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->tile_split(lv+1, factor);
    // sparT->Print(std::cerr, 1);
    #ifdef DEBUG
    std::cerr << std::endl << "Tile Split done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptTileMerge(void* ptr, int lv, int factor) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->tile_merge(lv+1, factor);
    #ifdef DEBUG
    std::cerr << std::endl << "Tile Merge done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptMove(void* ptr, int srcLv, int dstLv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->moveLv(srcLv+1, dstLv+1);
    // sparT->Print(std::cerr, 1);
    #ifdef DEBUG
    std::cerr << std::endl << "Move done, time = " << (TI-tic)*1000.0 << "(ms)" << std::endl;
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptVectorize(void* ptr, int lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->vectorize(lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Vectorize done, time = " << (TI-tic)*1000.0 << "(ms)" << std::endl;
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptDevectorize(void* ptr) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->devectorize();
    #ifdef DEBUG
    std::cerr << std::endl << "Devectorize done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    return (void*)sparT;
  }

  void _mlir_ciface_sptPrint(void* ptr) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->Print(std::cerr, 1);
  }

  void* _mlir_ciface_structAccess(void* ptr, uint64_t index) {
    SparlayStruct* SS = (SparlayStruct*)(ptr);
    return SS->get(index);
  }

  void* _mlir_ciface_spwNew() {
    SparlayWindow* ret = new SparlayWindow;
    return (void*)ret;
  }

  void* _mlir_ciface_spwAssign(void* ptr, uint64_t i, uint64_t j, int v) {
    SparlayWindow* ret = (SparlayWindow*)(ptr);
    ret->assign(i,j,v);
    return (void*)ret;
  }

  void* _mlir_ciface_spwTile(void* ptr, uint64_t i, uint64_t type, int size) {
    SparlayWindow* ret = (SparlayWindow*)(ptr);
    ret->tile(i,type,size);
    return (void*)ret;
  }

  void* _mlir_ciface_sptSplit(StridedMemRefType<float, 1>* thres, void* ptr, void* win) {
    std::cerr << "enter split" << std::endl;
    SparlayStruct* ret = new SparlayStruct;
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    assert(thres->offset == 0);
    int64_t size = thres->sizes[0];
    float* _thres_data = thres->data;
    std::vector<float> thres_data;
    for (int64_t i = 0; i < size; ++i) { thres_data.push_back(_thres_data[i]); }
    std::cerr << "size = " << size << std::endl;
    std::vector<SparlayStorage*> cand;
    for (int64_t i = 0; i < size + 1; ++i) {
      cand.push_back(new SparlayStorage);
      SparlayStorage* newT = new SparlayStorage;
      (*newT) = sparT->copy();
      // ret->vec.push_back((void*)newT);
    }
    SparlayWindow* swin = (SparlayWindow*)(win);
    std::cerr << "==============" << std::endl;
    std::cerr << "window: " << std::endl;
    std::cerr << "affine: " << std::endl;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        std::cerr << swin->M[i][j] << ' ';
      }
      std::cerr << std::endl;
    }
    std::cerr << "tile: " << std::endl;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        std::cerr << swin->T[i][j] << ' ';
      }
      std::cerr << std::endl;
    }
    std::cerr << "===============" << std::endl;
    float mx_density = decompose::getMaxDensity(sparT, swin);
    std::cerr << "mx_density = " << mx_density << std::endl;
    decompose::sortCrd(sparT, swin, thres_data, mx_density, cand);
    for (size_t i = 0; i < cand.size(); ++i) {
      cand[i]->Print(std::cerr, 1);
      ret->vec.push_back((void*)cand[i]);
    }
    std::cerr << "leave split" << std::endl;
    return (void*)ret;
  }

  void _mlir_ciface_getCrd(StridedMemRefType<int, 1>* ref, void* ptr, uint64_t dim) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    ref->basePtr = ref->data = sparT->vLevel[dim]->crd.data();
    ref->offset = 0;
    ref->sizes[0] = sparT->vLevel[dim]->crd.size();
    ref->strides[0] = 1;
  }

  void _mlir_ciface_getPtr(StridedMemRefType<int, 1>* ref, void* ptr, uint64_t dim) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    ref->basePtr = ref->data = sparT->vLevel[dim]->ptr.data();
    ref->offset = 0;
    ref->sizes[0] = sparT->vLevel[dim]->ptr.size();
    ref->strides[0] = 1;
  }

  void _mlir_ciface_getValue(StridedMemRefType<float, 1>* ref, void* ptr, uint64_t dim) {
    assert(dim == (uint64_t)0);
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    ref->basePtr = ref->data = sparT->valueArray.data();
    ref->offset = 0;
    ref->sizes[0] = sparT->valueArray.size();
    ref->strides[0] = 1;
  }

  uint64_t _mlir_ciface_getSize(void* ptr, uint64_t dim) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    return sparT->vLevel[dim+1]->crd.size();
  }

// #define GETINDICES(TYPE)
  void _mlir_ciface_getTensorIndices(StridedMemRefType<uint64_t, 1> *ref, void *ptr, uint64_t dim) {   

      SparseCoordinate<uint64_t, double> *tensor = nullptr;
      tensor = static_cast<SparseCoordinate<uint64_t, double> *>(ptr);

      std::vector<uint64_t> *index;

      tensor->getIndex(&index, dim);

      ref->basePtr = ref->data = index->data();  
      ref->offset = 0;  
      ref->sizes[0] = index->size();  
      ref->strides[0] =1; 

      // printf("ref->basePtr: %x\n", ref->basePtr);
      // printf("ref->size: %zu\n", index->size());
      // printf("ref->data: ");
      // for (unsigned i = 0; i < index->size(); i++) {
      //     printf("%lu  ", *(ref->data + ref->offset + i * ref->strides[0]));
      // }
      // printf("\n");
  }

// #define GETVALUES(TYPE)
  void _mlir_ciface_getTensorValues(StridedMemRefType<double, 1> *ref, void *ptr) {
      SparseCoordinate<uint64_t, double> *tensor = nullptr;
      tensor = static_cast<SparseCoordinate<uint64_t, double> *>(ptr);

      std::vector<double> *value;

      tensor->getValue(&value);

      ref->data = value->data();    
      ref->basePtr = value->data();
      ref->offset = 0;  
      ref->sizes[0] = value->size();  
      ref->strides[0] = 1; 

      // printf("value->basePtr: %x\n", ref->basePtr);
      // printf("value->size: %zu\n", value->size());
      // printf("value->data: ");
      // for (unsigned i = 0; i < value->size(); i++) {
      //     printf("%f  ", *(ref->data + ref->offset + i * ref->strides[0]));
      // }
      // printf("\n");

  }

  void _mlir_ciface_calculateCSRSpMV(StridedMemRefType<double, 1> *out, 
                                      StridedMemRefType<uint64_t, 1> *ptr, 
                                      StridedMemRefType<uint64_t, 1> *col, 
                                      StridedMemRefType<double, 1> *value, 
                                      StridedMemRefType<double, 1> *input) {
    uint64_t row = ptr->sizes[0] - 1;
    double *result = new double[row];
//      printf("row size is: %d\n", row);
    for(uint64_t i = 0; i < row; i++) {
      double temp = 0;
      for(uint64_t j = ptr->data[i]; j < ptr->data[i+1]; j++) {
      temp += value->data[j] * input->data[col->data[j]];
  //	  printf("value->data[%d] is: %f, col->data[%d] is: %d, input->data[%d] is: %f\n", j, value->data[j], j, col->data[j], col->data[j], input->data[col->data[j]]);
      }
      result[i] = temp;
//        printf("outdata[%d] is %f\n", i, out->data[i]);
    }
    out->data = result;
    out->basePtr = result;
    out->offset = 0;  
    out->strides[0] = 1;
  }  

  void delSparlayTensor(void *tensor) {
    delete static_cast<SparlayStorage *>(tensor);
  }

  uint64_t _mlir_ciface_sparseDimSize(void *tensor, uint64_t d) {
    return static_cast<SparlayStorage *>(tensor)->getDimSize(d);
  }

  void _mlir_ciface_endInsert(void *tensor) {
//    std::cout << "Start endInsert " << std::endl;
    return static_cast<SparlayStorage *>(tensor)->endInsert();
  }

  void _mlir_ciface_lexInsert(void *tensor,
                              StridedMemRefType<uint64_t, 1> *cref, 
                              StridedMemRefType<float, 0> *vref) {
//    std::cout << "Start lexInsert " << std::endl;        
    assert(tensor &&cref &&vref);                                        
    assert(cref->strides[0] == 1);                               
    uint64_t *cursor = cref->data + cref->offset;                
    assert(cursor);                                                       
    float *value = vref->data + vref->offset;                                    
    static_cast<SparlayStorage *>(tensor)->lexInsert(cursor, *value);
  }

  void _mlir_ciface_expInsert(                                          
        void *tensor, StridedMemRefType<uint64_t, 1> *cref,                   
        StridedMemRefType<float, 1> *vref, StridedMemRefType<bool, 1> *fref,       
        StridedMemRefType<uint64_t, 1> *aref, uint64_t count) {  
//    std::cerr << "Start Expand Insert " << std::endl;           
    assert(tensor &&cref &&vref &&fref &&aref);                            
    assert(cref->strides[0] == 1);                                          
    assert(vref->strides[0] == 1);                                          
    assert(fref->strides[0] == 1);                                           
    assert(aref->strides[0] == 1);                                            
    assert(vref->sizes[0] == fref->sizes[0]);                                 
    uint64_t *cursor = cref->data + cref->offset;                          
    float *values = vref->data + vref->offset;                                   
    bool *filled = fref->data + fref->offset;                                 
    uint64_t *added = aref->data + aref->offset;
//    std::cerr << "Enter Expand Insert " << std::endl;                            
    static_cast<SparlayStorage *>(tensor)->expInsert(               
        cursor, values, filled, added, count);                          
  }

  void* _mlir_ciface_newSparlayTensor(StridedMemRefType<SparlayDimLevelType, 1> *aref,
                               StridedMemRefType<uint64_t, 1> *sref,
                               StridedMemRefType<uint64_t, 1> *pref, void *ptr ) {

    std::cout << "Start newSparlayTensor " << std::endl;
    assert(aref && sref && pref);
    assert(aref->strides[0] == 1 && sref->strides[0] == 1 && pref->strides[0] == 1);
    assert(aref->sizes[0] == sref->sizes[0] && sref->sizes[0] == pref->sizes[0]);
    const SparlayDimLevelType *sparsity = aref->data + aref->offset;
    uint64_t *shape = sref->data + sref->offset;
    uint64_t *perm = pref->data + pref->offset;
    uint64_t rank = aref->sizes[0];
    std::vector<uint64_t> vshape;
//    std::cerr << "sref->offset " << sref->offset << std::endl;
//    std::cerr << *shape << std::endl;
//    std::cerr << *(shape+1) << std::endl;
//    std::cerr << *(shape+2) << std::endl;
    for(uint64_t i = 0; i < rank; ++i) {
      vshape.push_back(shape[i]);
    }
//    std::cout << "Create SparlayStorage " << std::endl;
    auto *tensor = new SparlayStorage(vshape, perm, sparsity);
    return tensor;
  }

    // void _mlir_ciface_release(void *ptr) {
    //     delete []ptr;
    // }

    // void delSparseCoordinate(void *tensor) {
    //     delete static_cast<SparseCoordinate<uint64_t, double> *>(tensor);
    // }

    using index_type = uint64_t;
    index_type getTensorDim(void* ptr, index_type dim) {
        char* fileName = static_cast<char*>(ptr);
        char field[64];
        char symmetry[64];   

        FILE *file = fopen(fileName, "r"); 
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }

        index_type metaData[512]; 
        if (strstr(fileName, ".mtx")) {                                                             
            readMTXHeader(file, fileName, metaData, field, symmetry);                                                
        } else if (strstr(fileName, ".tns")) {                                                      
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {                                                                                    
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        }

        index_type request_dim = dim + 2;
        return metaData[request_dim];
    }

    // Vanilla DIA SpMM
    void _mlir_ciface_kernel_dia_spmm(StridedMemRefType<float, 2> *outC,
                                      void* inA, 
                                      StridedMemRefType<float, 2> *inB, 
                                      StridedMemRefType<float, 2> *inC) {
        printf("enter in kernel_dia_spmm\n");
        SparlayStorage* spA = (SparlayStorage*)inA;
        // printf("spA->vLevel.size = %zu \n", spA->vLevel.size());
        std::shared_ptr<LevelStorage> spA_dim0 = spA->vLevel[0];
        std::shared_ptr<LevelStorage> spA_dim1 = spA->vLevel[1];
        // std::vector<float> spA_data = spA->valueArray;
        std::vector< std::vector<float> > spA_vector = spA->vectorArray;
        // printf("spA_data.size = %zu\n", spA_data.size());
        // printf("spA_vector.size = %zu\n", spA_vector.size());
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];
        int64_t kSize = inC->sizes[1];
        printf("iSize = %ld, jSize = %ld, kSize = %ld\n", iSize, jSize, kSize);
        // std::vector<int> spA_dim0_crd = spA_dim0->crd;
        std::vector<int> spA_dim1_crd = spA_dim1->crd;
        // // int spA_dim0_size = spA_dim0->size;
        // // int spA_dim1_size = spA_dim1->size;
        // printf("spA_dim0_crd = ");
        // for (auto elm: spA_dim0_crd) {
        //   printf("%d ", elm);
        // }
        // printf("\n");
        // printf("spA_dim1_crd = ");
        // for (auto elm: spA_dim1_crd) {
        //   printf("%d ", elm);
        // }
        // printf("\n");
        // printf("spA_dim0_size = %d, spA_dim1_size = %d \n",spA_dim0_size,spA_dim1_size);
        
        // printf("spA_vector = \n");
        // for (auto v: spA_vector) {
        //   for (auto elm: v) {
        //     printf("%f ", elm);
        //   }
        //   printf("\n");
        // }
        // printf("\n");

        // A*B + C
        outC->basePtr = outC->data = inC->data;
        outC->offset = inC->offset;
        outC->strides[0] = outC->strides[1] = 1;
        outC->sizes[0] = inC->sizes[0];
        outC->sizes[1] = inC->sizes[1];
        // printf("inB_data = \n");
        // for (unsigned j=0; j < jSize; j++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", inB->data[j*kSize+k]);
        //   printf("\n");
        // }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", outC->data[i*iSize+k]);
        //   printf("\n");
        // }
        for (unsigned diag = 0; diag < spA_dim1_crd.size(); diag++) {
          for (int i = 0; i < iSize; i++) {
            int j = spA_dim1_crd[diag] + i;
            if (j >=0 && j < jSize) {
              for (int k = 0; k < kSize; k++) {
                outC->data[i*kSize+k] += spA_vector[diag][i] * inB->data[j*kSize+k];
              }
            }
          }
        }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++) {
        //   for (unsigned k = 0; k < kSize; k++)
        //     printf("%f ", outC->data[i*kSize+k]);
        //   printf("\n");
        // }
        // printf("\n");
    }

    void _mlir_ciface_kernel_dia_spmv(StridedMemRefType<float, 1> *outC,
                                      void* inA, 
                                      StridedMemRefType<float, 1> *inB, 
                                      StridedMemRefType<float, 1> *inC) {
        SparlayStorage* spA = (SparlayStorage*)inA;
        int32_t* spA_dim0_crd = spA->vLevel[1]->crd.data();
        uint64_t spA_dim0_size = spA->vLevel[1]->crd.size();
        std::vector< std::vector<float> > spA_vector = spA->vectorArray;
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];

        // A*B + C
        outC->basePtr = outC->data = inC->data;
        outC->offset = inC->offset;
        outC->strides[0] = 1;
        outC->sizes[0] = inC->sizes[0];

        uint64_t diag;
        int i, j;
        float sum;
        double start = omp_get_wtime();
        for (unsigned time = 0; time < 10000; time++) {
          #pragma omp parallel for private(diag,i,j,sum) 
          for (diag = 0; diag < spA_dim0_size; diag++) {
            sum=0;
            #pragma omp simd reduction(+:sum)
            for (i = 0; i < iSize; i++) {
              j = spA_dim0_crd[diag] + i;
              if (j >=0 && j < jSize) {
                sum += spA_vector[diag][i] * inB->data[j];
              }
            }
            outC->data[i]=sum;
          }
        }
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/10000 << " ms"<< std::endl;
    }

    void _mlir_ciface_calculateCOOSpMV(StridedMemRefType<float, 1> *out, void *ptr, 
                                     StridedMemRefType<float, 1> *input, StridedMemRefType<float, 1> *ref) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    int32_t *row_crd = sparT->vLevel[1]->crd.data();
    int32_t *col_crd = sparT->vLevel[2]->crd.data();
    float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::cout << "nnz is " << nnz << std::endl;
    std::cout << input->data << std::endl;
    std::cout << ref->data << std::endl;
    for(uint64_t i = 0; i < nnz; i++) {
      int32_t rowInd =row_crd[i];
      int32_t colInd = col_crd[i];
      ref->data[rowInd] += values[i] * input->data[colInd];
    }
    std::cout << "End loop " << std::endl;
    out->data = ref->data;
    out->basePtr = ref->data;
    out->offset = 0;  
    out->strides[0] = 1;
  }

  void _mlir_ciface_calculateCOOSpMM(StridedMemRefType<float, 2> *out, void *ptr, 
                                     StridedMemRefType<float, 2> *input, StridedMemRefType<float, 2> *ref) {
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    int32_t *row_crd = sparT->vLevel[1]->crd.data();
    int32_t *col_crd = sparT->vLevel[2]->crd.data();
    float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    uint64_t kSize = input->sizes[1];
    std::cout << "nnz is " << nnz << std::endl;
    std::cout << input->data << std::endl;
    std::cout << ref->data << std::endl;
    for(uint64_t i = 0; i < nnz; i++) {
      for(uint64_t k = 0; k < kSize; k++) {
        int32_t rowInd =row_crd[i];
        int32_t colInd = col_crd[i];
        ref->data[rowInd*kSize + k] += values[i] * input->data[colInd*kSize + k];
      }
    }
    std::cout << "End loop " << std::endl;
    out->data = ref->data;
    out->basePtr = ref->data;
    out->offset = 0;  
    out->strides[0] = 1;
    out->strides[1] = 1;
  }

  void _mlir_ciface_kernel_bdia_spmv_iter(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
        
        int ib, i, k, diag, is, ie;
        DataType sum;
        SparlayStorage* spA_CSR = (SparlayStorage*)inA_CSR;
        SparlayStorage* spA_BDIA = (SparlayStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];
        double start = omp_get_wtime();

        for (unsigned time = 0; time < 10000; time++) {
          #pragma omp parallel for private(ib,i,k,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=0;
              #pragma omp simd reduction(+:sum)
              for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                // std::cout << "i="<<i<<", k="<<k<<", CSR_value[k]="<<CSR_value[k]<<
                //   ", CSR_dim2_crd[k]="<<CSR_dim2_crd[k]<<", inB->data[CSR_dim2_crd[k]]="
                //   <<inB->data[CSR_dim2_crd[k]]<<std::endl;
                sum+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]]);
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
              }
              inC->data[i] = sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              #pragma omp simd
              for (i = is; i < ie; i++) {
                inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
                // if(i==0) {
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              inB->data[i] = inC->data[i];
            }
          }
        }
        
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/10000 << " ms"<< std::endl;

        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->strides[0] = inC->strides[0];
        for(unsigned i = 0; i <4; i++ )
          std::cout <<"outC->data["<<i<<"]=" << outC->data[i]<<std::endl;

    }

    void _mlir_ciface_kernel_bdia_spmm_iter(StridedMemRefType<DataType, 2> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 2> *inB, 
                                      StridedMemRefType<DataType, 2> *inC) {
        
        int ib, i, k, j, diag, is, ie;
        
        SparlayStorage* spA_CSR = (SparlayStorage*)inA_CSR;
        SparlayStorage* spA_BDIA = (SparlayStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];
        int64_t kSize = inB->sizes[1];
        assert(kSize == inC->sizes[1]);
        DataType *sum;
        double start = omp_get_wtime();

        for (unsigned time = 0; time < 1000; time++) {
          #pragma omp parallel for private(ib,i,k,j,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=new DataType[kSize]();
              // sum=0;
              
              for (j=0;j<kSize;j++) {
                #pragma omp simd reduction(+:sum[j])
                for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                // std::cout << "i="<<i<<", k="<<k<<", CSR_value[k]="<<CSR_value[k]<<
                //   ", CSR_dim2_crd[k]="<<CSR_dim2_crd[k]<<", inB->data[CSR_dim2_crd[k]]="
                //   <<inB->data[CSR_dim2_crd[k]]<<std::endl;
                
                  sum[j]+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]*kSize+j]);
                }
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
                inC->data[i*kSize+j] = sum[j];
              }
              // for (j=0;j<kSize;j++) {
              //   inC->data[i*kSize+j] = sum[j];
              // }
              delete[] sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              // #pragma omp simd
              for (i = is; i < ie; i++) {
                #pragma omp simd
                for(j=0; j<kSize;j++) {
                
                  inC->data[i*kSize+j] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[(i+diag)*kSize+j];
                }
                // if(i==0) { 
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
            // Robot motion planing
            // for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
            //   for(j=0; j<kSize;j++) {
            //     inB->data[i*kSize+j] = inC->data[i*kSize+j];
            //   }
            // }
          }
        }
        
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/1000 << " ms"<< std::endl;

        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->sizes[1] = inC->sizes[1];
        outC->strides[0] = inC->strides[0];
        outC->strides[1] = inC->strides[1];
        for(unsigned i = 0; i <4; i++ ) {
          for(unsigned j = 0; j <4; j++ )
            std::cout <<outC->data[i*kSize+j]<<"  ";
          std::cout << std::endl;
        }

    }


  void _mlir_ciface_kernel_bdia_spmv(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
        
        // std::cout << "inB = ";
        // for (unsigned i = 0; i < jSize; i++)
        //   std::cout << inB->data[i] << " ";
        // std::cout << std::endl;

        

        // A*B + C
        // outC->basePtr = outC->data = inC->data;
        // outC->offset = inC->offset;
        // outC->strides[0] = 1;
        // outC->sizes[0] = inC->sizes[0];
        // printf("inC_data = \n");
        // for (unsigned i=0; i < 4; i++)
        //     printf("%f ", inC->data[i]);
        // printf("\n");
        int ib, i, k, diag, is, ie;
        DataType sum;
        SparlayStorage* spA_CSR = (SparlayStorage*)inA_CSR;
        SparlayStorage* spA_BDIA = (SparlayStorage*)inA_BDIA;
        int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
        int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
        int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
        int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
        int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
        DataType* CSR_value = spA_CSR->valueArray.data();
        
        // float* inB_data = inB->data;
        // float* inC_data = inC->data;
        int blockSize = spA_BDIA->vLevel[3]->size;
        std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
        int64_t iSize = inC->sizes[0];
        int64_t jSize = inB->sizes[0];
        double start = omp_get_wtime();
        // std::cout << "n_blocks = " <<  n_blocks << std::endl;
        // // std::cout << "BDIA_dim1_ptr_size = " << BDIA_dim1_ptr_size << std::endl;
        // // std::cout << "spA->vLevel[3]->size = " << blockSize << std::endl;
        // std::cout << "spA_BDIA->vLevel[1]->ptr = ";
        // for (auto x: spA_BDIA->vLevel[1]->ptr)
        //   std::cout << x << "  ";
        // std::cout << std::endl;
        // std::cout << "spA_BDIA->vLevel[2]->crd = ";
        // for (auto x: spA_BDIA->vLevel[2]->crd)
        //   std::cout << x << "  ";
        // std::cout << std::endl;
        // std::cout << "spA_BDIA->vectorArray = ";
        // // for (auto i:spA_vector) {
        // //   for (auto j: i) {
        // //     std::cout << j << "  ";
        // //   }
        // //   std::cout << std::endl;
        // // }
        // for (unsigned i = 0; i < BDIA_vector.size(); i++) {
        //   // for (unsigned j = 0; j < BDIA_vector[i].size(); j++)
        //     std::cout << BDIA_vector[i] << " ";
        //   std::cout << std::endl;
        // }
        // std::cout << "CSR_dim1_ptr size = " <<  spA_CSR->vLevel[1]->ptr.size() << std::endl;
        // std::cout << "spA_CSR->vLevel[1]->ptr = ";
        // for (auto j: spA_CSR->vLevel[1]->ptr) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        // std::cout << "CSR_dim2_crd size = " <<  spA_CSR->vLevel[2]->crd.size() << std::endl;
        // std::cout << "spA_CSR->vLevel[2]->crd = ";
        // for (auto j: spA_CSR->vLevel[2]->crd) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        // std::cout << "CSR_value size = " <<  spA_CSR->valueArray.size() << std::endl;
        // std::cout << "spA_CSR->valueArray = ";
        // for (auto j: spA_CSR->valueArray) {
        //     std::cout << j << "  ";
        //   }
        // std::cout << std::endl;
        for (unsigned time = 0; time < 10000; time++) {
          #pragma omp parallel for private(ib,i,k,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=0;
              #pragma omp simd reduction(+:sum)
              for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
                // std::cout << "i="<<i<<", k="<<k<<", CSR_value[k]="<<CSR_value[k]<<
                //   ", CSR_dim2_crd[k]="<<CSR_dim2_crd[k]<<", inB->data[CSR_dim2_crd[k]]="
                //   <<inB->data[CSR_dim2_crd[k]]<<std::endl;
                sum+=CSR_value[k]*(inB->data[CSR_dim2_crd[k]]);
                // if(i==0) {
                  // std::cout << "i="<<i<<", k="<<k<<", CSR_value="<<CSR_value[k]<<", crd="<<CSR_dim2_crd[k]
                  // <<", inB->data="<<inB->data[CSR_dim2_crd[k]]<<std::endl;
                // }
              }
              inC->data[i] = sum;
              // if(i==0)
                // std::cout<<"sum="<< sum<<", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
            }
            // std::cout << "tid = " << omp_get_thread_num() << std::endl;
            for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
              diag = BDIA_dim2_crd[k];
              is = std::max(ib*blockSize, -diag);
              ie = std::min({(ib+1)*blockSize, (int)iSize-diag, (int)iSize});
              #pragma omp simd
              for (i = is; i < ie; i++) {
                inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
                // if(i==0) {
                //   std::cout << "i="<<i<<", i+diag="<<i+diag<<", BDIA_vector="<<BDIA_vector[k*blockSize+i-ib*blockSize]
                //     << ", inB->data="<< inB->data[i+diag] << ", inC->data["<<i<<"]="<<inC->data[i]<<std::endl;
                // }
              }
              // for (i = 0; i < blockSize; i++) {
              //   if ((i+ib*blockSize+diag >=0) && (i+ib*blockSize+diag < jSize))
              //     inC->data[i+ib*blockSize] += BDIA_vector[k*blockSize+i] * inB->data[i+ib*blockSize+diag];
              // }
            }
          }
        }
        
        double end = omp_get_wtime();
        std::cout << "omp time = " << end-start << " s"<< std::endl;
        std::cout << "avg time = " << (end-start)*1000/10000 << " ms"<< std::endl;

        outC->data = inC->data;
        outC->basePtr = inC->basePtr;
        outC->offset = inC->offset;  
        outC->sizes[0] = inC->sizes[0];
        outC->strides[0] = inC->strides[0];
        for(unsigned i = 0; i <4; i++ )
          std::cout <<"outC->data["<<i<<"]=" << outC->data[i]<<std::endl;
          // std::cout <<"inC->data[0]=" <<inC->data[0] << ", outC->data[0]=" << outC->data[0]<<std::endl;
        // std::cout << "spA->vectorArray = " << std::endl;
        // // for (auto i:spA_vector) {
        // //   for (auto j: i) {
        // //     std::cout << j << "  ";
        // //   }
        // //   std::cout << std::endl;
        // // }
        // for (unsigned i = 0; i < spA_vector.size(); i++) {
        //   for (unsigned j = 0; j < spA_vector[i].size(); j++)
        //     std::cout << spA_vector[i][j] << " ";
        //   std::cout << std::endl;
        // }
        // printf("outC_data = \n");
        // for (unsigned i=0; i < iSize; i++)
        //     printf("%f ", outC->data[i]);
        // printf("\n");
    }

  void* _mlir_ciface_decompose_BDIA(void* ptr, int32_t blockSize, float thres) {
    SparlayStorage* sparT = (SparlayStorage*)ptr;
    // int32_t *row_crd = sparT->vLevel[1]->crd.data();
    // int32_t *col_crd = sparT->vLevel[2]->crd.data();
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    // float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::vector<int> row_crd(sparT->vLevel[1]->crd);
    std::vector<int> col_crd(sparT->vLevel[2]->crd);
    std::vector<float> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[1]->same_path.push_back(0);
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->vLevel[2]->same_path.push_back(0);
    sparT->valueArray.clear();
    // for (unsigned i = 0; i < nnz; i++) {
    //   std::cout << "row = " << row_crd[i] << "col = "<< col_crd[i] << std::endl;
    // }
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    // std::cout << "row_size = " << row_size << ", col_size = " << col_size << ", nnz = " << nnz << std::endl;
    assert(col_size >= row_size);
    // bool *root_same_path = sparT->vLevel[0]->same_path.data();
    // bool *row_same_path = sparT->vLevel[1]->same_path.data();
    // bool *col_same_path = sparT->vLevel[2]->same_path.data();
    // std::cout << "root_same_path size= " << sparT->vLevel[0]->same_path.size() <<  std::endl;
    // std::cout << "row_same_path size= " << sparT->vLevel[1]->same_path.size() <<  std::endl;
    // std::cout << "col_same_path size= " << sparT->vLevel[2]->same_path.size() <<  std::endl;
    // std::cout << "root_ptr size= " << sparT->vLevel[0]->ptr.size() <<  std::endl;
    // std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;

    int** diag_nnz = new int *[((row_size-1)/blockSize)+1];
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
      diag_nnz[i] = new int[row_size+col_size-1];
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
      for (unsigned j = 0; j < row_size+col_size-1; j++)
        diag_nnz[i][j] = 0;
    for(uint64_t i = 0; i < nnz; i++) {
      // if (values[i] == 0) 
      //   continue;
      int new_dim0 = row_crd[i]/blockSize;
      int new_dim1 = col_crd[i]-row_crd[i];
      diag_nnz[new_dim0][new_dim1+col_size-1] += 1;
    }
    // std::cout << "diag_nnz:" << std::endl;
    // for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++) {
    //   for (unsigned j = 0; j < row_size+col_size-1; j++)
    //    std::cout <<  diag_nnz[i][j] << "  ";
    //   std::cout << std::endl;
    // }
    // split the matrix
    // step 1: initialize vectorArray
    auto T_BDIA = new SparlayStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.push_back(0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);
    // SparlayStorage* T_COO = new SparlayStorage;
    // T_COO->initCOO(row_size,col_size);
    

    int row_diag_count = 0;
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      for (unsigned j = 0; j < row_size+col_size-1; j++) {
        if (diag_nnz[i][j] > blockSize*thres) {
          row_diag_count++;
          T_BDIA->vLevel[2]->crd.push_back(j-col_size+1);
          // std::vector<float> new_vec(blockSize, 0.0);
          // T_BDIA->vectorArray.push_back(new_vec);
          for (int k = 0; k < blockSize; k++)
            T_BDIA->vector_1d.push_back(0);
        }
      }
      T_BDIA->vLevel[1]->ptr.push_back(row_diag_count);
    }
    // std::cout << "T_BDIA->vLevel[1]->ptr = ";
    // for (auto elm: T_BDIA->vLevel[1]->ptr)
    //   std::cout << elm << "  ";
    // std::cout << std::endl;
    // std::cout << "T_BDIA->vLevel[2]->crd = ";
    // for (auto elm: T_BDIA->vLevel[2]->crd)
    //   std::cout << elm << "  ";
    // std::cout << std::endl;
    // std::cout << "T_BDIA->vectorArray.size = " << T_BDIA->vectorArray.size() << std::endl;

    //step 2: distribute values
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    // std::vector<int> punch_pos;
    int dia_nnz_count = 0;
    for(unsigned i = 0; i < nnz; i++) {
      // if (values[i] == 0) 
      //   continue;
      int new_dim1 = row_crd[i]/blockSize;
      int new_dim2 = col_crd[i]-row_crd[i];
      int new_dim3 = row_crd[i]%blockSize;
      if (diag_nnz[new_dim1][new_dim2+col_size-1] > blockSize*thres) {
        // if (row_crd[i] == 0)
        //   std::cout << "col = "<< col_crd[i] << ", values =" << values[i] << std::endl;
        // BDIA
        int diag_block;
        for (diag_block = dim1_ptr[new_dim1]; diag_block < dim1_ptr[new_dim1+1]; diag_block++) 
          if (dim2_crd[diag_block] == new_dim2)
            break;
        // T_BDIA->vectorArray[diag_block][new_dim3] = values[i];
        T_BDIA->vector_1d[diag_block*blockSize+new_dim3] = values[i];
        dia_nnz_count++;
        // punch_pos.push_back(i);
      } 
      else {
        if (sparT->valueArray.size() > 0) {
          sparT->vLevel[1]->same_path.push_back(row_crd[i] == sparT->vLevel[1]->crd.back());
          sparT->vLevel[2]->same_path.push_back(
              (row_crd[i] == sparT->vLevel[1]->crd.back()) && (col_crd[i] == sparT->vLevel[2]->crd.back()));
        }
        sparT->vLevel[1]->crd.push_back(row_crd[i]);
        sparT->vLevel[2]->crd.push_back(col_crd[i]);
        sparT->valueArray.push_back(values[i]);
        // T_COO->vLevel[1]->crd.push_back(row_crd[i]);
        // T_COO->vLevel[1]->same_path.push_back(row_crd[i]== T_COO->vLevel[1]->crd.back());
        // T_COO->vLevel[2]->crd.push_back(col_crd[i]);
        // T_COO->valueArray.push_back(values[i]);
      }
    }
    // for (auto pos: punch_pos) {
    //     sparT->valueArray[pos]=0;
    // }
    sparT->vLevel[0]->ptr.push_back(sparT->vLevel[1]->crd.size());
    
    // std::cout << "row_same_path size= " << sparT->vLevel[1]->same_path.size() <<  std::endl;
    // std::cout << "col_same_path size= " << sparT->vLevel[2]->same_path.size() <<  std::endl;
    // std::cout << "root_ptr size= " << sparT->vLevel[0]->ptr.size() <<  std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << dia_nnz_count <<  std::endl;
    // std::cout << "T_BDIA->vectorArray = " << std::endl;
    // for (auto i=dim1_ptr[0]; i < dim1_ptr[1]; i++) {
    //     std::cout << "diag=" << dim2_crd[i] << ", "<< T_BDIA->vectorArray[i][0] << "  "<<std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "T_COO->valueArray = " << std::endl;
    // for (unsigned x = 0; x < T_COO->vLevel[1]->crd.size(); x++) {
    //   if (T_COO->vLevel[1]->crd[x]==0)
    //     std::cout <<T_COO->vLevel[1]->crd[x]<<", "<<T_COO->vLevel[2]->crd[x]<<", " <<T_COO->valueArray[x] << "  "<<std::endl;
    //   else 
    //     break;
    // }
    // std::cout << std::endl;
    for (unsigned i = 0; i < ((row_size-1)/blockSize)+1; i++)
        free(diag_nnz[i]);
    free(diag_nnz);

    // T_COO->finalizeCOO();
    SparlayStruct* ret = new SparlayStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }

  void* _mlir_ciface_decompose_BDIA_opt(void* ptr, int32_t blockSize, float thres) {
    SparlayStorage* sparT = (SparlayStorage*)ptr;
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::vector<int> row_crd(sparT->vLevel[1]->crd);
    std::vector<int> col_crd(sparT->vLevel[2]->crd);
    std::vector<float> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->valueArray.clear();
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    assert(col_size >= row_size);
    
    // step 1: initialize vectorArray
    auto T_BDIA = new SparlayStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.push_back(0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);

    // assume read-in data is in row-major order
    double start = omp_get_wtime();

    uint64_t diag_block_count = 0;
    uint64_t diag_nnz_count = 0;
    std::vector<unsigned> row_ptr;
    std::vector<unsigned> dia_row_ptr;
    row_ptr.push_back(0);
    dia_row_ptr.push_back(0);
    int* diag_nnz = new int[blockSize+col_size-1];
    for(unsigned i = 0; i < blockSize+col_size-1; i++) 
      diag_nnz[i] = 0;
    int prev_row_block = row_crd[0]/blockSize;
    for(uint64_t i = 0; i < nnz; i++) {
      int new_dim1 = row_crd[i]/blockSize;
      int new_dim2 = col_crd[i]-row_crd[i];
      if (new_dim1 == prev_row_block) {
        diag_nnz[new_dim2+(new_dim1+1)*blockSize-1] += 1;
      } else {
        for (uint64_t j = 0; j < blockSize+col_size-1; j++) {
          if (diag_nnz[j]> blockSize*thres) {
            diag_block_count++;
            diag_nnz_count += diag_nnz[j];
            int64_t offset=j-(prev_row_block+1)*blockSize+1;
            // std::cout <<"row="<<prev_row_block<<", j="<<j<< ", diag="<< offset
            // <<", diag_nnz[j]="<<diag_nnz[j]<< ", diag_nnz_count=" << diag_nnz_count <<std::endl;
            T_BDIA->vLevel[2]->crd.push_back(offset);
            for (int k = 0; k < blockSize; k++)
              T_BDIA->vector_1d.push_back(0);
          }
          diag_nnz[j] = 0;
        }
        for (int m = prev_row_block; m <new_dim1; m++) {
          T_BDIA->vLevel[1]->ptr.push_back(diag_block_count);
          row_ptr.push_back(i);
          dia_row_ptr.push_back(diag_nnz_count);
        }
        
        prev_row_block = new_dim1;
        diag_nnz[new_dim2+(new_dim1+1)*blockSize-1] += 1;
      }
    }
    for (uint64_t j = 0; j < blockSize+col_size-1; j++) {
      if (diag_nnz[j]> blockSize*thres) {
        diag_block_count++;
        diag_nnz_count += diag_nnz[j];
        int64_t offset=j-(prev_row_block+1)*blockSize+1;
        T_BDIA->vLevel[2]->crd.push_back(offset);
        for (int k = 0; k < blockSize; k++)
          T_BDIA->vector_1d.push_back(0);
      }
    }
    for (unsigned m = prev_row_block; m <((row_size-1)/blockSize)+1; m++) {
      T_BDIA->vLevel[1]->ptr.push_back(diag_block_count);
      row_ptr.push_back(nnz);
      dia_row_ptr.push_back(diag_nnz_count);
    }
    
    delete []diag_nnz;

    // parallelize
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    sparT->vLevel[1]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path[0]=0;
    sparT->vLevel[2]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path[0]=0;
    sparT->valueArray.resize(nnz-diag_nnz_count);
    sparT->vLevel[0]->ptr.push_back(nnz-diag_nnz_count);
    unsigned i, pos;
    int iter2_dim1, iter2_dim2, iter2_dim3, start_pos, end_pos, insert_pos;
    unsigned COO_pos;
    bool is_BDIA;
    double end_1 = omp_get_wtime();
    for (unsigned time = 0; time < 100; time++) {
    #pragma omp parallel for private(i, pos,iter2_dim1, iter2_dim2, \
          iter2_dim3, start_pos, end_pos, insert_pos, COO_pos, is_BDIA)
    for (i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      COO_pos=row_ptr[i]-dia_row_ptr[i];
      for(pos = row_ptr[i]; pos < row_ptr[i+1]; pos++) {
        iter2_dim1 = row_crd[pos]/blockSize;
        iter2_dim2 = col_crd[pos]-row_crd[pos];
        iter2_dim3 = row_crd[pos]%blockSize;
        start_pos = dim1_ptr[iter2_dim1];
        end_pos = dim1_ptr[iter2_dim1+1];
        // std::cout<<"start_pos="<<start_pos<<", end_pos="<<end_pos<<std::endl;
        is_BDIA=false;
        for (insert_pos = start_pos; 
            insert_pos < end_pos; insert_pos++) {
          if (iter2_dim2 == dim2_crd[insert_pos]) {
            is_BDIA = true;
            break;
          }
        }
        if (is_BDIA) {
          T_BDIA->vector_1d[insert_pos*blockSize+iter2_dim3] = values[pos];
        } else {
          // COO_pos = COO_start_pos+COO_pos_iter;
          if (COO_pos > 0) {
            sparT->vLevel[1]->same_path[COO_pos]=(row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]);
            sparT->vLevel[2]->same_path[COO_pos]=(
                (row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]) && (col_crd[pos] == sparT->vLevel[2]->crd[COO_pos-1]));
          }
          sparT->vLevel[1]->crd[COO_pos]=(row_crd[pos]);
          sparT->vLevel[2]->crd[COO_pos]=(col_crd[pos]);
          sparT->valueArray[COO_pos]=(values[pos]);
          // std::cout<<"COO_pos="<<COO_pos<<", crd[1] size="<<sparT->vLevel[1]->crd.size()
          // <<", crd="<<sparT->vLevel[1]->crd[COO_pos]<<", value="<<sparT->valueArray[COO_pos]<< std::endl;
          COO_pos++;
        }
      }
    }
    }
    double end = omp_get_wtime();
    std::cout << "decompose before omp time = " << end_1-start << " s"<< std::endl;
    std::cout << "decompose omp time = " << end-end_1 << " s"<< std::endl;
    std::cout << "decompose total time = " << (end-end_1)/100+end_1-start << " s"<< std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << diag_nnz_count <<  std::endl;
    // std::cout<< "row_ptr: ";
    // for(auto elm : row_ptr)
    //   std::cout<<elm<< "  ";
    // std::cout<<std::endl;

    SparlayStruct* ret = new SparlayStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }

  void* _mlir_ciface_decompose_BDIA_opt2(void* ptr, int32_t blockSize, float thres) {
    SparlayStorage* sparT = (SparlayStorage*)ptr;
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    uint64_t nnz = sparT->vLevel[2]->crd.size();
    std::vector<int> row_crd(sparT->vLevel[1]->crd);
    std::vector<int> col_crd(sparT->vLevel[2]->crd);
    std::vector<DataType> values(sparT->valueArray);
    sparT->vLevel[0]->ptr.pop_back();
    sparT->vLevel[1]->crd.clear();
    sparT->vLevel[1]->same_path.clear();
    sparT->vLevel[2]->crd.clear();
    sparT->vLevel[2]->same_path.clear();
    sparT->valueArray.clear();
    std::cout << "blockSize = " << blockSize << ", thres = " << thres << std::endl;
    assert(col_size >= row_size);
    
    // step 1: initialize vectorArray
    auto T_BDIA = new SparlayStorage();
    for (unsigned i = 0; i <= 3; i++) 
      T_BDIA->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    T_BDIA->vLevel[1]->type = LVFUSE ;
    T_BDIA->vLevel[1]->ptr.resize(((row_size-1)/blockSize)+2, 0);
    T_BDIA->vLevel[2]->type = LVTRIM ;
    T_BDIA->vLevel[3]->size = blockSize;
    T_BDIA->dimSizes.push_back(row_size);
    T_BDIA->dimSizes.push_back(col_size);
    
    // assume read-in data is in row-major order
    double start = omp_get_wtime();

    std::vector<int> diag_block_count(((row_size-1)/blockSize)+1, 0);
    // std::cout<<"diag_block_count size="<<diag_block_count.size()<<std::endl;
    uint64_t diag_nnz_count = 0;
    double mem_0 = omp_get_wtime();
    std::vector<unsigned> row_ptr(((row_size-1)/blockSize)+2, 0);
    // row_ptr.push_back(0);
    double mem_1 = omp_get_wtime();
    std::vector<unsigned> dia_row_ptr(((row_size-1)/blockSize)+2, 0);
    double mem_2 = omp_get_wtime();
    
    double mem_3 = omp_get_wtime();
    // std::vector<int> diag_nnz((((row_size-1)/blockSize)+1)*(blockSize+col_size-1), 0);
    std::vector<std::vector<int>> diag_off(((row_size-1)/blockSize)+1);

    double mem_4 = omp_get_wtime();

    int first_dim1 = row_crd[0]/blockSize;
    for (int m = 0; m < first_dim1; m++)
      row_ptr[m+1]=0;
    int prev_dim1, new_dim1, init_j;
    unsigned init_i;
    double end_0 = omp_get_wtime();
    #pragma omp parallel for private(prev_dim1, new_dim1, init_j, init_i)
    for(init_i = 1; init_i < nnz; init_i++) {
      prev_dim1 = row_crd[init_i-1]/blockSize;
      new_dim1 = row_crd[init_i]/blockSize;
      if (new_dim1 != prev_dim1) {
        for (init_j = prev_dim1; init_j < new_dim1; init_j++)
          row_ptr[init_j+1]=init_i;
      } 
    }
    // std::cout << "new_dim1 = "<<new_dim1<<std::endl;
    for (unsigned m = row_crd[nnz-1]/blockSize; m < ((row_size-1)/blockSize)+1; m++)
      row_ptr[m+1] = nnz;
    assert(row_ptr.size() == ((row_size-1)/blockSize)+2);
    // std::sort(row_ptr.begin(), row_ptr.end());
    // std::cout << "row_ptr = ";
    // for (unsigned m = 0; m < ((row_size-1)/blockSize)+2; m++)
    //   std::cout << row_ptr[m] << "  ";
    // std::cout << std::endl;

    //parallelize
    double end_1 = omp_get_wtime();
    unsigned iter1_i, iter1_pos, iter1_j;
    int iter1_dim2;
    std::vector<int> diag_nnz;
    for(unsigned time = 0; time < 1000; time++) {
    #pragma omp parallel for private(diag_nnz, iter1_i, iter1_pos, iter1_j, iter1_dim2)
    for (iter1_i = 0; iter1_i < ((row_size-1)/blockSize)+1; iter1_i++) {
      diag_nnz.clear();
      diag_nnz.resize(blockSize+col_size-1, 0);
      diag_block_count[iter1_i] = 0;
      dia_row_ptr[iter1_i+1] = 0;
      std::vector<int>().swap(diag_off[iter1_i]);
      for(iter1_pos = row_ptr[iter1_i]; iter1_pos < row_ptr[iter1_i+1]; iter1_pos++) {
        // iter1_dim1 = row_crd[iter1_pos]/blockSize;
        iter1_dim2 = col_crd[iter1_pos]-row_crd[iter1_pos];
        // iter1_dim3 = row_crd[iter1_pos]%blockSize;
        diag_nnz[iter1_dim2+(iter1_i+1)*blockSize-1] += 1;
      }
      // std::cout<<"row block="<<iter1_i<<", diag_nnz=";
      // for (auto elm: diag_nnz)
      //   std::cout<<elm<<"  ";
      // std::cout<<std::endl;
      for (iter1_j = 0; iter1_j < blockSize+col_size-1; iter1_j++) {
        if (diag_nnz[iter1_j] > blockSize*thres) {
          diag_block_count[iter1_i] += 1;
          dia_row_ptr[iter1_i+1] += diag_nnz[iter1_j];
          int offset = (int)iter1_j-(iter1_i+1)*blockSize+1;
          diag_off[iter1_i].push_back(offset);
          // if (iter1_i==0)
            // std::cout<<"dia_row_ptr["<<iter1_i+1<<  "] = "<<dia_row_ptr[iter1_i+1]<<std::endl;
          // diag_nnz_count += diag_nnz[iter1_i][iter1_j];
        }
      }
    }
    }

    double end_2 = omp_get_wtime();
    

    int total_dia_block = 0;
    // std::cout<<"((row_size-1)/blockSize)+1="<<((row_size-1)/blockSize)+1<<std::endl;
    for (unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
      dia_row_ptr[init+1] += dia_row_ptr[init];
      total_dia_block += diag_block_count[init];
      T_BDIA->vLevel[1]->ptr[init+1] = total_dia_block;
      // std::cout<<"init="<<init<<", dia_row_ptr[init+1]="<<dia_row_ptr[init+1]<<", diag_block_count[init]="<<diag_block_count[init]<<std::endl;
      // std::cout << "T_BDIA->vLevel[1]->ptr["<<init+1<<"]="<<T_BDIA->vLevel[1]->ptr[init+1]<<std::endl;
      
    }
      // std::cout << "T_BDIA->vLevel[1]->ptr.size()="<<T_BDIA->vLevel[1]->ptr.size()<<std::endl;
      
    diag_nnz_count = dia_row_ptr[((row_size-1)/blockSize)+1];
    T_BDIA->vLevel[2]->crd.reserve(total_dia_block);
    for (unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
      for (auto elm: diag_off[init])
        T_BDIA->vLevel[2]->crd.push_back(elm);
      std::vector<int>().swap(diag_off[init]);
    }
    std::vector<std::vector<int>>().swap(diag_off);
    std::vector<int>().swap(diag_block_count);
    
    
    assert(T_BDIA->vLevel[2]->crd.size()==(unsigned)total_dia_block);
    T_BDIA->vector_1d.resize(total_dia_block*blockSize, 0.0);

    // unsigned iter3_i, iter3_j;
    // #pragma omp parallel for private(iter3_i, iter3_j)
    // for (iter3_i = 0; iter3_i < ((row_size-1)/blockSize)+1; iter3_i++) {
    //   for (iter3_j = 0; iter3_j < blockSize+col_size-1; iter3_j++) {
    //     if (diag_nnz[iter3_i*(blockSize+col_size-1) + iter3_j] > blockSize*thres) {
    //       int64_t offset = iter3_j-(iter3_i+1)*blockSize+1;
    //       unsigned pos = T_BDIA->vLevel[1]->ptr[iter3_i];
    //       T_BDIA->vLevel[2]->crd[pos] = offset;
    //     }
    //   }
    // }

    // parallelize
    int* dim1_ptr = T_BDIA->vLevel[1]->ptr.data();
    int* dim2_crd = T_BDIA->vLevel[2]->crd.data();
    // std::cout << "T_BDIA->vLevel[2]->crd:"<<std::endl;
    // // for(unsigned init = 0; init < ((row_size-1)/blockSize)+1; init++) {
    //   for (auto elm: T_BDIA->vLevel[2]->crd)
    //     std::cout << elm <<"  ";
    //   std::cout<<std::endl;
    // // }
    // std::cout << "nnz-diag_nnz_count: "<<nnz-diag_nnz_count<< ", sparT->vLevel[1]->crd.maxsize="<<
    // sparT->vLevel[1]->crd.max_size()<<std::endl;

    sparT->vLevel[1]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[1]->same_path[0]=0;
    sparT->vLevel[2]->crd.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path.resize(nnz-diag_nnz_count);
    sparT->vLevel[2]->same_path[0]=0;
    sparT->valueArray.resize(nnz-diag_nnz_count);
    sparT->vLevel[0]->ptr.push_back(nnz-diag_nnz_count);
    unsigned i, pos;
    int iter2_dim1, iter2_dim2, iter2_dim3, start_pos, end_pos, insert_pos;
    unsigned COO_pos;
    bool is_BDIA;
    double end_3 = omp_get_wtime();
    for (unsigned time = 0; time < 1000; time++) {
    #pragma omp parallel for private(i, pos,iter2_dim1, iter2_dim2, \
          iter2_dim3, start_pos, end_pos, insert_pos, COO_pos, is_BDIA)
    for (i = 0; i < ((row_size-1)/blockSize)+1; i++) {
      COO_pos=row_ptr[i]-dia_row_ptr[i];
      for(pos = row_ptr[i]; pos < row_ptr[i+1]; pos++) {
        iter2_dim1 = row_crd[pos]/blockSize;
        iter2_dim2 = col_crd[pos]-row_crd[pos];
        iter2_dim3 = row_crd[pos]%blockSize;
        start_pos = dim1_ptr[iter2_dim1];
        end_pos = dim1_ptr[iter2_dim1+1];
        // std::cout<<"start_pos="<<start_pos<<", end_pos="<<end_pos<<std::endl;
        is_BDIA=false;
        for (insert_pos = start_pos; 
            insert_pos < end_pos; insert_pos++) {
          if (iter2_dim2 == dim2_crd[insert_pos]) {
            is_BDIA = true;
            break;
          }
        }
        if (is_BDIA) {
          T_BDIA->vector_1d[insert_pos*blockSize+iter2_dim3] = values[pos];
          // std::cout<<"T_BDIA->vector_1d["<<insert_pos*blockSize+iter2_dim3<<"] = "<<values[pos]<<std::endl;
        } else {
          // COO_pos = COO_start_pos+COO_pos_iter;
          if (COO_pos > 0) {
            sparT->vLevel[1]->same_path[COO_pos]=(row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]);
            sparT->vLevel[2]->same_path[COO_pos]=(
                (row_crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]) && (col_crd[pos] == sparT->vLevel[2]->crd[COO_pos-1]));
          }
          sparT->vLevel[1]->crd[COO_pos]=(row_crd[pos]);
          sparT->vLevel[2]->crd[COO_pos]=(col_crd[pos]);
          sparT->valueArray[COO_pos]=(values[pos]);
          // std::cout<<"COO_pos="<<COO_pos<<", crd[1] size="<<sparT->vLevel[1]->crd.size()
          // <<", crd="<<sparT->vLevel[1]->crd[COO_pos]<<", value="<<sparT->valueArray[COO_pos]<< std::endl;
          COO_pos++;
        }
      }
    }
    }
    // std::cout << "sparT->vLevel[1]->crd.size() = " << sparT->vLevel[1]->crd.size() <<  std::endl;
    // std::cout << "sparT->vLevel[0]->ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    // std::cout << "sparT->vLevel[1]->same_path: ";
    // for (auto elm: sparT->vLevel[1]->same_path)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[2]->same_path: ";
    // for (auto elm: sparT->vLevel[2]->same_path)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[1]->crd: ";
    // for (auto elm: sparT->vLevel[1]->crd)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->vLevel[2]->crd: ";
    // for (auto elm: sparT->vLevel[2]->crd)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    // std::cout << "sparT->valueArray: ";
    // for (auto elm: sparT->valueArray)
    //   std::cout<<elm<<"  ";
    // std::cout<<std::endl;
    double end = omp_get_wtime();
    std::cout << "mem_0-start time = " << mem_0-start << " s"<< std::endl;
    std::cout << "mem_1-mem_0 time = " << mem_1-mem_0 << " s"<< std::endl;
    std::cout << "mem_2-mem_1 time = " << mem_2-mem_1 << " s"<< std::endl;
    std::cout << "mem_3-mem_2 time = " << mem_3-mem_2 << " s"<< std::endl;
    std::cout << "mem_4-mem_3 time = " << mem_4-mem_3 << " s"<< std::endl;
    std::cout << "end_0-mem_4 time = " << end_0-mem_4 << " s"<< std::endl;
    std::cout << "decompose end_0 - start time = " << end_0 - start << " s"<< std::endl;
    std::cout << "decompose end_1 - end_0 time = " << end_1 - end_0 << " s"<< std::endl;
    std::cout << "decompose end_2 - end_1 time = " << end_2 - end_1 << " s"<< std::endl;
    std::cout << "decompose end_3 - end_2 time = " << end_3 - end_2 << " s"<< std::endl;
    std::cout << "decompose end - end_3 time = " << end - end_3 << " s"<< std::endl;
    std::cout << "decompose end_3 - start time = " << end_3-start << " s"<< std::endl;
    std::cout << "decompose total time = " << (end-end_3)/1000+end_3-end_2 + (end_2-end_1)/1000+(end_1-start)<< " s"<< std::endl;
    std::cout << "root_ptr[1] = " << sparT->vLevel[0]->ptr[1] <<  std::endl;
    std::cout << "diag_nnz_count = " << diag_nnz_count <<  std::endl;

    SparlayStruct* ret = new SparlayStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BDIA);
    return (void*) ret;
  }
  

  void release(void *tensor) {
    delete static_cast<SparlayStruct *>(tensor);
  }

} // extern C

