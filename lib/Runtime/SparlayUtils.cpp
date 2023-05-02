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
#define DEBUG
//#define PRINT
//#define PARALLEL

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "Eigen/Dense"
#include "CVector.hpp"
#define DataType float
#define THREAD_NUM 48

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

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

struct Pack1c1v {
    int crd1;
    DataType vals;
};
struct Pack2c {
  int crd1;
  int crd2;
};

struct Pack2c1v {
  int crd1;
  int crd2;
  DataType vals;
};

/*!
 * \brief assume that Level 0 is root
 */
class SparlayStorage {
public:

  std::vector<uint64_t> dimSizes;
  std::vector< std::shared_ptr<LevelStorage> > vLevel;
  std::vector< std::shared_ptr<Vector2i> > exprs;
  std::vector<DataType> valueArray;
  std::vector< std::vector<float> > vectorArray;
  std::vector<DataType*> vectorPointer;
  std::vector<DataType> vector_1d;
  std::vector<int> query;
  std::vector<int> sum;
  std::vector<Pack2c> pack_vector;
  int singleVectorSize;

  #define LVINFO 4
  #define LVTRIM 1
  #define LVFUSE 2
  #define LVINDT 8

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
  bool enumerate(const int start_lv, const int end_lv);
  bool sums(const int level);
  bool reorder(const int dst_lv, const int slice_lv);
  bool schedule(const int dst_lv, const int slice_lv, const int partitions);
  bool swap(const int LU, const int LD);
  bool add(const int Ltarget, const int Lsrc);
  bool sub(const int Ltarget, const int Lsrc);
  bool vectorize(const int lv);
  bool devectorize(const int level);
  bool pad(const int start_lv, const int end_lv);
  bool neg(const int level);
  bool moveLv(const int srcLevel, const int targetLevel);
  bool tile_merge(const int lv, const int factor);
  bool tile_split(const int lv, const int factor);

  bool pack(const int start_lv, const int end_lv);
  bool partition(const int level);

  bool cust_pad(const int start_lv, const int end_lv);
  bool cust_pad_opt(const int start_lv, const int end_lv);

  void getSize(size_t lv) {
    assert(lv < exprs.size());
    // assert(dimSizes.size() == 3);
//    std::cerr << "dimSizes[0] is " << dimSizes[0] << ", and dimSizes[1] is " << dimSizes[1] << std::endl; 
    Vector2i t0(0,0), t1(0,dimSizes[1]), t2(dimSizes[0],0), t3(dimSizes[0],dimSizes[1]);
    const auto& expr = exprs[lv];
//    std::cerr << "midstage expr[" << lv << "] is " << *exprs[lv] << std::endl;
    int mn = std::min(0, std::min(expr->dot(t1), std::min(expr->dot(t2), expr->dot(t3))));
    int mx = std::max(0, std::max(expr->dot(t1), std::max(expr->dot(t2), expr->dot(t3))));
//    std::cerr << "max is " << mx << ", and mn is " << mn << std::endl;
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
    auto expLS = exprs[srcLv];
    for (int i = srcLv; i >= dstLv+1; --i) {
  //    std::cerr << i << std::endl;
      vLevel[i] = vLevel[i-1];
      exprs[i] = exprs[i-1];
    }
    vLevel[dstLv] = svLS;
    exprs[dstLv] = expLS;
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
    } else if (vectorPointer.size()) {
      static std::vector< DataType* > new_vector_array;
      assert(perm.size() == vectorPointer.size());
      new_vector_array.clear();
      new_vector_array.resize(perm.size(), {});
      for (size_t i = 0; i < vectorPointer.size(); ++i) {
        new_vector_array[i] = std::move(vectorPointer[perm[i]]);
      }
      vectorPointer = std::move(new_vector_array);
    } else {
      std::cerr << "where is the value?" << std::endl;
      assert(0);
    }
  }

  void clearVector() {
    vectorPointer.clear();
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
    fout << "crd[" << i << "]: ";
    for (const auto ele: vLevel[i]->crd) {
      fout << std::setw(8) << ele;
    }
    fout << "      (Type:" << vLevel[i]->type << ")";
    fout << " [Size:" << vLevel[i]->size << "]";
    fout << std::endl;
    fout << "expr[" << i << "]: ";
    fout << *exprs[i] << std::endl; 
    if (verbose && vLevel[i]->same_path.size()) {
//      assert(vLevel[i]->same_path.size() == vLevel[i]->crd.size());
      fout << "smp[" << i << "]: ";
      for (const auto ele: vLevel[i]->same_path) {
        fout << std::setw(8) << ele;
      }
      fout << std::endl;
    }
    if (vLevel[i]->ptr.size()) {
      fout << "ptr[" << i << "]: ";
      for (const auto ele: vLevel[i]->ptr) {
        fout << std::setw(8) << ele;
      }
      fout << std::endl;
    }
  }
  if (valueArray.size()) {
    fout << "values: ";
    for (size_t i = 0; i < valueArray.size(); ++i) {
      fout << std::setw(8) << valueArray[i];
    }
    fout << std::endl;
  } else if (vectorPointer.size()) {
    for (int j = 0; j < this->singleVectorSize; ++j) {
      fout << "values: ";
      for (size_t i = 0; i < vectorPointer.size(); ++i) {
        fout << std::setw(8) << vectorPointer[i][j];
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
    if((size_t)rowStore->crd[i] >= bucket.size()) {
      std::cerr << "Wrong value in " << i << " position is " << (size_t)rowStore->crd[i] << std::endl;
    }
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
//#ifdef DEBUG
//      for (size_t k = 0; k < pos.size(); ++k) std::cerr << pos[k] << ' ';
//      std::cerr << std::endl;
//#endif
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
//  std::cerr << "Enter moveLv" << std::endl;
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
//  std::cerr << "Start move storage " << std::endl;
  this->moveStorage(srcLv, dstLv);

  static std::vector< std::vector<int> > new_crd;
  static std::vector< std::vector<bool> > new_same_path;
  new_crd.resize(vLevel.size()-dstLv);
  new_same_path.resize(vLevel.size()-dstLv);

  for (size_t i = dstLv; i < vLevel.size(); ++i) {
    new_crd[i-dstLv].resize(vLevel[dstLv]->crd.size(),0);
//    std::cerr << "After resize new_crd[" << i-dstLv << "]with size of "  << vLevel[dstLv]->crd.size() << std::endl;

    new_same_path[i-dstLv].resize(vLevel[dstLv]->crd.size(),0);
//    std::cerr << "After resize new_same_path[" << i-dstLv << "] with size of "  << vLevel[dstLv]->crd.size() << std::endl;
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
//  std::cerr << "Enter tile_split" << std::endl;
  assert(!(vLevel[lv]->type & LVFUSE));
  assert(vLevel[lv]->type & LVTRIM);
  std::vector<int> new_crd;
  std::vector<bool> new_same_path;
  new_crd.resize(vLevel[lv]->crd.size(),0);
  new_same_path.resize(vLevel[lv]->crd.size(),0);

#ifdef PARALLEL
  if (new_crd.size()) {
    new_same_path[0] = 0;
    vLevel[lv]->same_path[0] = 0;
  }
  int* lv_crd = vLevel[lv]->crd.data();
  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(factor) private(new_crd, lv_crd)
  for (size_t i = 0; i < new_crd.size(); ++i) {
    new_crd[i] = lv_crd[i]%factor;
    lv_crd[i] /= factor;
  }
  #pragma omp barrier
  #pragma omp parallel for num_threads(THREAD_NUM) shared(lv, new_crd, lv_crd) private(new_same_path)
  for (size_t i = 1; i < new_crd.size(); ++i) {
    new_same_path[i] = (new_crd[i] == new_crd[i-1]);
    new_same_path[i] = new_same_path[i] && (vLevel[lv]->same_path[i]);
    vLevel[lv]->same_path[i] = (lv_crd[i] == lv_crd[i-1]);
    if (lv != 1) (vLevel[lv]->same_path[i]) = (vLevel[lv]->same_path[i]) && (vLevel[lv-1]->same_path[i]);
  }
  #pragma omp barrier
#else
  for (size_t i = 0; i < new_crd.size(); ++i) {
//    std::cerr << i << std::endl;
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
  std::cerr << "After loop" << std::endl;
#endif

  vLevel[lv]->size = ceil((float)vLevel[lv]->size/factor);
  std::vector<int> empty_ptr = {};
  std::shared_ptr<Vector2i> ptr = std::make_shared<Vector2i>(0,0);
  *ptr = *exprs[lv];
  this->newStorage(
    lv+1, 
    std::shared_ptr<LevelStorage>(new LevelStorage(LVTRIM, factor, new_crd, empty_ptr, new_same_path)), 
    ptr
  );
  return 1;
}

bool SparlayStorage::tile_merge(int lv, int factor) {
  assert(!(vLevel[lv]->type & LVFUSE));
  assert(vLevel[lv]->type & LVTRIM);
  assert(vLevel[lv+1]->type & LVTRIM);
  assert(!(vLevel[lv+1]->type&LVFUSE));
  assert(vLevel[lv+1]->size == factor);

#ifdef PARALLEL
  int* lv_crd = vLevel[lv]->crd.data();
  int* nextlv_crd = vLevel[lv+1]->crd.data();
  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(vLevel, factor, lv) private(lv_crd, nextlv_crd)
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    (lv_crd[i] *= factor) += nextlv_crd[i];
  }
  #pragma omp barrier

  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(vLevel, lv_crd)
  for (size_t i = 1; i < vLevel[lv]->crd.size(); ++i) {
    vLevel[lv]->same_path[i] = (vLevel[lv]->same_path[i] && (lv_crd[i]==lv_crd[i-1]));
  }
  #pragma omp barrier
#else
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    (vLevel[lv]->crd[i] *= factor) += vLevel[lv+1]->crd[i];
    if (i != 0) {
      vLevel[lv]->same_path[i] = (vLevel[lv]->same_path[i] && (vLevel[lv]->crd[i]==vLevel[lv]->crd[i-1]));
    }
  }
#endif

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
    if (ed_level_size > 1e9) {
      std::cerr << "The untrimmed level is too huge to be stored, or inefficient conversion sequence" << std::endl;
      assert(0);
    }
  }
  assert(!(vLevel[st_lv]->type & LVTRIM));
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
//#ifdef PRINT
//  Print(std::cerr, 1);
//  std::cerr << "ret: ";
//  for (size_t i = 0; i < ret.size(); ++i) std::cerr << std::setw(8) << ret[i];
//  std::cerr << std::endl;
//#endif
  return ret;
}

//Frist, right before this primitive, every level is b LVTRIM and LVFUSE 
//Second, right before this primitive, we think the coordinate at every level has already been sorted. 
bool SparlayStorage::enumerate(const int dst_lv, const int slice_lv) {
  for(int i = 1; i <= slice_lv; i++) {
    assert(vLevel[i]->type & LVTRIM);
    assert(!(vLevel[i]->type & LVFUSE));
  }

  std::vector<int> new_crd(vLevel[slice_lv]->crd.size());

  new_crd[0] = 0;
  int idx = 0;
  for(size_t i = 1; i < vLevel[slice_lv]->crd.size(); i++) {
    if(vLevel[slice_lv-1]->same_path[i]) {
      idx++;
    } else {
      idx = 0;
    }
    new_crd[i] = idx;
  }

  int max = 0;
  for(size_t i = 0; i < this->sum.size(); i++) {
    if(this->sum[i] > max) {
      max = this->sum[i];
    }
  }

  std::vector<bool> new_same_path = {};
  std::vector<int> empty_ptr = {};
  std::shared_ptr<Vector2i> ptr = std::make_shared<Vector2i>(0,0);
  this->newStorage(
    dst_lv, 
    std::shared_ptr<LevelStorage>(new LevelStorage((LVINDT^LVTRIM), max, new_crd, empty_ptr, new_same_path)), 
    ptr
  );
  
  return 1;
}

bool SparlayStorage::sums(const int lv) { 

  int total_sum_size = 1;
  for(int i = 1; i <= lv; i++) {
    total_sum_size *= vLevel[i]->size;
  }

  int strides[lv];
  for(int i = 1; i <= lv; i++) {
    strides[i-1] = 1;
    for(int j = i+1; j <= lv; j++){
      strides[i-1] *= vLevel[j]->size;
    }
  }

  int* sums = (int*)calloc(total_sum_size, sizeof(int));
  if(vLevel[lv]->type == (LVFUSE ^ LVINFO)) {
    for(int i = 0; i < lv; i++) {
      assert(!(vLevel[i]->type & LVTRIM));
    }
    for(int i = 0; i < total_sum_size; i++) {
      sums[i] = vLevel[lv]->ptr[i+1] - vLevel[lv]->ptr[i];
    }
  } else if((vLevel[lv]->type & LVTRIM) && !(vLevel[lv]->type & LVFUSE)) {
    for(size_t i = 0; i < vLevel[lv]->crd.size(); i++) {
      int offset = 0;
      for(int l = 1; l <= lv; l++) {
        offset += vLevel[l]->crd[i] * strides[l-1];
      }
      sums[offset]++;
    }
  } else {
    assert("The sum operator on the other level types is not supported");
  }

  this->sum = std::move(std::vector<int>(sums, sums+total_sum_size));

  return 1;
}

bool SparlayStorage::reorder(const int dst_lv, const int slice_lv) {
  assert(vLevel[slice_lv]->type & LVTRIM);
  int total_size = 1;
//  std::cerr << slice_lv << std::endl;
  for(int i = 1; i <= slice_lv; i++) {
    assert(!(vLevel[i]->type & LVFUSE));
    assert((vLevel[i]->type & LVTRIM));
    total_size *= vLevel[i]->size;
  }
  assert(total_size == this->sum.size());


  std::vector<int> new_crd(vLevel[slice_lv]->crd.size());
  std::vector<int> sorted(this->sum.size());
  std::vector<int> indices(this->sum.size());
  for (size_t i = 0; i < this->sum.size(); ++i) {
      sorted[i] = i;
  }

  std::vector<std::pair<int, int>> indexed_vec;
  for (size_t i = 0; i < sorted.size(); ++i) {
    indexed_vec.push_back(std::make_pair(sorted[i], i));
  }
  std::sort(indexed_vec.begin(), indexed_vec.end(), [this](const std::pair<int, int>& a, const std::pair<int, int>& b) {
    if(a.first == b.first) {
      return a.second < b.second;
    } 
    return this->sum[a.first] > this->sum[b.first]; 
  });
  for (size_t i = 0; i < indexed_vec.size(); ++i) {
    indices[indexed_vec[i].second] = i;
  }

  for(size_t i = 0; i < vLevel[slice_lv]->crd.size(); i++) {
    new_crd[i] = indices[vLevel[slice_lv]->crd[i]];
  }
  std::vector<bool> new_same_path(vLevel[slice_lv]->same_path.size());
  new_same_path = vLevel[slice_lv]->same_path;
  std::vector<int> empty_ptr = {};
  std::shared_ptr<Vector2i> ptr = std::make_shared<Vector2i>(0,0);
  this->newStorage(
    dst_lv, 
    std::shared_ptr<LevelStorage>(new LevelStorage((LVINDT^LVTRIM), vLevel[slice_lv]->size, new_crd, empty_ptr, new_same_path)), 
    ptr
  );

  return 1;
}

struct CoreInfo {
 int core_id;
 int num_nnz;
 int num_rows;
};

struct nnz_greater {
bool operator()(const CoreInfo& a,const CoreInfo& b) const{
    return a.num_nnz > b.num_nnz;
}
};

bool SparlayStorage::schedule(const int dst_lv, const int slice_lv, const int partitions) {
  int total_size = 1;
//  std::cerr << slice_lv << std::endl;
  for(int i = 1; i <= slice_lv; i++) {
    assert(!(vLevel[i]->type & LVFUSE));
    assert((vLevel[i]->type & LVTRIM));
    total_size *= vLevel[i]->size;
  }
//  std::cerr << "sum size " << this->sum.size() << " total_size" << total_size << std::endl;
  assert(total_size == this->sum.size());
  std::vector<int> new_crd(vLevel[slice_lv]->crd.size());

  std::vector<CoreInfo> cores_info;
  std::vector<int> core_map(this->sum.size());
  for(int i = 0; i < partitions; i++) {
    CoreInfo coreinfo;
    coreinfo.core_id = i;
    coreinfo.num_nnz = 0;
    coreinfo.num_rows = 0;
    cores_info.push_back(coreinfo);
  }
  std::make_heap(cores_info.begin(), cores_info.end(), nnz_greater());
  for(int i = 0; i < this->sum.size(); i++) {
    int cur_row_nnz = this->sum[i];
    std::pop_heap(cores_info.begin(), cores_info.end(), nnz_greater());
    CoreInfo min_core = cores_info.back();
    min_core.num_nnz = min_core.num_nnz + cur_row_nnz;
    min_core.num_rows = min_core.num_rows + 1;
    cores_info.pop_back();
    core_map[i] = min_core.core_id;
    cores_info.push_back(min_core);
    std::push_heap(cores_info.begin(), cores_info.end(), nnz_greater());
  }
  int strides[slice_lv];
  for(int i = 1; i <= slice_lv; i++) {
    strides[i-1] = 1;
    for(int j = i+1; j <= slice_lv; j++){
      strides[i-1] *= vLevel[j]->size;
    }
  }

  for(size_t i = 0; i < vLevel[slice_lv]->crd.size(); i++) {
    int offset = 0;
    for(int l = 1; l <= slice_lv; l++){
      offset += vLevel[l]->crd[i] * strides[l-1];
    }
//    std::cerr << "offset" << offset << std::endl;
    new_crd[i] = core_map[offset];
  }

  std::vector<bool> new_same_path = {};
  std::vector<int> empty_ptr = {};
  std::shared_ptr<Vector2i> ptr = std::make_shared<Vector2i>(0,0);
  this->newStorage(
    dst_lv, 
    std::shared_ptr<LevelStorage>(new LevelStorage((LVINDT^LVTRIM), partitions, new_crd, empty_ptr, new_same_path)), 
    ptr
  );

  return 1;
}

// The difference between pad and vectorize is, both coordinate and value of the 0 values is padded
// Inaddition to adding the value and coordinate of zero values, pad operation needs to trim and separate the level to formulate a singlton structure,
// this is for the following move operation. 
bool SparlayStorage::pad(const int start_lv, const int end_lv) {
//  std::cerr << "Start level is " << start_lv << " and end level is " << end_lv << std::endl;
  for(int i = start_lv; i <= end_lv; i++) {
    assert(vLevel[i]->type & LVTRIM);
    assert(!(vLevel[i]->type & LVFUSE));
  }
  for(int i = 2; i < start_lv; i++) {
    assert(vLevel[i]->type & LVFUSE);
    assert(!(vLevel[i]->type & LVTRIM));
  }
  assert(vLevel[1]->type & LVINDT);
  assert(vLevel[start_lv-1]->type & LVINFO);

  int strides_low[end_lv-start_lv+1];
  int strides_high[start_lv-2];
  std::vector<std::vector<int>> coordArray;
  coordArray.resize(start_lv-1, {});
  for(int i = start_lv; i <= end_lv; i++) {
    strides_low[i-start_lv] = 1;
    for(int j = i+1; j <= end_lv; j++){
      strides_low[i-start_lv] *= vLevel[j]->size;
    }
//    std::cerr << "strides_low[" << i-start_lv << "] is " << strides_low[i-start_lv] << std::endl;
  }

  for(int i = 2; i <= start_lv-1; i++) {
    strides_high[i-2] = 1;
    for(int j = i+1; j < start_lv; j++){
      strides_high[i-2] *= vLevel[j]->size;
    }
//    std::cerr << "strides_high[" << i-2 << "] is " << strides_high[i-2] << std::endl;
  }

  //Direct trim and separate the dense level
  for(int i = 0; i < this->query.size(); i++) {
    for(int j = vLevel[start_lv-1]->ptr[i]; j < vLevel[start_lv-1]->ptr[i+1]; j++) {
      int coordinate = i;
      for(int k = 2; k < start_lv; k++) {
        int this_coord = coordinate / strides_high[k-2];
        coordinate = coordinate - this_coord * strides_high[k-2];
//        std::cerr << "this_coord is " << this_coord << " coordinate is " << coordinate << " strides is " << strides_high[k-2] << std::endl;
        vLevel[k]->crd.push_back(this_coord);
      }
    }
  }

  //Add coordinate of padded 0 values
  for(int i = 0; i < this->query.size(); i++) {
    if(query[i] != vLevel[1]->size) {
      int remain = vLevel[1]->size - query[i];
      int offset = 0;
      int idx = vLevel[start_lv-1]->ptr[i];
      while(remain != 0) {
        int offset_comp = 0;
        if(idx < vLevel[start_lv-1]->ptr[i+1]) {
          for(int k = start_lv; k <= end_lv; k++) {
            offset_comp += vLevel[k]->crd[idx] * strides_low[k-2];
          }
          if(offset != offset_comp) {
            //pad low
            int coordinate = offset;
            for(int l = start_lv; l <= end_lv; l++) {
              int this_coord = coordinate / strides_low[l-start_lv];
              coordinate = coordinate - this_coord * strides_low[l-start_lv];
              vLevel[l]->crd.push_back(this_coord);
            }
            //pad high
            int coord_out = i;
            for(int l = 2; l < start_lv; l++) {
              int this_coord = coord_out / strides_high[l-2];
              coord_out = coord_out - this_coord * strides_high[l-2];
//              std::cerr << "this_coord is " << this_coord << " coord_out is " << coord_out << std::endl;
              vLevel[l]->crd.push_back(this_coord); 
            }
            vLevel[end_lv]->same_path.push_back(0);

            //pad value
            if(this->vectorPointer.size()) {

              DataType* new_obj = (DataType*)calloc(this->singleVectorSize, sizeof(DataType));
              vectorPointer.push_back(new_obj);
            } else if (!this->valueArray.empty()) {
              valueArray.push_back(0);
            } else {
              assert("At least one valueArray or vectorPointer should have element !");
            }

            vLevel[1]->crd.push_back(this->query[i]);
            offset++;
            remain--;
            this->query[i]++;
          } else {
            offset++;
            idx++;
          }
        } else {
          //pad low
          int coordinate = offset;
          for(int l = start_lv; l <= end_lv; l++) {
            int this_coord = coordinate / strides_low[l-start_lv];
            coordinate = coordinate - this_coord * strides_low[l-start_lv];
            vLevel[l]->crd.push_back(this_coord);
          }
          //pad high
          int coord_out = i;
          for(int l = 2; l < start_lv; l++) {
            int this_coord = coord_out / strides_high[l-2];
            coord_out = coord_out - this_coord * strides_high[l-2];
            vLevel[l]->crd.push_back(this_coord); 
          }
          vLevel[end_lv]->same_path.push_back(0);

          //pad value
          if(this->vectorPointer.size()) {

            DataType* new_obj = (DataType*)calloc(this->singleVectorSize, sizeof(DataType));
            vectorPointer.push_back(new_obj);
          } else if (!this->valueArray.empty()) {
            valueArray.push_back(0);
          } else {
            assert("At least one valueArray or vectorPointer should have element !");
          }

          vLevel[1]->crd.push_back(this->query[i]);
          offset++;
          remain--;
          this->query[i]++;
        }
      }
    }  
  }
 
  assert(vLevel[1]->crd.size() == query.size() * vLevel[1]->size);
  vLevel[1]->type ^= LVTRIM;
  for(int i = 2; i <= start_lv; i++) {
    assert(vLevel[i]->crd.size() == query.size() * vLevel[1]->size);
    vLevel[i]->type = LVTRIM;
  }
  vLevel[start_lv-1]->ptr.clear();
  vLevel[0]->ptr.resize(2);
  vLevel[0]->ptr[0] = 0;
  vLevel[0]->ptr[1] = query.size() * vLevel[1]->size;
  vLevel[0]->type = (LVINFO ^ LVFUSE);
  return 1;
}

bool SparlayStorage::grow(const int lv) {
  if (!(vLevel[lv]->type & LVTRIM)) { //already not trimmed
    return 1;
  }
  if (lv == (int)(vLevel.size()-1)) {
    std::cerr << "Error: Attempt to **Grow** the last level" << std::endl;
    assert(0);
  }
  int st_lv = 0;
  for (size_t i = 1; i < vLevel.size(); ++i) {
    if (vLevel[i]->type & LVTRIM) { st_lv = i; break; }
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
      vLevel[i]->type ^= LVTRIM;
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
  if (vLevel[lv]->type & LVTRIM) return 1; //already trimmed
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
          vLevel[cur_lv]->crd.push_back(i % cur_lv_size);
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
  if (vLevel[lv]->type & LVFUSE) return 1;
  if (vLevel[lv]->type & LVTRIM) {
    int upper_lv = lv;
    while (upper_lv!=0 && (vLevel[upper_lv-1]->type&LVTRIM) && !(vLevel[upper_lv-1]->type&LVFUSE)) upper_lv--;
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
      vLevel[lv]->type |= LVFUSE;
      return 1;
    }
    //update possibly the ptr of a fused level
    if (upper_lv != 0) {
      assert(vLevel[upper_lv-1]->type & LVFUSE); //it must be a fused level
      int cur_lv = upper_lv-1;
      if (!(vLevel[cur_lv]->type & LVINFO)) assert(vLevel[cur_lv]->ptr.size() == vLevel[cur_lv]->crd.size()+1);
      int saved_st_point = 0;
      assert(vLevel[cur_lv]->ptr[0] == 0);
      for (size_t i = 0; i < vLevel[cur_lv]->ptr.size()-1; ++i) {
        int cnt = vLevel[cur_lv]->ptr[i+1] - saved_st_point;
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
//  std::cerr << "Enter separate " << std::endl;
  if (!(vLevel[lv]->type & LVFUSE)) return 1;
  if (vLevel[lv]->type & LVTRIM) {
    int upper_lv = lv;
    while (upper_lv!=0 && (vLevel[upper_lv-1]->type&LVTRIM) && !(vLevel[upper_lv-1]->type&LVFUSE)) upper_lv--;
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
    //TODO: need to add support of separate the dense level
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

#ifdef PARALLEL 
  int* target_crd = vLevel[Ltarget]->crd.data();
  int* src_crd = vLevel[Lsrc]->crd.data();
  size_t total_size = vLevel[Lsrc]->crd.size(); 
  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(total_size) private(target_crd, src_crd) aligned(target_crd, src_crd : 32)
  for (size_t i = 0; i < total_size; ++i) {
    target_crd[i] += src_crd[i];
  }
  #pragma omp barrier
#else 
  for (size_t i = 0; i < vLevel[Lsrc]->crd.size(); ++i) {
    vLevel[Ltarget]->crd[i] += vLevel[Lsrc]->crd[i];
  }
#endif

  (*exprs[Ltarget]) += (*exprs[Lsrc]);
  getSize(Ltarget);
//  std::cerr << "Size of Level " << Ltarget << " is " << vLevel[Ltarget]->size << std::endl;
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

#ifdef PARALLRL
  int* target_crd = vLevel[Ltarget]->crd.data();
  int* src_crd = vLevel[Lsrc]->crd.data();
  size_t total_size = vLevel[Lsrc]->crd.size();
  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(vLevel, Lsrc, Ltarget) private(target_crd, src_crd) aligned(target_crd, src_crd : 32)
  for (size_t i = 0; i < total_size; ++i) {
    target_crd[i] -= src_crd[i];
  }
  #pragma omp barrier
#else
  for (size_t i = 0; i < vLevel[Lsrc]->crd.size(); ++i) {
    vLevel[Ltarget]->crd[i] -= vLevel[Lsrc]->crd[i];
  }
#endif

//  std::cerr << "Ltarget exprs[" << Ltarget << "] is " << std::endl << *exprs[Ltarget] << std::endl << "Lsrc exprs[" << Lsrc << "] is " << std::endl << *exprs[Lsrc] << std::endl;
  (*exprs[Ltarget]) -= (*exprs[Lsrc]);
  getSize(Ltarget);
//  std::cerr << "Size of Level " << Ltarget << " is " << vLevel[Ltarget]->size << std::endl;
  return 1;
}

//All the levels above lv will be converted to dense type.
//TODO all the other primitives support the vectorPointer
bool SparlayStorage::vectorize(const int lv) {
//  assert(lv == 2);
//  assert((size_t)lv == vLevel.size()-1);
//  bool mark = !(vLevel[lv-1]->type&LVFUSE);]
//  std::cerr << "mark info: " << vLevel[lv]->type << ", " << LVFUSE << ", " << !(vLevel[lv]->type&LVFUSE) << std::endl;
//  std::cerr << "Enter vectorize, vLevel.size() is " << vLevel.size() << ", and lv is " << lv << std::endl;
  int strides[vLevel.size()];
  strides[vLevel.size()-1] = 1;
  for(int i = 0; i < vLevel.size()-1; i++) {
    strides[i] = 1;
    for(int j = i+1; j <= vLevel.size()-1; j++){
      strides[i] *= vLevel[j]->size;
    } 
  }
  fuse(lv-1);
//  std::cerr << "After fuse inside vectorize " << std::endl;
//  this->Print(std::cerr, 1);
  auto& father_ptr = vLevel[lv-1]->ptr;
  auto& father_crd = vLevel[lv-1]->crd;
  int cur_lv_size = 1;
  for(int i = lv; i < vLevel.size(); i++) {
    cur_lv_size *= vLevel[i]->size;
  }
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vectorPointer.size() == 0);
    vectorPointer.resize(father_crd.size(),{});
    std::cerr << "Need to have " << father_crd.size() << " vectors" << std::endl;
//    std::cerr << "level[" << lv-1 << "] crd size is " << father_crd.size() << std::endl;
//    std::cerr << "level[" << lv-1 << "] ptr size is " << father_ptr.size() << std::endl; 
    assert(valueArray.size() == vLevel[lv]->crd.size());
    DataType* new_obj = (DataType*)calloc(cur_lv_size * father_crd.size(), sizeof(DataType));
//    static std::vector<float> V;
    for (size_t i = 0; i < father_crd.size(); ++i) {
//      std::cerr << "i = " << i << std::endl;
//      V.clear();
//      V.resize(cur_lv_size, 0);
      DataType* start = new_obj + i*cur_lv_size;
//      std::vector<DataType> V = std::move(std::vector<DataType>(start, end));
      assert(father_ptr[i+1] > prev_ptr);
      for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
        int offset = 0;
        for(int k = lv; k < vLevel.size(); k++){
          offset += vLevel[k]->crd[j] * strides[k];
        }
        start[offset] = std::move(valueArray[j]);
      }

//      vectorArray[i] = std::move(V);
      vectorPointer[i] = start;
//      std::cerr << "After move into vectorArray " << std::endl;
      int add_one = (father_ptr[i+1]>prev_ptr);
      int new_ptr = father_ptr[i] + add_one;
      prev_ptr = father_ptr[i+1];
      father_ptr[i+1] = new_ptr;
    }
    this->singleVectorSize = cur_lv_size;
    valueArray.clear();
    for(size_t i = lv; i < vLevel.size(); i++) {
      vLevel[i]->crd.clear();
      vLevel[i]->ptr.clear();
      vLevel[i]->same_path.clear();
      vLevel[i]->type = LVFUSE; 
    }
//    vLevel.pop_back();
  } else {
    std::cerr << "should not happen" << std::endl;
    assert(0);
  }
  //The father level of the vectorized level must be separated, since it is the last level next to value, which cannot be merged.
  separate(lv-1);
  
  return 1;
}

//Devectorize is the process of coverting the dense tensor slice that start from dimension level lv to COO representation of the corresponding tensor slice.
bool SparlayStorage::devectorize(const int level) {
//  std::cerr << "Enter devectorize " << std::endl;
  int lv = level - 1;
  bool mark = !(vLevel[lv]->type&LVFUSE);
  fuse(lv);
  auto& father_ptr = vLevel[lv]->ptr;
  auto& father_crd = vLevel[lv]->crd;
  int strides[vLevel.size()];
  strides[vLevel.size()-1] = 1;
  for(int i = 0; i < vLevel.size()-1; i++) {
    strides[i] = 1;
    for(int j = i+1; j <= vLevel.size()-1; j++){
      strides[i] *= vLevel[j]->size;
    } 
  }
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vLevel[lv]->crd.size() == vectorPointer.size());
//    for(int i = level; i < )
    std::vector<int> new_crd;
    std::vector<bool> new_same_path;
    static std::vector<float> new_value;
    new_value.clear();
    size_t Size = this->singleVectorSize;
    if (vectorPointer.size() != 0) {
      for (size_t i = 0; i < father_crd.size(); ++i) {
        assert(father_ptr[i+1] == prev_ptr+1);
        int cnt = 0;
        for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
          for (size_t k = 0; k < this->singleVectorSize; ++k) {
            if (vectorPointer[j][k]) {
              int remain = k;
              for(int l = level; l < vLevel.size(); l++) {
                int coord = remain / strides[l];
                remain = remain % strides[l];
                if(vLevel[l]->crd.empty()){ 
                  vLevel[l]->same_path.push_back(0);
                } else {
                  vLevel[l]->same_path.push_back(coord == vLevel[l]->crd.back());
                }
                vLevel[l]->crd.push_back(coord);
              }
              cnt++;
              new_value.push_back(vectorPointer[j][k]);
            }
          }
        }
        int new_ptr = father_ptr[i] + cnt;
        prev_ptr = father_ptr[i+1];
        father_ptr[i+1] = new_ptr;
      }
    }
    for(int i = level; i < vLevel.size(); i++) {
      vLevel[i]->type = LVTRIM; 
    }
//    std::vector<int> empty_ptr = {};
//    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage(1, Size, new_crd, empty_ptr, new_same_path)));
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
#ifdef PARALLEL
  int* lv_crd = vLevel[lv]->crd.data();
  size_t total_size = vLevel[lv]->crd.size();
  #pragma omp parallel for simd num_threads(THREAD_NUM) shared(total_size) private(lv_crd) aligned(lv_crd : 32)
  for (size_t i = 0; i < total_size; ++i) {
    lv_crd[i] = -lv_crd[i];
  }
  #pragma omp barrier
#else
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    vLevel[lv]->crd[i] = -vLevel[lv]->crd[i];
  }  
#endif
  (*exprs[lv]) = -(*exprs[lv]);
  return 1;
}

//This version we push_back 0 at the end of vector and reorder.
bool SparlayStorage::cust_pad(const int start_lv, const int end_lv) {
  for(int i = 1; i < vLevel.size(); i++) {
    assert(vLevel[i]->type & LVTRIM);
  }

  int strides[end_lv-start_lv+1];
  int subtensor_size = 1;
  for(int i = start_lv; i <= end_lv; i++) {
    strides[i-start_lv] = 1;
    for(int j = i+1; j <= end_lv; j++){
      strides[i-start_lv] *= vLevel[j]->size;
    } 
    subtensor_size *= vLevel[i]->size;
  }

  int* block = (int*)calloc(subtensor_size, sizeof(int));
  int idx = 0;
  for(int l = start_lv; l <= end_lv; l++) {
    idx += vLevel[start_lv]->crd[0] * strides[l-start_lv];
  }
  block[idx] = 1;
  int remember = 0;
  size_t before_pad_size = vLevel[start_lv]->crd.size();
  for(size_t i = 1; i < before_pad_size; i++) {
    if((start_lv == 1) || (vLevel[start_lv-1]->same_path[i])) {
      int offset = 0;
      for(int l = start_lv; l <= end_lv; l++) {
        offset += vLevel[start_lv]->crd[i] * strides[l-start_lv];
      }
      block[offset] = 1;
    } else {
      //Pad happens here
      int buffer_crd[start_lv-1];
      for(int j = 0; j < subtensor_size; j++) {
        if(block[j] != 0) {
          for(int k = 1; k < start_lv; k++) {
            buffer_crd[k-1] = vLevel[k]->crd[remember];
//            std::cerr << "remember is " << i << "buffer_crd[" << k-1 << "] is " << buffer_crd[k-1] << std::endl;
          }
          break;
        }
      }
      for(int j = 0; j < subtensor_size; j++) { 
        if(block[j] == 0) {
          int offset = j;
          for(int k = start_lv; k <= end_lv; k++) {
            int index = offset / strides[k-start_lv];
            offset = offset - index * strides[k-start_lv];
            vLevel[k]->crd.push_back(index);
            vLevel[k]->same_path.push_back(0);
          }
          for(int k = 1; k < start_lv; k++){
            vLevel[k]->crd.push_back(buffer_crd[k-1]);
            vLevel[k]->same_path.push_back(0);
          }
          for(int k =end_lv+1; k < vLevel.size(); k++) {
            vLevel[k]->crd.push_back(0);
            vLevel[k]->same_path.push_back(0);
          }
          this->valueArray.push_back(0);
        }
      }
      remember = i;
      std::memset(block, 0, subtensor_size * sizeof(int));
      int offset = 0;
      for(int l = start_lv; l <= end_lv; l++) {
        offset += vLevel[start_lv]->crd[i] * strides[l-start_lv];
      }
      block[offset] = 1;
    }
  }

  int buffer_crd[start_lv-1];
  for(int j = 0; j < subtensor_size; j++) {
    if(block[j] != 0) {
      for(int k = 1; k < start_lv; k++) {
        buffer_crd[k-1] = vLevel[k]->crd[remember];
//        std::cerr << "remember is " << remember << "buffer_crd[" << k-1 << "] is " << buffer_crd[k-1] << std::endl;
      }
      break;
    }
  }
  for(int j = 0; j < subtensor_size; j++) { 
    if(block[j] == 0) {
      int offset = j;
      for(int k = start_lv; k <= end_lv; k++) {
        int index = offset / strides[k-start_lv];
        offset = offset - index * strides[k-start_lv];
        vLevel[k]->crd.push_back(index);
        vLevel[k]->same_path.push_back(0);
      }
      for(int k = 1; k < start_lv; k++){
        vLevel[k]->crd.push_back(buffer_crd[k-1]);
        vLevel[k]->same_path.push_back(0);
      }
      for(int k =end_lv+1; k < vLevel.size(); k++) {
        vLevel[k]->crd.push_back(0);
        vLevel[k]->same_path.push_back(0);
      }
      this->valueArray.push_back(0);
    }
  }

  return 1;
}

//One shot write 
bool SparlayStorage::cust_pad_opt(const int start_lv, const int end_lv) {
  for(size_t i = 1; i <= end_lv; i++) {
    assert(vLevel[i]->type & LVTRIM);
  }

  int strides[end_lv-start_lv+1];
  int subtensor_size = 1;
  for(int i = start_lv; i <= end_lv; i++) {
    strides[i-start_lv] = 1;
    for(int j = i+1; j <= end_lv; j++){
      strides[i-start_lv] *= vLevel[j]->size;
    } 
    subtensor_size *= vLevel[i]->size;
  }
  int high_level_size = 0;
  if(start_lv != 1) {
    for(size_t i = 0; i < vLevel[start_lv-1]->same_path.size(); i++){
      if(vLevel[start_lv-1]->same_path[i] == 0) {
        high_level_size++;
      }
    }
  } else {
    high_level_size = 1;
  }
  int vector_size = 1;
  for(size_t i = vLevel.size()-1; i > end_lv; i--) {
    if(vLevel[i]->type == LVFUSE) {
      vector_size *= vLevel[i]->size;
    }
  }

  static std::vector< std::vector<int> > level_crds;
  level_crds.resize(vLevel.size()-end_lv-1);
  static std::vector<DataType*> value_pointer;
  value_pointer.resize(high_level_size * subtensor_size, nullptr);
  static std::vector<DataType> valuearray;
  valuearray.resize(high_level_size * subtensor_size, 0);

  for(size_t l = end_lv+1; l < this->vLevel.size(); l++) {
    level_crds[l-end_lv-1].resize(high_level_size*subtensor_size, 0);
  }
  int idx = 0;
  for(size_t i = 0; i < this->vLevel[start_lv]->crd.size(); i++) {
    int offset = 0;
    if(start_lv != 1 && this->vLevel[start_lv-1]->same_path[i] == 0) {
      idx++;
    }
    offset = start_lv == 1 ? 0 : (idx-1) * subtensor_size;
    for(int l = start_lv; l <= end_lv; l++) {
      offset += this->vLevel[l]->crd[i] * strides[l-start_lv];
    }

    for(int l = end_lv+1; l < this->vLevel.size(); l++) {
      level_crds[l-end_lv-1][offset] = vLevel[l]->crd[i];
    }

    if(vector_size > 1) {
      value_pointer[offset] = this->vectorPointer[i];
    } else {
      valuearray[offset] = this->valueArray[i];
    }  
  }

  if(vector_size > 1) {
    for(int i = 0; i < high_level_size * subtensor_size; i++) {
      if(value_pointer[i] == nullptr) {
        value_pointer[i] = (DataType*)calloc(vector_size, sizeof(DataType));
      }
    }
  }

  for(int l = end_lv+1; l < vLevel.size(); l++) {
    this->vLevel[l]->crd = std::move(level_crds[l-end_lv-1]);

  }
  if(vector_size > 1) {
    this->vectorPointer = std::move(value_pointer);
  } else {
    this->valueArray = std::move(valuearray);
  }

  for(int l = start_lv; l <= end_lv; l++) {
    this->vLevel[l]->type = LVFUSE;
    this->vLevel[l]->crd.clear();
    this->vLevel[l]->same_path.clear();
  }

/*
  for (const auto ele: vLevel[3]->crd) {
    std::cerr << std::setw(8) << ele;
  }
  std::cerr << "\n";
  for (const auto ele: this->valueArray) {
    std::cerr << std::setw(8) << ele;
  }
  std::cerr << "\n";
*/
  return 1;
}

//TODO: very ugly and hardcode implementation, need to support the struct with variable number of integers.
bool SparlayStorage::pack(const int start_lv, const int end_lv) {
  for(int i = start_lv; i < end_lv; i++) {
    assert(vLevel[i]->type = LVTRIM);
  }
  assert(end_lv < vLevel.size());
  assert(end_lv - start_lv == 1);
  this->pack_vector.resize(vLevel[start_lv]->crd.size());

  for(size_t i = 0; i < vLevel[start_lv]->crd.size(); i++) {
    pack_vector[i].crd1 = vLevel[start_lv]->crd[i];
    pack_vector[i].crd2 = vLevel[end_lv]->crd[i];
  }
  return 1;
}

bool SparlayStorage::partition(const int lv) {
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
    std::cerr << "Enter the check function" << std::endl;
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
      #ifdef DEBUG
      std::cerr << std::endl << "Fuse done, time = " << TI-tic << "(s)" << std::endl;
      #endif
      #ifdef PRINT
      std::cerr << "Print after Fuse" << std::endl;
      sparT->Print(std::cerr, 1);
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
      #ifdef PRINT
      std::cerr << "Print after grow" << std::endl;
      sparT->Print(std::cerr, 1);
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
      #ifdef PRINT
      std::cerr << "Print after trim" << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
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
      #ifdef PRINT
      std::cerr << "Print after separate" << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
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
      #ifdef PRINT
      std::cerr << "Print after swap" << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
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
      #ifdef PRINT
      std::cerr << "Print after sub" << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
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
      #ifdef PRINT
      std::cerr << "Print after add " << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
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
      #ifdef PRINT
      std::cerr << "Print after neg " << std::endl;
      sparT->Print(std::cerr, 1);
      #endif
      return (void*)sparT;
  }

  void* _mlir_ciface_sptEnumerate(void* ptr, int start_lv, int end_lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->enumerate(start_lv+1, end_lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Enumerate done, time = " << (TI-tic) << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after enumerate" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptPad(void* ptr, int start_lv, int end_lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->pad(start_lv+1, end_lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Pad done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after pad" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptReorder(void* ptr, int dst_lv, int slice_lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->reorder(dst_lv+1, slice_lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Reorder done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after reorder" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptSum(void* ptr, int lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->sums(lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Sum done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after sum" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptSchedule(void* ptr, int dst_lv, int slice_lv, int p) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->schedule(dst_lv+1, slice_lv+1, p);
    #ifdef DEBUG
    std::cerr << std::endl << "Schedule done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after schedule" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
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
    #ifdef PRINT
    std::cerr << "Print tile split" << std::endl;
    sparT->Print(std::cerr, 1);
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
    #ifdef PRINT
    std::cerr << "Print after Tile Merge" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptMove(void* ptr, int srcLv, int dstLv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->moveLv(srcLv+1, dstLv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Move done, time = " << (TI-tic) << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after move(" << srcLv << ", " << dstLv << ")" << std::endl;
    sparT->Print(std::cerr, 1);
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
    std::cerr << std::endl << "Vectorize done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after vectorize" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptDevectorize(void* ptr, int lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->devectorize(lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Devectorize done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after devectorize" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptPack(void* ptr, int start_lv, int end_lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->pack(start_lv+1, end_lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Pack done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after pack" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptPartition(void* ptr, int lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->partition(lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "Partition done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after partition" << std::endl;
    sparT->Print(std::cerr, 1);
    #endif
    return (void*)sparT;
  }

  void* _mlir_ciface_sptCustPad(void* ptr, int start_lv, int end_lv) {
    #ifdef DEBUG
    auto tic = TI;
    #endif
    SparlayStorage* sparT = (SparlayStorage*)(ptr);
    sparT->cust_pad_opt(start_lv+1, end_lv+1);
    #ifdef DEBUG
    std::cerr << std::endl << "CustPad done, time = " << TI-tic << "(s)" << std::endl;
    #endif
    #ifdef PRINT
    std::cerr << "Print after CustPad" << std::endl;
    sparT->Print(std::cerr, 1);
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

  #pragma omp declare simd uniform(x, y) linear(i : 1) aligned(x, y : 32) notinbranch
  void xpy(float* x, float* y, int i) {
    y[i] = x[i] + y[i];
  }

  void _mlir_ciface_kernel_hetero_bdia_spmv_iter(StridedMemRefType<DataType, 1> *outC,
                                      void* inA_CSR, 
                                      void* inA_BDIA, 
                                      StridedMemRefType<DataType, 1> *inB, 
                                      StridedMemRefType<DataType, 1> *inC) {
    
    int ib, i, k, diag, is, ie;
    SparlayStorage* spA_CSR = (SparlayStorage*)inA_CSR;
    SparlayStorage* spA_BDIA = (SparlayStorage*)inA_BDIA;
    int32_t* BDIA_dim1_ptr = spA_BDIA->vLevel[1]->ptr.data();
    int n_blocks = spA_BDIA->vLevel[1]->ptr.size();
    int32_t* BDIA_dim2_crd = spA_BDIA->vLevel[2]->crd.data();
    int32_t* CSR_dim1_ptr = spA_CSR->vLevel[1]->ptr.data();
    int32_t* CSR_dim2_crd = spA_CSR->vLevel[2]->crd.data();
    int32_t csr_nnz = spA_CSR->vLevel[2]->crd.size();
//    std::cout << "CSR NNZ is " << csr_nnz << std::endl;
    DataType* CSR_value = spA_CSR->valueArray.data();

    int blockSize = spA_BDIA->vLevel[3]->size;
    std::vector<DataType> BDIA_vector = spA_BDIA->vector_1d;
    int32_t num_rows = (int32_t)inC->sizes[0];
    int32_t num_cols = (int32_t)inB->sizes[0];
    int runs = 50;
    float alpha = 1.0;
    float beta = 0.0;
    DataType* out = inC->data;
    
    int32_t vec_block = 256;
    int32_t vec_num_blocks = std::ceil((float)num_rows / (float)vec_block);
    DataType* csr_out = (float*)std::malloc(num_rows * sizeof(DataType));

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5;
    // device malloc
    float* cu_csrVal=0;
    cudaStat1 = cudaMalloc((void**)&cu_csrVal, csr_nnz * sizeof(DataType));
    int* cu_csrRowPtr=0;
    cudaStat2 = cudaMalloc((void**)&cu_csrRowPtr, (num_rows + 1) * sizeof(int));
    int* cu_csrColInd=0;
    cudaStat3 = cudaMalloc((void**)&cu_csrColInd, csr_nnz * sizeof(int));
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }
    cudaStat1 = cudaMemcpy(cu_csrVal, CSR_value, csr_nnz * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << "CSR_value: " << CSR_value[0] << " " << CSR_value[1] << " " << CSR_value[2] << " " << CSR_value[3] << std::endl;
    cudaStat2 = cudaMemcpy(cu_csrRowPtr, CSR_dim1_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
//    std::cout << "CSR_dim1_ptr: " << CSR_dim1_ptr[0] << " " << CSR_dim1_ptr[1] << " " << CSR_dim1_ptr[2] << " " << CSR_dim1_ptr[3] << std::endl;
    cudaStat3 = cudaMemcpy(cu_csrColInd, CSR_dim2_crd, csr_nnz * sizeof(int), cudaMemcpyHostToDevice);
//    std::cout << "CSR_dim2_crd: " << CSR_dim2_crd[0] << " " << CSR_dim2_crd[1] << " " << CSR_dim2_crd[2] << " " << CSR_dim2_crd[3] << std::endl;
    if ((cudaStat1 != cudaSuccess) ||
        (cudaStat2 != cudaSuccess) ||
        (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    float* cu_InVec=0;
    cudaStat4 = cudaMalloc((void**)&cu_InVec, num_cols * sizeof(DataType));
    float* cu_OutVec=0;
    cudaStat5 = cudaMalloc((void**)&cu_OutVec, num_rows * sizeof(DataType));
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Device malloc failed");
        exit(-1);
    }
    cudaStat4 = cudaMemcpy(cu_InVec, inB->data, num_cols * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << inB->data[0] << " " << inB->data[1] << " " << inB->data[2] << " " << inB->data[3] << std::endl;
    cudaStat5 = cudaMemcpy(cu_OutVec, inC->data, num_rows * sizeof(DataType), cudaMemcpyHostToDevice);
//    std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;
    if ((cudaStat4 != cudaSuccess) || (cudaStat5 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        exit(-1);
    }

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t gpu_csr_matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix
    cusparseCreateCsr(&gpu_csr_matA, num_rows, num_cols, csr_nnz,
                      cu_csrRowPtr, cu_csrColInd, cu_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F); 
    cusparseCreateDnVec(&vecX, num_cols, cu_InVec, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_rows, cu_OutVec, CUDA_R_32F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();
    double start0 = omp_get_wtime();
    for (int i = 0; i < runs; i++) {
      cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, gpu_csr_matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_MV_ALG_DEFAULT, dBuffer);
//      if(cudaStat6 != CUSPARSE_STATUS_SUCCESS) {
//        printf("Fail compute CSR SpMV on GPU\n");
//        exit(-1);
//      }
//      cudaDeviceSynchronize();
      #pragma omp parallel for private(ib,k,diag,is,ie)
      for (ib = 0; ib < n_blocks-1; ib++) {
        for (k = BDIA_dim1_ptr[ib]; k < BDIA_dim1_ptr[ib+1]; k++) {
          diag = BDIA_dim2_crd[k];
          is = std::max(ib*blockSize, -diag);
          ie = std::min({(ib+1)*blockSize, (int)num_rows-diag, (int)num_rows});
          #pragma omp simd
          for (i = is; i < ie; i++) {
            inC->data[i] += BDIA_vector[k*blockSize+i-ib*blockSize] * inB->data[i+diag];
          }
        }
      }
      cudaDeviceSynchronize();
//      std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;
    }
    double end0 = omp_get_wtime();
    std::cout << "bdia on CPU and csr on GPU total time = " << end0-start0 << " s"<< std::endl;
    std::cout << "Heterogeneous avg time = " << (end0-start0)*1000/runs << " ms"<< std::endl;
    
    double start1 = omp_get_wtime();
    cudaStat5 = cudaMemcpy(csr_out, cu_OutVec, num_rows * sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    std::cout << csr_out[0] << " " << csr_out[1] << " " << csr_out[2] << " " << csr_out[3] << std::endl;
//    for(int i = 0; i < num_rows; i++) {
//      inC->data[i] = inC->data[i] + csr_out[i];
//    }
    #pragma omp parallel for private(ib,i) 
    for (ib = 0; ib < vec_num_blocks; ib++) {
      # pragma omp simd aligned (csr_out, out : 32)
      for (i = ib*vec_block; i < std::min((ib+1)*vec_block, (int)num_rows); i++) {
        xpy(csr_out, out, i);
      }
    }

    double end1 = omp_get_wtime();
    std::cout << "Merge time = " << end1-start1 << " s"<< std::endl;
    std::cout << "Merge avg time = " << (end1-start1)*1000/runs << " ms"<< std::endl;
    outC->data = inC->data;
    outC->basePtr = inC->basePtr;
    outC->offset = inC->offset;  
    outC->sizes[0] = inC->sizes[0];
    outC->strides[0] = inC->strides[0];
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
        double csr_time = 0.0;
        double bdia_time = 0.0;
        unsigned runs = 50;
        std::cout << inC->data[0] << " " << inC->data[1] << " " << inC->data[2] << " " << inC->data[3] << std::endl;

        double start0 = omp_get_wtime();
        for (unsigned time = 0; time < runs; time++) {
          #pragma omp parallel for private(ib,i,k,sum,diag,is,ie) 
          for (ib = 0; ib < n_blocks-1; ib++) {
            for (i = ib*blockSize; i < std::min((ib+1)*blockSize, (int)iSize); i++) {
              sum=0;
              #pragma omp simd reduction(+:sum)
              for(k=CSR_dim1_ptr[i]; k<CSR_dim1_ptr[i+1]; k++) {
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
        double end0 = omp_get_wtime();
        std::cout << "Hybrid total time = " << (end0-start0) << " s"<< std::endl;
        std::cout << "Hybrid avg time = " << (end0-start0)*1000/runs << " ms"<< std::endl;


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

  void output_header(std::ofstream& outfile, int row_size, int col_size, int nnz) {
    outfile << "%%MatrixMarket matrix coordinate integer general\n";
    outfile << "%\n";
    outfile << "% This is a test sparse matrix in Matrix Market Exchange Format.\n";
    outfile << "% see https://math.nist.gov/MatrixMarket\n";
    outfile << "%\n";
    outfile << row_size << " " << col_size << " " << nnz << "\n"; 
  }

  void* _mlir_ciface_decompose_BDIA(void* ptr, int32_t blockSize, float thres) {
    SparlayStorage* sparT = (SparlayStorage*)ptr;
    // int32_t *row_crd = sparT->vLevel[1]->crd.data();
    // int32_t *col_crd = sparT->vLevel[2]->crd.data();
    
    uint64_t row_size = sparT->dimSizes.data()[0];
    uint64_t col_size = sparT->dimSizes.data()[1];
    // float *values = sparT->valueArray.data();
    uint64_t nnz = sparT->vLevel[2]->crd.size();
//    std::vector<int> row_crd(sparT->vLevel[1]->crd);
//    std::vector<int> col_crd(sparT->vLevel[2]->crd);
//    std::vector<float> values(sparT->valueArray);
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
      int new_dim0 = sparT->vLevel[1]->crd[i]/blockSize;
      int new_dim1 = sparT->vLevel[2]->crd[i]-sparT->vLevel[1]->crd[i];
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
    std::string output_file_path = "/work/shared/users/staff/zz546/Sparse_Layout_Dialect/test/Data/output_matrix_market.mtx";
    std::ofstream outfile(output_file_path);
    output_header(outfile, row_size, col_size, nnz);
    for(unsigned i = 0; i < nnz; i++) {
      // if (values[i] == 0) 
      //   continue;
      int new_dim1 = sparT->vLevel[1]->crd[i]/blockSize;
      int new_dim2 = sparT->vLevel[2]->crd[i]-sparT->vLevel[1]->crd[i];
      int new_dim3 = sparT->vLevel[1]->crd[i]%blockSize;
      if (diag_nnz[new_dim1][new_dim2+col_size-1] > blockSize*thres) {
        outfile << sparT->vLevel[1]->crd[i]+1 << " " << sparT->vLevel[2]->crd[i]+1 << " " << std::scientific << std::setprecision(3) << sparT->valueArray[i] << "\n"; 
        // if (row_crd[i] == 0)
        //   std::cout << "col = "<< col_crd[i] << ", values =" << values[i] << std::endl;
        // BDIA
        int diag_block;
        for (diag_block = dim1_ptr[new_dim1]; diag_block < dim1_ptr[new_dim1+1]; diag_block++) 
          if (dim2_crd[diag_block] == new_dim2)
            break;
        // T_BDIA->vectorArray[diag_block][new_dim3] = values[i];
        T_BDIA->vector_1d[diag_block*blockSize+new_dim3] = sparT->valueArray[i];
        dia_nnz_count++;
        // punch_pos.push_back(i);
      } 
      else {
        if (sparT->valueArray.size() > 0) {
          sparT->vLevel[1]->same_path.push_back(sparT->vLevel[1]->crd[i] == sparT->vLevel[1]->crd.back());
          sparT->vLevel[2]->same_path.push_back(
              (sparT->vLevel[1]->crd[i] == sparT->vLevel[1]->crd.back()) && (sparT->vLevel[2]->crd[i] == sparT->vLevel[2]->crd.back()));
        }
        sparT->vLevel[1]->crd.push_back(sparT->vLevel[1]->crd[i]);
        sparT->vLevel[2]->crd.push_back(sparT->vLevel[2]->crd[i]);
        sparT->valueArray.push_back(sparT->valueArray[i]);
        // T_COO->vLevel[1]->crd.push_back(row_crd[i]);
        // T_COO->vLevel[1]->same_path.push_back(row_crd[i]== T_COO->vLevel[1]->crd.back());
        // T_COO->vLevel[2]->crd.push_back(col_crd[i]);
        // T_COO->valueArray.push_back(values[i]);
      }
    }
    outfile.seekp(0);
    output_header(outfile, row_size, col_size, dia_nnz_count);
    outfile.close();


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
//    std::vector<int> row_crd(sparT->vLevel[1]->crd);
//    std::vector<int> col_crd(sparT->vLevel[2]->crd);
//    std::vector<float> values(sparT->valueArray);
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
    int prev_row_block = sparT->vLevel[1]->crd[0]/blockSize;

    for(uint64_t i = 0; i < nnz; i++) {
      int new_dim1 = sparT->vLevel[1]->crd[i] / blockSize;
      int new_dim2 = sparT->vLevel[2]->crd[i] - sparT->vLevel[1]->crd[i];
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
        iter2_dim1 = sparT->vLevel[1]->crd[pos]/blockSize;
        iter2_dim2 = sparT->vLevel[2]->crd[pos]-sparT->vLevel[1]->crd[pos];
        iter2_dim3 = sparT->vLevel[1]->crd[pos]%blockSize;
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
          T_BDIA->vector_1d[insert_pos*blockSize+iter2_dim3] = sparT->valueArray[pos];
        } else {
          // COO_pos = COO_start_pos+COO_pos_iter;
          if (COO_pos > 0) {
            sparT->vLevel[1]->same_path[COO_pos]=(sparT->vLevel[1]->crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]);
            sparT->vLevel[2]->same_path[COO_pos]=(
                (sparT->vLevel[1]->crd[pos] == sparT->vLevel[1]->crd[COO_pos-1]) && (sparT->vLevel[2]->crd[pos] == sparT->vLevel[2]->crd[COO_pos-1]));
          }
          sparT->vLevel[1]->crd[COO_pos]=(sparT->vLevel[1]->crd[pos]);
          sparT->vLevel[2]->crd[COO_pos]=(sparT->vLevel[2]->crd[pos]);
          sparT->valueArray[COO_pos]=(sparT->valueArray[pos]);
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
    for(unsigned time = 0; time < 1; time++) {
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
    for (unsigned time = 0; time < 1; time++) {
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

  void* _mlir_ciface_decompose_BELL_COO(void* ptr, int32_t blockSize, float block_thres, float col_thres) {
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
    std::cout << "blockSize = " << blockSize << ", block_thres = " << block_thres << ", col_thres = " << col_thres << std::endl;
    
    // step 1: initialize vectorArray
    auto T_BELL = new SparlayStorage();
    for (unsigned i = 0; i <= 5; i++) 
      T_BELL->vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage));
    // T_BELL->vLevel[0]->type = LVTRIM ; // how to add FUSE in the same level?
    T_BELL->vLevel[1]->type = LVFUSE | LVTRIM ;
    T_BELL->vLevel[2]->size = ((row_size-1)/blockSize)+1;
    T_BELL->vLevel[2]->type = LVFUSE ;
    T_BELL->vLevel[3]->type = LVFUSE ;
    T_BELL->vLevel[4]->size = blockSize;
    T_BELL->vLevel[5]->size = blockSize;
    T_BELL->dimSizes.push_back(row_size);
    T_BELL->dimSizes.push_back(col_size);
    
    // step 2: assume read-in data is in row-major order
    std::vector<unsigned> row_block_ptr(((row_size-1)/blockSize)+2, 0);
    int prev_dim0, new_dim0, init_j;
    unsigned init_i;
    #pragma omp parallel for private(prev_dim0, new_dim0, init_j, init_i)
    for (init_i = 1; init_i < nnz; init_i++) {
      prev_dim0 = row_crd[init_i-1]/blockSize;
      new_dim0 = row_crd[init_i]/blockSize;
      if (new_dim0 != prev_dim0) {
        for (init_j = prev_dim0; init_j < new_dim0; init_j++)
          row_block_ptr[init_j+1]=init_i;
      } 
    }
    for (unsigned m = row_crd[nnz-1]/blockSize; m < ((row_size-1)/blockSize)+1; m++)
      row_block_ptr[m+1] = nnz;
    // std::cout << "row_block_ptr = ";
    // for (unsigned n = 0; n < row_block_ptr.size(); n++) {
    //   std::cout << row_block_ptr[n] << "  ";
    // }
    // std::cout << "\n";

    // step 2: compute level 1 crd - the max
    unsigned iter1_i, iter1_pos, iter1_j;
    int col_block_id;
    std::vector<unsigned> col_blocks(((row_size-1)/blockSize)+1, 0); 
    // std::vector<unsigned> col_block_nnz(((col_size-1)/blockSize)+1, 0);
    std::vector<std::vector<unsigned>> col_block_crd(((row_size-1)/blockSize)+1);
    #pragma omp parallel for private(iter1_i, iter1_pos, iter1_j, col_block_id)  
    for (iter1_i = 0; iter1_i < ((row_size-1)/blockSize)+1; iter1_i++) {
      std::vector<unsigned> col_block_nnz(((col_size-1)/blockSize)+1, 0);
      for (iter1_pos = row_block_ptr[iter1_i]; iter1_pos < row_block_ptr[iter1_i+1]; iter1_pos++) {
        col_block_id = col_crd[iter1_pos]/blockSize;
        col_block_nnz[col_block_id] += 1;
      }
      // std::cout << "col_block_nnz = ";
      // for (unsigned n = 0; n < col_block_nnz.size(); n++) {
      //   std::cout << col_block_nnz[n] << "  ";
      // }
      // std::cout << "\n";
      for (iter1_j = 0; iter1_j < ((col_size-1)/blockSize)+1; iter1_j++) {
        if (col_block_nnz[iter1_j] > blockSize*blockSize*block_thres) {
          col_blocks[iter1_i] += 1;
          col_block_crd[iter1_i].push_back(iter1_j);
        }
      }
    }
    unsigned max_nnz = *std::max_element(col_blocks.begin(), col_blocks.end());
    unsigned level1_size = unsigned(std::ceil(max_nnz * col_thres));
    for (unsigned i = 0; i < level1_size; i++)
      T_BELL->vLevel[1]->crd.push_back(i);
    // std::cout << "col_blocks = ";
    // for (unsigned n = 0; n < col_blocks.size(); n++) {
    //   std::cout << col_blocks[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "col_block_crd = ";
    // for (unsigned n = 0; n < col_block_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < col_block_crd[n].size(); m++) 
    //     std::cout << col_block_crd[n][m] << "  ";
    // }
    // std::cout << "\n";

    // step 3: compute level 3 crd. Only after the crd order is determined can values be dispatched
    T_BELL->vLevel[3]->crd.resize(T_BELL->vLevel[2]->size * level1_size, 0);
    unsigned iter2_i, iter2_j, pad_val;
    #pragma omp parallel for private(iter2_i, iter2_j, pad_val)  
    for (iter2_i = 0; iter2_i < ((row_size-1)/blockSize)+1; iter2_i++) {
      for (iter2_j = 0; iter2_j < std::min(level1_size, (unsigned)col_block_crd[iter2_i].size()); iter2_j++) {
        T_BELL->vLevel[3]->crd[iter2_i*level1_size + iter2_j] = col_block_crd[iter2_i][iter2_j];
      }
      pad_val = 0;
      while(iter2_j < level1_size) {
        if (std::count(col_block_crd[iter2_i].begin(), col_block_crd[iter2_i].end(), pad_val)) {
          pad_val += 1;
          continue;
        } else {
          T_BELL->vLevel[3]->crd[iter2_i*level1_size + iter2_j] = pad_val;
          pad_val += 1;
          iter2_j += 1;
        }
      }
    }
    // std::cout << "level3_crd = ";
    // for (unsigned n = 0; n < T_BELL->vLevel[3]->crd.size(); n++) {
    //   std::cout << T_BELL->vLevel[3]->crd[n] << "  ";
    // }
    // std::cout << "\n";

    // step 4: compute nnz, and row insert boundaries
    unsigned iter3_i, iter3_pos;
    unsigned inner_row_id, inner_col_id;
    bool is_BELL;
    unsigned find_val_id;
    std::vector<std::vector<unsigned>> COO_row_crd(((row_size-1)/blockSize)+1);
    std::vector<std::vector<unsigned>> COO_col_crd(((row_size-1)/blockSize)+1);
    std::vector<std::vector<float>> COO_val(((row_size-1)/blockSize)+1);
    T_BELL->vectorArray.resize(T_BELL->vLevel[2]->size * level1_size, std::vector<float>(blockSize*blockSize, 0));
    #pragma omp parallel for private(iter3_i, iter3_pos, col_block_id, inner_row_id, inner_col_id, is_BELL)
    for (iter3_i = 0; iter3_i < ((row_size-1)/blockSize)+1; iter3_i++) {
      for (iter3_pos = row_block_ptr[iter3_i]; iter3_pos < row_block_ptr[iter3_i+1]; iter3_pos++) {
        col_block_id = col_crd[iter3_pos]/blockSize;
        inner_row_id = row_crd[iter3_pos]%blockSize;
        inner_col_id = col_crd[iter3_pos]%blockSize;
        // check if col_block_id is in BELL
        is_BELL = false;
        for (find_val_id = iter3_i*level1_size;
             find_val_id < (iter3_i + 1)*level1_size;
             find_val_id ++) {
          if (T_BELL->vLevel[3]->crd[find_val_id] == col_block_id) {
            is_BELL = true;
            break;
          }
        }
        if (is_BELL) {
          unsigned correct_col_id = find_val_id%level1_size;
          // std::cout << "correct_col_id = " << correct_col_id << "\n";
          T_BELL->vectorArray[iter3_i*level1_size+correct_col_id][inner_row_id*blockSize+inner_col_id] = values[iter3_pos];
        } else {
          COO_row_crd[iter3_i].push_back(row_crd[iter3_pos]);
          COO_col_crd[iter3_i].push_back(col_crd[iter3_pos]);
          COO_val[iter3_i].push_back(values[iter3_pos]);
        }
      }
    }
    // std::cout << "vectorArray = ";
    // for (unsigned n = 0; n < T_BELL->vectorArray.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < T_BELL->vectorArray[n].size(); m++) 
    //     std::cout << T_BELL->vectorArray[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_row_crd, size = " << COO_row_crd.size();
    // for (unsigned n = 0; n < COO_row_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_row_crd[n].size(); m++) 
    //     std::cout << COO_row_crd[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_col_crd, size = " << COO_col_crd.size();
    // for (unsigned n = 0; n < COO_col_crd.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_col_crd[n].size(); m++) 
    //     std::cout << COO_col_crd[n][m] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO_val, size = " << COO_val.size();
    // for (unsigned n = 0; n < COO_val.size(); n++) {
    //   std::cout << "\n";
    //   for (unsigned m = 0; m < COO_val[n].size(); m++) 
    //     std::cout << COO_val[n][m] << "  ";
    // }
    // std::cout << "\n";

    // step 5: compute valueArray. Dispatch the values to COO and BELL.
    unsigned COO_nnz = 0;
    unsigned iter4_i, iter4_j;
    sparT->vLevel[1]->same_path.push_back(0);
    sparT->vLevel[2]->same_path.push_back(0);
    for (iter4_i = 0; iter4_i < ((row_size-1)/blockSize)+1; iter4_i++) {
      for (iter4_j = 0; iter4_j < COO_row_crd[iter4_i].size(); iter4_j++) {
        sparT->vLevel[1]->crd.push_back(COO_row_crd[iter4_i][iter4_j]);
        sparT->vLevel[2]->crd.push_back(COO_col_crd[iter4_i][iter4_j]);
        sparT->valueArray.push_back(COO_val[iter4_i][iter4_j]);
        if (COO_nnz > 0) {
          bool same_row = (sparT->vLevel[1]->crd[COO_nnz] == sparT->vLevel[1]->crd[COO_nnz-1]);
          bool same_col = (sparT->vLevel[2]->crd[COO_nnz] == sparT->vLevel[2]->crd[COO_nnz-1]);
          sparT->vLevel[1]->same_path.push_back(same_row);
          sparT->vLevel[2]->same_path.push_back(same_row && same_col);
        }
        COO_nnz += 1;
      }
    }    
    sparT->vLevel[1]->ptr.push_back(COO_nnz);
    // std::cout << "COO level1 crd = ";
    // for (unsigned n = 0; n < sparT->vLevel[1]->crd.size(); n++) {
    //   std::cout << sparT->vLevel[1]->crd[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO level2 crd = ";
    // for (unsigned n = 0; n < sparT->vLevel[2]->crd.size(); n++) {
    //   std::cout << sparT->vLevel[2]->crd[n] << "  ";
    // }
    // std::cout << "\n";
    // std::cout << "COO value = ";
    // for (unsigned n = 0; n < sparT->valueArray.size(); n++) {
    //   std::cout << sparT->valueArray[n] << "  ";
    // }
    // std::cout << "\n";

    SparlayStruct* ret = new SparlayStruct;
    ret->vec.push_back((void*)sparT);
    ret->vec.push_back((void*)T_BELL);
    return (void*) ret;
  }
  

  void release(void *tensor) {
    delete static_cast<SparlayStruct *>(tensor);
  }

} // extern C

