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

#include "mlir/ExecutionEngine/CRunnerUtils.h"

// #ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <iomanip>
#include "Eigen/Dense"

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

class LevelStorage {
public:
  int type; //0: none, 1: trimmed, 2: fused, 3: trim+fuse
  int size; //dense iteration bound
  bool lazy;
  std::vector<int> crd;
  std::vector<int> ptr;
  std::vector<bool> same_path;
  LevelStorage(
    int _type = 0, int _size = 0, std::vector<int> _crd = {}, std::vector<int> _ptr = {}, std::vector<bool> _same_path = {}
  ) {
    type = _type, size = _size;
    crd = move(_crd);
    ptr = move(_ptr);
    same_path = move(_same_path);
    lazy = 0;
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
 
class LLStorage {
public:
  std::vector<float> vec;
  LLStorage(std::vector<float> _vec = {}) { vec = move(_vec); }
  LLStorage(float v) { vec.clear(); vec.push_back(v); }
  bool operator == (const LLStorage& A) {
    if (vec.size() != A.vec.size()) return 0;
    for (size_t i = 0; i < vec.size(); ++i) if (vec[i] != A.vec[i]) return 0;
    return 1;
  }
};

/*!
 * \brief assume that Level 0 is root
 */
class SparlayStorage {
public:

  std::vector< std::shared_ptr<LevelStorage> > vLevel;
  std::vector< std::shared_ptr<Vector2i> > exprs;
  std::vector<int> oriSize;
  std::vector< std::shared_ptr<LLStorage> > ptValue;

  #define LVINFO 4
  #define LVTRIM 1
  #define LVFUSE 2

  SparlayStorage() {}

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

  void getSize(int lv) {
    assert(lv < exprs.size());
    assert(oriSize.size() == 3);
    
    Vector2i t0(0,0), t1(0,oriSize[2]), t2(oriSize[1],0), t3(oriSize[1],oriSize[2]);
    const auto& expr = exprs[lv];
    int mn = std::min(0, std::min(expr->dot(t1), std::min(expr->dot(t2), expr->dot(t3))));
    int mx = std::max(0, std::max(expr->dot(t1), std::max(expr->dot(t2), expr->dot(t3))));
    vLevel[lv]->size = (mx - mn);
  }

  void Print(std::ostream& fout, bool verbose=0);
  SparlayStorage copy() {
    SparlayStorage ret;
    for (size_t i = 0; i < vLevel.size(); ++i) {
      ret.vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage(*vLevel[i])));
    }
    for (size_t i = 0; i < ptValue.size(); ++i) {
      ret.ptValue.push_back(std::shared_ptr<LLStorage>(new LLStorage(*ptValue[i])));
    }
    return ret;
  }
  bool operator == (const SparlayStorage& A) {
    if (A.vLevel.size() != vLevel.size()) return 0;
    if (ptValue.size() != A.ptValue.size()) return 0;
    for (size_t i = 0; i < vLevel.size(); ++i) if (!((*vLevel[i]) == (*A.vLevel[i]))) return 0;
    for (size_t i = 0; i < ptValue.size(); ++i) if (!((*ptValue[i]) == (*A.ptValue[i]))) return 0;
    return 1;
  }
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
  auto A = ptValue[0]->vec.size();
  for (size_t j = 0; j < ptValue.size(); ++j) {
    assert(ptValue[j]->vec.size() == A);
  }
  for (size_t i = 0; i < A; ++i) {
    fout << "val: ";
    for (size_t j = 0; j < ptValue.size(); ++j) {
      fout << std::setw(8) << ptValue[j]->vec[i];
    }
    fout << std::endl;
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
  std::vector<LLStorage*> valStore;

  //Must be row major, otherwise the fuse operation will be incorrect
  //trim(0), fuse(None), (d0, ... dn)

  rowStore->type = colStore->type = 1;
  rowStore->size = H, colStore->size = W;
  rowStore->crd.reserve(N_ele), colStore->crd.reserve(N_ele), valStore.reserve(N_ele);
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
    valStore.push_back(new LLStorage(v));
  }

  rootStore->ptr.push_back(rowStore->crd.size());

  auto ret = new SparlayStorage();
  ret->vLevel.push_back(move(std::shared_ptr<LevelStorage>(rootStore)));

  std::vector< std::vector<int> > bucket;
  std::vector<int> pos, oriID;
  bucket.resize(H, {});
  pos.resize(rowStore->crd.size(), 0);
  oriID.resize(pos.size(), 0);
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    assert(rowStore->crd[i] < bucket.size());
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
      for (int k = 0; k < pos.size(); ++k) std::cerr << pos[k] << ' ';
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

  ret->vLevel.push_back(move(std::shared_ptr<LevelStorage>(rowStore)));
  ret->vLevel.push_back(move(std::shared_ptr<LevelStorage>(colStore)));
  ret->ptValue.reserve(N_ele);
  for (int i = 0; i < N_ele; ++i) {
    ret->ptValue.push_back(move(std::shared_ptr<LLStorage>(valStore[i])));
  }
  ret->oriSize.push_back(0);
  ret->oriSize.push_back(H), ret->oriSize.push_back(W);
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,0)));
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(1,0)));
  ret->exprs.push_back(std::shared_ptr<Vector2i>(new Vector2i(0,1)));
  return ret;
}

/*!
 * \param pos current dense position
 */
void SparlayStorage::dfsLowerPtr(int cur_lv, int id, int pos, int target_lv, std::vector<int>& ret) {
  if (cur_lv == target_lv) {
    assert(ret.size() > pos);
    assert(ret[pos+1] == -1);
    if (vLevel[cur_lv]->ptr.size()) {
      assert(id < vLevel[cur_lv]->ptr.size());
      ret[pos+1] = vLevel[cur_lv]->ptr[id+1];
    } else {
      ret[pos+1] = id+1;
    }
    return;
  }
  int nxtLevelSize = vLevel[cur_lv+1]->size;
  if (vLevel[cur_lv]->ptr.size()) {
    int idL = vLevel[cur_lv]->ptr[id], idR = vLevel[cur_lv]->ptr[id+1];
    assert(vLevel[cur_lv+1]->crd.size() >= idR);
    for (int to = idL; to < idR; ++to) {
      int to_pos = pos * nxtLevelSize + vLevel[cur_lv+1]->crd[to];
      dfsLowerPtr(cur_lv+1, to, to_pos, target_lv, ret);
    }
  } else {
    assert(vLevel[cur_lv+1]->crd.size() > id);
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
  for (size_t i = 0; i < vLevel[st_lv]->ptr.size(); ++i) {
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
  for (int i = 0; i < ret.size(); ++i) std::cerr << std::setw(8) << ret[i];
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
  while (lower_lv != 0 && (vLevel[lower_lv-1]->type & 1)) lower_lv--;
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
    cur_ptr = move(new_cur_ptr);
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
      register size_t sz = vLevel[cur_lv]->ptr.size()-1;
      for (size_t i = 0; i < sz; ++i) {
        int cnt = vLevel[cur_lv]->ptr[i+1] - vLevel[cur_lv]->ptr[i];
        register auto bound = vLevel[cur_lv]->ptr[i+1];
        for (auto j = saved_st_point; j < bound; ++j) {
          if (crd_deleted[j]) cnt--;
        }
        // assert(cnt>=0);
        saved_st_point = vLevel[cur_lv]->ptr[i+1];
        vLevel[cur_lv]->ptr[i+1] = vLevel[cur_lv]->ptr[i]+cnt;
      }
    }
    //TODO: precompute crd size and reserve vectors to speed up
    auto sz = crd_deleted.size();
    int crd_reserve_size = 0;
    for (size_t i = 0; i < sz; ++i) {
      crd_reserve_size += (!crd_deleted[i]);
    }
    for (auto cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      assert(vLevel[cur_lv]->crd.size() == vLevel[lv]->crd.size());
      assert(vLevel[cur_lv]->ptr.size() == 0);
      assert(vLevel[cur_lv]->crd.size() == vLevel[cur_lv]->same_path.size());
      std::vector<int> new_crd;
      std::vector<bool> new_same_path;
      new_crd.reserve(crd_reserve_size);
      new_same_path.reserve(crd_reserve_size);
      bool is_same_path = 1; /** !!!!!!! **/
      sz = vLevel[cur_lv]->crd.size();
      for (size_t i = 0; i < sz; ++i) {
        is_same_path &= vLevel[cur_lv]->same_path[i];
        if (!crd_deleted[i]) {
          new_crd.push_back(vLevel[cur_lv]->crd[i]);
          new_same_path.push_back(is_same_path);
          is_same_path = 1;
        }
      }
      vLevel[cur_lv]->crd = new_crd;
      vLevel[cur_lv]->same_path = new_same_path;
    }
    vLevel[lv]->ptr.reserve(vLevel[lv]->crd.size()+1);
    vLevel[lv]->ptr.push_back(0);
    int cnt = 1;
    sz = crd_deleted.size();
    register int prev_pushed = 0;
    for (size_t i = 1; i < sz; ++i) {
      if (!crd_deleted[i]) {
        prev_pushed += cnt;
        vLevel[lv]->ptr.push_back(prev_pushed);
        cnt = 1;
      } else {
        cnt++;
      }
    }
    vLevel[lv]->ptr.push_back(prev_pushed + cnt);
    // assert(vLevel[lv]->ptr.size() == vLevel[lv]->crd.size()+1);
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
    // assert(vLevel[lv]->ptr.size() == vLevel[lv]->crd.size()+1);
    //TODO: precompute crd size and reserve vectors to speed up
    for (int cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      std::vector<int> new_crd;
      std::vector<bool> new_same_path;
      new_crd.resize(vLevel[lv]->ptr[vLevel[lv]->crd.size()], 0);
      new_same_path.resize(vLevel[lv]->ptr[vLevel[lv]->crd.size()], 0);
      int ptr = 0;
      for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
        for (auto j = vLevel[lv]->ptr[i]; j < vLevel[lv]->ptr[i+1]; ++j) {
          new_crd[ptr] = vLevel[cur_lv]->crd[i];
          if (j == vLevel[lv]->ptr[i]) new_same_path[ptr] = vLevel[cur_lv]->same_path[i];
          else new_same_path[ptr] = 1;
          ptr++;
        }
      }
      vLevel[cur_lv]->crd = new_crd;
      vLevel[cur_lv]->same_path = new_same_path;
    }
    vLevel[lv]->ptr.clear();
    vLevel[lv]->type ^= LVFUSE;
  } else {
    vLevel[lv]->type ^= LVFUSE;
  }
  return 1;
}

bool bucketSortTwoLayer(
  std::shared_ptr<LevelStorage> rowStore, std::shared_ptr<LevelStorage> colStore, std::vector<std::shared_ptr<LLStorage> >& valStore
) {
  std::vector<int> Count;
  std::vector<int> bucketHolder;
  std::vector<int> bucketPos;
  std::vector<int> bucketStIndex;
  std::vector<int> pos, oriID;
  Count.resize(rowStore->size, 0);
  bucketStIndex.resize(rowStore->size, 0);
  bucketPos.resize(rowStore->size, 0);
  bucketHolder.resize(rowStore->crd.size(), 0);
  pos.resize(rowStore->crd.size(), 0);
  oriID.resize(pos.size(), 0);
  int mnCrd = 0;
  for (size_t i = 0; i < rowStore->crd.size(); ++i) mnCrd = std::min(mnCrd, rowStore->crd[i]);
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    Count[rowStore->crd[i]-mnCrd]++;
    pos[i] = i;
    oriID[i] = i;
  }
  for (size_t i = 1; i < rowStore->size; ++i) {
    bucketStIndex[i] = bucketStIndex[i-1] + Count[i-1];
  }
  assert(bucketStIndex[rowStore->size-1]+Count[rowStore->size-1] == rowStore->crd.size());
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    int cur_crd = rowStore->crd[i]-mnCrd;
    bucketHolder[bucketStIndex[cur_crd]+(bucketPos[cur_crd]++)] = i;
  }
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    int cur_pos = pos[bucketHolder[i]];
    // assert(cur_pos >= ptr);
    if (cur_pos != i) {
      std::swap(rowStore->crd[i], rowStore->crd[cur_pos]);
      std::swap(colStore->crd[i], colStore->crd[cur_pos]);
      std::swap(valStore[i], valStore[cur_pos]);
      pos[oriID[i]] = cur_pos;
      oriID[cur_pos] = oriID[i];
    }
  }
  for (size_t i = 1; i < rowStore->crd.size(); ++i) {
    rowStore->same_path[i] = (rowStore->crd[i] == rowStore->crd[i-1]);
    colStore->same_path[i] = (
      (rowStore->crd[i] == rowStore->crd[i-1]) && (colStore->crd[i] == colStore->crd[i-1])
    );
  }
  return 1;
}

bool bucketSortOneLayer(
  std::shared_ptr<LevelStorage> rowStore, std::vector<std::shared_ptr<LLStorage> >& valStore
) {
  std::vector<int> Count;
  std::vector<int> bucketHolder;
  std::vector<int> bucketPos;
  std::vector<int> bucketStIndex;
  std::vector<int> pos, oriID;
  Count.resize(rowStore->size, 0);
  bucketStIndex.resize(rowStore->size, 0);
  bucketPos.resize(rowStore->size, 0);
  bucketHolder.resize(rowStore->crd.size(), 0);
  pos.resize(rowStore->crd.size(), 0);
  oriID.resize(pos.size(), 0);
  int mnCrd = 0;
  for (size_t i = 0; i < rowStore->crd.size(); ++i) mnCrd = std::min(mnCrd, rowStore->crd[i]);
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    Count[rowStore->crd[i]-mnCrd]++;
    pos[i] = i;
    oriID[i] = i;
  }
  for (size_t i = 1; i < rowStore->size; ++i) {
    bucketStIndex[i] = bucketStIndex[i-1] + Count[i-1];
  }
  assert(bucketStIndex[rowStore->size-1]+Count[rowStore->size-1] == rowStore->crd.size());
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    int cur_crd = rowStore->crd[i]-mnCrd;
    bucketHolder[bucketStIndex[cur_crd]+(bucketPos[cur_crd]++)] = i;
  }
  for (size_t i = 0; i < rowStore->crd.size(); ++i) {
    int cur_pos = pos[bucketHolder[i]];
    // assert(cur_pos >= ptr);
    if (cur_pos != i) {
      std::swap(rowStore->crd[i], rowStore->crd[cur_pos]);
      std::swap(valStore[i], valStore[cur_pos]);
      pos[oriID[i]] = cur_pos;
      oriID[cur_pos] = oriID[i];
    }
  }
  for (size_t i = 1; i < rowStore->crd.size(); ++i) {
    rowStore->same_path[i] = (rowStore->crd[i] == rowStore->crd[i-1]);
  }
  return 1;
}

bool SparlayStorage::swap(const int LU, const int LD) {
  assert(LU < LD);
  assert(LU == 1 && LD == 2);
  for (int i = LU; i <= LD; ++i) {
    assert(!(vLevel[i]->type & LVFUSE));
  }
  if ((vLevel[LU]->type&LVTRIM) && (vLevel[LD]->type&LVTRIM)) {
    std::swap(vLevel[LU], vLevel[LD]);
    std::swap(exprs[LU], exprs[LD]);
    #define rowStore vLevel[LU]
    #define colStore vLevel[LD]
    #define valStore ptValue

    bucketSortTwoLayer(rowStore, colStore, valStore);

    #undef rowStore
    #undef colStore
    #undef valStore
  } else {
    std::cerr << "Not implemented" << std::endl;
    assert(0);
  }
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
  if (Ltarget < Lsrc) {
    bucketSortTwoLayer(vLevel[Ltarget], vLevel[Lsrc], ptValue);
  }
  return 1;
}

bool SparlayStorage::sub(const int Ltarget, const int Lsrc) {
  assert(Lsrc != Ltarget);
  for (int i = Lsrc; i <= Ltarget; ++i) {
    assert(!(vLevel[i]->type & LVFUSE));
  }
  assert(vLevel[Lsrc]->crd.size() == vLevel[Ltarget]->crd.size());
  for (size_t i = 0; i < vLevel[Lsrc]->crd.size(); ++i) {
    vLevel[Ltarget]->crd[i] -= vLevel[Lsrc]->crd[i];
  }
  (*exprs[Ltarget]) -= (*exprs[Lsrc]);
  getSize(Ltarget);
  if (Ltarget < Lsrc) {
    bucketSortTwoLayer(vLevel[Ltarget], vLevel[Lsrc], ptValue);
  }
  return 1;
}

bool SparlayStorage::vectorize(const int lv) {
  assert(lv == 2);
  assert(lv == vLevel.size()-1);
  bool mark = !(vLevel[lv-1]->type&LVFUSE);
  fuse(lv-1);
  auto& father_ptr = vLevel[lv-1]->ptr;
  auto& father_crd = vLevel[lv-1]->crd;
  int cur_lv_size = vLevel[lv]->size;
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vLevel[lv]->crd.size() == ptValue.size());
    std::vector<float> V;
    for (int i = 0; i < father_crd.size(); ++i) {
      V.clear();
      V.resize(cur_lv_size, 0.0);
      assert(father_ptr[i+1] > prev_ptr);
      for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
        V[vLevel[lv]->crd[j]] = ptValue[j]->vec[0];
      }
      ptValue[i]->vec = V; //copy
      int add_one = (father_ptr[i+1]>prev_ptr);
      int new_ptr = father_ptr[i] + add_one;
      prev_ptr = father_ptr[i+1];
      father_ptr[i+1] = new_ptr;
    }
    vLevel.pop_back();
    int vec_size = father_crd.size();
    while (ptValue.size() > vec_size) ptValue.pop_back();
  } else {
    std::cerr << "Should not happen" << std::endl;
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
  int cur_lv_size = vLevel[lv]->size;
  if (father_ptr.size()) {
    int prev_ptr = 0;
    assert(vLevel[lv]->crd.size() == ptValue.size());
    std::vector<int> new_crd;
    std::vector<bool> new_same_path;
    std::vector<std::shared_ptr<LLStorage> > new_value;
    int Size = ptValue[0]->vec.size();
    for (int i = 0; i < father_crd.size(); ++i) {
      assert(father_ptr[i+1] == prev_ptr+1);
      int cnt = 0;
      for (int j = prev_ptr; j < father_ptr[i+1]; ++j) {
        for (int k = 0; k < ptValue[j]->vec.size(); ++k) {
          if (ptValue[j]->vec[k]) {
            cnt++;
            new_crd.push_back(k);
            new_same_path.push_back(0);
            new_value.push_back(std::shared_ptr<LLStorage>(new LLStorage({ptValue[j]->vec[k]})));
          }
        }
      }
      int new_ptr = father_ptr[i] + cnt;
      prev_ptr = father_ptr[i+1];
      father_ptr[i+1] = new_ptr;
    }
    vLevel.push_back(std::shared_ptr<LevelStorage>(new LevelStorage(1, Size, new_crd, {}, new_same_path)));
    ptValue = new_value;
  } else {
    std::cerr << "Should not happen" << std::endl;
    assert(0);
  }
  if (mark) separate(lv);
  return 1;
}

bool SparlayStorage::neg(int lv) {
  assert(!(vLevel[lv]->type & LVFUSE));
  if (lv != 1) {
    std::cerr << "Not implemented yet" << std::endl;
    assert(0);
  }
  for (size_t i = 0; i < vLevel[lv]->crd.size(); ++i) {
    vLevel[lv]->crd[i] = -vLevel[lv]->crd[i];
  }
  bucketSortTwoLayer(vLevel[lv], vLevel[lv+1], ptValue);
  return 1;
}

#define TI (double)clock()/CLOCKS_PER_SEC

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
        auto tic = TI;
        char* fileName = static_cast<char*>(ptr);
        std::ifstream fin(fileName);
        void* ret = readFromFile(fin);
        fin.close();
        std::cerr << std::endl << "Read from file done, time = " << TI-tic << "(s)" << std::endl;
        return ret;
    }

    void* _mlir_ciface_sptFuse(void* ptr, int lv) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->fuse(lv+1);
        std::cerr << std::endl << "Fuse done, time = " << TI-tic << "(s)" << std::endl;
        return (void*)sparT;
    }

    void* _mlir_ciface_sptGrow(void* ptr, int lv) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->grow(lv+1);
        std::cerr << std::endl << "Grow done, time = " << TI-tic << "(s)" << std::endl;
        return (void*)sparT;
    }

    void* _mlir_ciface_sptTrim(void* ptr, int lv) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->trim(lv+1);
        std::cerr << std::endl << "Trim done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptSeparate(void* ptr, int lv) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->separate(lv+1);
        std::cerr << std::endl << "Separate done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptSwap(void* ptr, int LU, int LD) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->swap(LU+1, LD+1);
        std::cerr << std::endl << "Swap done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptSub(void* ptr, int Ltarget, int Lsrc) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->sub(Ltarget+1, Lsrc+1);
        std::cerr << std::endl << "Sub done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr, 1);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptAdd(void* ptr, int Ltarget, int Lsrc) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->add(Ltarget+1, Lsrc+1);
        std::cerr << std::endl << "Add done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr, 1);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptNeg(void* ptr, int lv) {
        auto tic = TI;
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->neg(lv+1);
        std::cerr << std::endl << "Neg done, time = " << TI-tic << "(s)" << std::endl;
        // sparT->Print(std::cerr);
        return (void*)sparT;
    }

    void* _mlir_ciface_sptVectorize(void* ptr, int lv) {
      auto tic = TI;
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->vectorize(lv+1);
      std::cerr << std::endl << "Vectorize done, time = " << TI-tic << "(s)" << std::endl;
      return (void*)sparT;
    }

    void* _mlir_ciface_sptDevectorize(void* ptr) {
      auto tic = TI;
      SparlayStorage* sparT = (SparlayStorage*)(ptr);
      sparT->devectorize();
      std::cerr << std::endl << "Devectorize done, time = " << TI-tic << "(s)" << std::endl;
      return (void*)sparT;
    }

    void _mlir_ciface_sptPrint(void* ptr) {
        SparlayStorage* sparT = (SparlayStorage*)(ptr);
        sparT->Print(std::cerr, 1);
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

    void _mlir_ciface_calculateCOOSpMV(StridedMemRefType<double, 1> *out, 
                                       StridedMemRefType<uint64_t, 1> *row, 
                                       StridedMemRefType<uint64_t, 1> *col, 
                                       StridedMemRefType<double, 1> *value, 
                                       StridedMemRefType<double, 1> *input,
                                       uint64_t size_0, uint64_t size_1) {
      uint64_t nnz = row->sizes[0];
    //   uint64_t out_size = out->sizes[0];
    //   printf("out size = %lu \n", out_size);
    //   printf("nnz is: %lu\n", nnz);
      double *result = new double[size_0];
      for (uint64_t i = 0; i < size_0; i++) {
        // out->data[i] = 0;
        result[i] = 0;
      }

      for(uint64_t i = 0; i < nnz; i++) {
        // double temp = 0;
        uint64_t rowInd = row->data[i];
        uint64_t colInd = col->data[i];
        result[rowInd] += value->data[i] * input->data[colInd];
        // printf("value->data is: %f, input->data[%lu] is: %f \n", value->data[i], colInd, input->data[colInd]);
        // printf("outdata[%lu] is %f\n", rowInd, result[rowInd]);
      }

        out->data = result;    
        out->basePtr = result;
        out->offset = 0;  
        out->strides[0] = 1;
        
    //     printf("output: (");
    //   for (uint64_t i = 0; i < size_0; i++) {
    //     printf("%f ", out->data[i]);
    //     // out->data[i] = result[i];
    //   }
    //   printf(")\n");
      delete[] result;
    } 

    // void _mlir_ciface_release(void *ptr) {
    //     delete []ptr;
    // }

    // void delSparseCoordinate(void *tensor) {
    //     delete static_cast<SparseCoordinate<uint64_t, double> *>(tensor);
    // }

}

// #endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
