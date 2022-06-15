#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <cassert>
#include <iomanip>

typedef long long u64;

class LevelStorage {
public:
  int type; //0: none, 1: trimmed, 2: fused, 3: trim+fuse
  int size; //dense iteration bound
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
  }
  bool operator == (const LevelStorage& A) {
    if (type != A.type || size != A.size) return 0;
    if (crd.size() != A.crd.size() || ptr.size() != A.ptr.size() || same_path.size() != A.same_path.size()) return 0;
    for (int i = 0; i < crd.size(); ++i) if (crd[i] != A.crd[i]) return 0;
    for (int i = 0; i < ptr.size(); ++i) if (ptr[i] != A.ptr[i]) return 0;
    for (int i = 0; i < same_path.size(); ++i) if (same_path[i] != A.same_path[i]) return 0;
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
    for (int i = 0; i < vec.size(); ++i) if (vec[i] != A.vec[i]) return 0;
    return 1;
  }
};

/*!
 * \brief assume that Level 0 is root
 */
class SparlayStorage {
public:

  std::vector< std::unique_ptr<LevelStorage> > vLevel;
  std::vector< std::unique_ptr<LLStorage> > ptValue;

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
  bool swap(int i, int j);

  void Print(std::ostream& fout, bool verbose=0);
  SparlayStorage copy() {
    SparlayStorage ret;
    for (int i = 0; i < vLevel.size(); ++i) {
      ret.vLevel.push_back(std::unique_ptr<LevelStorage>(new LevelStorage(*vLevel[i])));
    }
    for (int i = 0; i < ptValue.size(); ++i) {
      ret.ptValue.push_back(std::unique_ptr<LLStorage>(new LLStorage(*ptValue[i])));
    }
    return ret;
  }
  bool operator == (const SparlayStorage& A) {
    if (A.vLevel.size() != vLevel.size()) return 0;
    if (ptValue.size() != A.ptValue.size()) return 0;
    for (int i = 0; i < vLevel.size(); ++i) if (!((*vLevel[i]) == (*A.vLevel[i]))) return 0;
    for (int i = 0; i < ptValue.size(); ++i) if (!((*ptValue[i]) == (*A.ptValue[i]))) return 0;
    return 1;
  }
};

SparlayStorage* readFromFile(std::istream& fin);

void SparlayStorage::Print(std::ostream& fout, bool verbose) {
  fout << "==============================================" << std::endl;
  for (int i = 0; i < vLevel.size(); ++i) {
    fout << "crd: ";
    for (const auto ele: vLevel[i]->crd) {
      fout << std::setw(8) << ele+1;
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
        fout << std::setw(8) << ele+1;
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
  ret->vLevel.push_back(move(std::unique_ptr<LevelStorage>(rootStore)));

  std::vector< std::vector<int> > bucket;
  std::vector<int> pos, oriID;
  bucket.resize(H, {});
  pos.resize(rowStore->crd.size(), 0);
  oriID.resize(pos.size(), 0);
  for (int i = 0; i < rowStore->crd.size(); ++i) {
    assert(rowStore->crd[i] < bucket.size());
    bucket[rowStore->crd[i]].push_back(i);
    pos[i] = i;
    oriID[i] = i;
  }
  int ptr = 0;
  for (int i = 0; i < bucket.size(); ++i) {
    for (int j = 0; j < bucket[i].size(); ++j) {
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

  for (int i = 1; i < rowStore->crd.size(); ++i) {
    rowStore->same_path.push_back(rowStore->crd[i] == rowStore->crd[i-1]);
    colStore->same_path.push_back(
      (rowStore->crd[i] == rowStore->crd[i-1]) && (colStore->crd[i] == colStore->crd[i-1])
    );
  }

  ret->vLevel.push_back(move(std::unique_ptr<LevelStorage>(rowStore)));
  ret->vLevel.push_back(move(std::unique_ptr<LevelStorage>(colStore)));
  ret->ptValue.reserve(N_ele);
  for (int i = 0; i < N_ele; ++i) {
    ret->ptValue.push_back(move(std::unique_ptr<LLStorage>(valStore[i])));
  }
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
  for (int i = 0; i < vLevel[st_lv]->ptr.size(); ++i) {
    if (vLevel[st_lv]->ptr[i] < vLevel[st_lv]->ptr[i+1])
      dfsLowerPtr(st_lv, i, i, ed_lv, ret);
  }
  ret[0] = 0;
  for (int i = 1; i < ret.size(); ++i) {
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
  if (lv == (vLevel.size()-1)) {
    std::cerr << "Error: Attempt to **Grow** the last level" << std::endl;
    assert(0);
  }
  int st_lv = 0;
  for (int i = 1; i < vLevel.size(); ++i) {
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
    vLevel[cur_lv]->ptr.push_back(0);
    int cur_lv_size = vLevel[cur_lv]->size;
    if (vLevel[cur_lv]->type & 2) {
      for (int i = 0; i < cur_ptr.size()-1; ++i) {
        if (cur_ptr[i] != cur_ptr[i+1]) {
          vLevel[cur_lv]->crd.push_back(i % cur_lv_size);
          vLevel[cur_lv]->ptr.push_back(cur_ptr[i+1]);
          vLevel[cur_lv]->same_path.push_back(0);
        }
      }
    } else {
      for (int i = 0; i < cur_ptr.size()-1; ++i) {
        for (int j = cur_ptr[i]; j < cur_ptr[i+1]; ++j) {
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
    for (int i = 0; i < cur_ptr.size()-1; i += cur_lv_size) {
      if (vLevel[cur_lv]->type & LVFUSE) {
        int cnt = 0;
        for (int j = 0; j < cur_lv_size; ++j) {
          cnt += (cur_ptr[i+j] != cur_ptr[i+j+1]);
        }
        new_cur_ptr.push_back(cnt+ *(new_cur_ptr.end()-1));
      } else {
        new_cur_ptr.push_back(cur_ptr[i+cur_lv_size-1]);
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
      for (int i = 0; i < vLevel[lv]->crd.size(); ++i) {
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
      for (int i = 0; i < vLevel[cur_lv]->ptr.size()-1; ++i) {
        int cnt = vLevel[cur_lv]->ptr[i+1] - vLevel[cur_lv]->ptr[i];
        for (int j = saved_st_point; j < vLevel[cur_lv]->ptr[i+1]; ++j) {
          if (crd_deleted[j]) cnt--;
        }
        assert(cnt>=0);
        saved_st_point = vLevel[cur_lv]->ptr[i+1];
        vLevel[cur_lv]->ptr[i+1] = vLevel[cur_lv]->ptr[i]+cnt;
      }
    }
    /*
     * Task:
     *  Gen new crd for level[upper_lv, lv]
     * 
     * Note: 
     *   1) crd size should === crd[lv]-sum(crd_deleted[lv])
     */
    //TODO: precompute crd size and reserve vectors to speed up
    for (int cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      /*
       * Note: 
       *  1) should update: crd, same_path(must have)
       *  2) should not have ptr
       */
      assert(vLevel[cur_lv]->crd.size() == vLevel[lv]->crd.size());
      assert(vLevel[cur_lv]->ptr.size() == 0);
      assert(vLevel[cur_lv]->crd.size() == vLevel[cur_lv]->same_path.size());
      std::vector<int> new_crd;
      std::vector<bool> new_same_path;
      bool is_same_path = 1; /** !!!!!!! **/
      for (int i = 0; i < vLevel[cur_lv]->crd.size(); ++i) {
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
    /*
     * Task:
     *  1) Gen ptr for Level[lv]
     * 
     * Note:
     *  1) Shouldn't be the last level is there is work to do.
     */
    assert(lv != vLevel.size() - 1);
    assert(crd_deleted.size() == vLevel[lv+1]->crd.size());
    assert(!crd_deleted[0]);
    vLevel[lv]->ptr.reserve(vLevel[lv]->crd.size()+1);
    vLevel[lv]->ptr.push_back(0);
    int cnt = 1;
    for (int i = 1; i < crd_deleted.size(); ++i) {
      if (!crd_deleted[i]) {
        vLevel[lv]->ptr.push_back((*(vLevel[lv]->ptr.end()-1)) + cnt);
        cnt = 1;
      } else {
        cnt++;
      }
    }
    vLevel[lv]->ptr.push_back((*(vLevel[lv]->ptr.end()-1)) + cnt);
    assert(vLevel[lv]->ptr.size() == vLevel[lv]->crd.size()+1);
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
      assert(vLevel[upper_lv-1]->type&2); //it must be a fused level
      int cur_lv = upper_lv-1;
      if (!(vLevel[cur_lv]->type&LVINFO)) assert(vLevel[cur_lv]->ptr.size() == vLevel[cur_lv]->crd.size()+1);
      assert(vLevel[cur_lv]->ptr[0] == 0);
      assert(vLevel[lv]->ptr.size()-1 == *(vLevel[cur_lv]->ptr.end()-1));
      for (int i = 0; i < vLevel[cur_lv]->ptr.size()-1; ++i) {
        int idR = vLevel[cur_lv]->ptr[i+1]-1;
        vLevel[cur_lv]->ptr[i+1] = vLevel[lv]->ptr[idR+1];
      }
    }
    assert(vLevel[lv]->ptr.size() == vLevel[lv]->crd.size()+1);
    //TODO: precompute crd size and reserve vectors to speed up
    for (int cur_lv = upper_lv; cur_lv <= lv; ++cur_lv) {
      /*
       * Note: 
       *  1) should update: crd, same_path(must have)
       *  2) should not have ptr
       */
      if (cur_lv != lv) assert(vLevel[cur_lv]->ptr.size() == 0);
      assert(vLevel[cur_lv]->crd.size() == vLevel[lv]->crd.size());
      assert(vLevel[cur_lv]->crd.size() == vLevel[cur_lv]->same_path.size());
      std::vector<int> new_crd;
      std::vector<bool> new_same_path;
      for (int i = 0; i < vLevel[lv]->crd.size(); ++i) {
        for (int j = vLevel[lv]->ptr[i]; j < vLevel[lv]->ptr[i+1]; ++j) {
          new_crd.push_back(vLevel[cur_lv]->crd[i]);
          if (j == vLevel[lv]->ptr[i]) new_same_path.push_back(vLevel[cur_lv]->same_path[i]);
          else new_same_path.push_back(1);
        }
      }
      vLevel[cur_lv]->crd = new_crd;
      vLevel[cur_lv]->same_path = new_same_path;
    }
    /*
     * Task: delete ptr for Level[lv]
     */
    vLevel[lv]->ptr.clear();
    vLevel[lv]->type ^= LVFUSE;
  } else {
    vLevel[lv]->type ^= LVFUSE;
  }
  return 1;
}

//TODO:
bool SparlayStorage::swap(int i, int j) {
  return 1;
}

#define TI (double)clock()/CLOCKS_PER_SEC

int main(int argc, char* argv[]) {
  std::ios::sync_with_stdio(0);
  if (argc != 2) {
    std::cerr << "Wrong Usage" << std::endl;
    return 0;
  }
  std::ifstream fin(argv[1]);
  //Read a COO and convert to CSR
  std::cerr << "Reading to COO, column major is converted to row major." << std::endl;
  float tic = TI;
  auto sparT = readFromFile(fin);
  std::cerr << "Read done, time is " << std::setprecision(2) << std::fixed << TI-tic << "(s)." << std::endl;
  std::shared_ptr<SparlayStorage> oriT = std::make_shared<SparlayStorage>(sparT->copy());
#ifdef PRINT
  sparT->Print(std::cerr,1);
#endif
  tic = TI;
  sparT->fuse(1);
#ifdef PRINT
  sparT->Print(std::cerr,1);
#endif
  sparT->grow(1);
#ifdef PRINT
  sparT->Print(std::cerr,1);
#endif
  std::cerr << std::endl << "Conversion to CSR with trivial assertation(checking) passed, time is " << std::setprecision(2) << std::fixed << TI-tic << "(s)." << std::endl << std::endl;
  tic = TI;
  sparT->trim(1);
#ifdef PRINT
  sparT->Print(std::cerr,1);
#endif
  sparT->separate(1);
#ifdef PRINT
  sparT->Print(std::cerr,1);
#endif
  std::cerr << std::endl << "Conversion back to COO done, time is " << std::setprecision(2) << std::fixed << TI-tic << "(s)." << std::endl;
  if ((*oriT) == (*sparT)) {
    std::cerr << std::endl << "Checking passed." << std::endl;
  } else {
    std::cerr << std::endl << "Failed, check all four operations." << std::endl;
    return 1;
  }
  return 0;
}