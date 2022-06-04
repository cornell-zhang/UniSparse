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
  int L, R; //dense iteration bound
  std::vector<int> crd;
  std::vector<int> ptr;
  std::vector<bool> same_path;
  std::vector<u64> pos;
  LevelStorage(
    int _type = 0, int _L = 0, int _R = 0, std::vector<int> _crd = {}, std::vector<int> _ptr = {}, std::vector<bool> _same_path = {},
    std::vector<u64> _pos = {}
  ) {
    type = _type, L = _L, R = _R;
    crd = move(_crd);
    ptr = move(_ptr);
    same_path = move(_same_path);
    pos = move(_pos);
  }
};
 
class LLStorage {
public:
  std::vector<float> vec;
  LLStorage(std::vector<float> _vec = {}) { vec = move(_vec); }
  LLStorage(float v) { vec.clear(); vec.push_back(v); }
};

class SparlayStorage {
public:

  std::vector< std::unique_ptr<LevelStorage> > vLevel;
  std::vector< std::unique_ptr<LLStorage> > ptValue;

  SparlayStorage() {}

  void dfsLowerPos(int cur_lv, int id, long long cur_pos, int target_lv, std::vector<long long>& ret);
  std::vector<long long> lowerPos(int st_lv, int ed_lv);

  bool trim(int level);
  bool fuse(const int level);
  bool grow(const int level);
  bool defuse(int level);
  bool swap(int i, int j);

  void expand_non_trimmed_level();
  void shrink_non_trimmed_level();

  void Print(std::ostream& fout, bool verbose=0);
};

SparlayStorage* readFromFile(std::istream& fin);

void SparlayStorage::Print(std::ostream& fout, bool verbose) {
  fout << "==============================================" << std::endl;
  for (int i = 0; i < vLevel.size(); ++i) {
    if (vLevel[i]->pos.size()) {
      assert(vLevel[i]->pos.size() == vLevel[i]->crd.size());
      fout << "pos: ";
      for (const auto ele: vLevel[i]->pos) {
        fout << std::setw(8) << ele+1;
      }
      fout << std::endl;
    }
    fout << "crd: ";
    for (const auto ele: vLevel[i]->crd) {
      fout << std::setw(8) << ele+1;
    }
    fout << "      (" << vLevel[i]->type << ")";
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
  std::vector<LLStorage*> valStore;

  //Must be row major, otherwise the fuse operation will be incorrect
  //trim(0), fuse(None), (d0, ... dn)

  rowStore->type = colStore->type = 1;
  rowStore->L = 0, rowStore->R = H, colStore->L = 0, colStore->R = W;
  rowStore->crd.reserve(N_ele), colStore->crd.reserve(N_ele), valStore.reserve(N_ele);
  rowStore->same_path.push_back(0), colStore->same_path.push_back(0);
  for (int row, col, i = 0; i < N_ele; ++i) {
    float v;
    fin >> row >> col >> v;
    --row, --col;
    // std::cerr << row << ' ' << col << ' ' << v << std::endl;
    rowStore->crd.push_back(row);
    if (i != 0) {
      rowStore->same_path.push_back(rowStore->crd[i] == rowStore->crd[i-1]);
    }
    rowStore->pos.push_back(row);
    colStore->crd.push_back(col);
    if (i != 0) {
      colStore->same_path.push_back(
        (rowStore->crd[i] == rowStore->crd[i-1]) && (colStore->crd[i] == colStore->crd[i-1])
      );
    }
    valStore.push_back(new LLStorage(v));
  }
  auto ret = new SparlayStorage();
  ret->vLevel.push_back(move(std::unique_ptr<LevelStorage>(rowStore)));
  ret->vLevel.push_back(move(std::unique_ptr<LevelStorage>(colStore)));
  ret->ptValue.reserve(N_ele);
  for (int i = 0; i < N_ele; ++i) {
    ret->ptValue.push_back(move(std::unique_ptr<LLStorage>(valStore[i])));
  }
  return ret;
}

void SparlayStorage::dfsLowerPos(int cur_lv, int id, long long cur_pos, int target_lv, std::vector<u64>& ret) {
  if (cur_lv == target_lv) {
    ret.push_back(cur_pos);
    return;
  }
  int nxtLevelSize = vLevel[cur_lv+1]->R - vLevel[cur_lv+1]->L;
  if (vLevel[cur_lv]->ptr.size()) {
    int idL = vLevel[cur_lv]->ptr[id], idR = vLevel[cur_lv]->ptr[id+1];
    assert(vLevel[cur_lv+1]->crd.size() >= idR);
    for (int to = idL; to < idR; ++to) {
      int to_pos = cur_pos * nxtLevelSize + vLevel[cur_lv+1]->crd[to];
      dfsLowerPos(cur_lv+1, to, to_pos, target_lv, ret);
    }
  } else {
    assert(vLevel[cur_lv+1]->crd.size() > id);
    dfsLowerPos(cur_lv+1, id, cur_pos * nxtLevelSize + vLevel[cur_lv+1]->crd[id], target_lv, ret);
  }
}

std::vector<u64> SparlayStorage::lowerPos(int st_lv, int ed_lv) {
  std::vector<u64> ret;
  assert(vLevel[st_lv]->pos.size());
  assert(vLevel[st_lv]->pos.size() == vLevel[st_lv]->crd.size());
  for (int i = 0; i < vLevel[st_lv]->pos.size(); ++i) {
    dfsLowerPos(st_lv, i, vLevel[st_lv]->pos[i], ed_lv, ret);
  }
#ifdef DEBUG
  Print(std::cerr, 1);
  std::cerr << "ret: ";
  for (int i = 0; i < ret.size(); ++i) std::cerr << std::setw(8) << ret[i];
  std::cerr << std::endl;
#endif
  assert(ret.size() == vLevel[ed_lv]->crd.size());
  return move(ret);
}


bool SparlayStorage::grow(const int lv) {
  if (!(vLevel[lv]->type & 1)) { //already not trimmed
    return 1;
  }
  if (lv == (vLevel.size()-1)) {
    std::cerr << "Error: Attempt to **Grow** the last level" << std::endl;
    assert(0);
  }
  int st_lv = -1;
  for (int i = 0; i < vLevel.size(); ++i) {
    if (vLevel[i]->type & 1) { st_lv = i; break; }
  }
  assert(st_lv != -1);
  std::cerr << st_lv << std::endl;
  assert(st_lv <= lv);
  auto new_pos = lowerPos(st_lv, lv+1);
  vLevel[lv+1]->pos = new_pos;
  for (int i = st_lv; i <= lv; ++i) {
    assert(vLevel[i]->type & 1);
    vLevel[i]->type ^= 1;
    vLevel[i]->crd.clear();
    vLevel[i]->ptr.clear();
    vLevel[i]->pos.clear();
    vLevel[i]->same_path.clear();
  }
  return 1;
}

//TODO: not finished yet.
bool SparlayStorage::trim(int lv) {
  if (vLevel[lv]->type & 1) return 1; //already trimmed
  int lower_lv = vLevel.size()-1;
  while (lower_lv != 0 && (vLevel[lower_lv+1]->type & 1)) lower_lv++;
  assert(lower_lv > lv);
  /*
   * Task: 
   *  1) Gen crd, ptr(is fused), same_path for Level[lv, lower_lv-1]
   *  2) Change type for Level[lv, lower_lv-1]
   *  3) Gen path for Level[lv]
   */
  // tagA: Save the pos of lower_lv and clear the pos of lower_lv
  std::vector<long long> child_pos = vLevel[lower_lv]->pos;
  vLevel[lower_lv]->pos.clear();
  // End of tagA
  assert(child_pos.size());
  assert(child_pos.size() == vLevel[lower_lv]->crd.size());
  for (int cur_lv = lower_lv - 1; cur_lv >= lv; --cur_lv) {
    assert(vLevel[cur_lv]->crd.size() == 0);
    assert(vLevel[cur_lv]->ptr.size() == 0);
    assert(vLevel[cur_lv]->same_path.size() == 0);
    assert(vLevel[cur_lv]->pos.size() == 0);
    // tagB: Build crd, same_path, pos, ptr(if fused)
    int child_lv_size = vLevel[cur_lv+1]->R - vLevel[cur_lv+1]->L;

    if (vLevel[cur_lv]->type & 2) {

    } else {

    }
    // End of tagB
  }
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
#ifdef DEBUG
      std::cerr << "Fuse has no work to do at Lv: " << lv << std::endl;
#endif 
      vLevel[lv]->type |= 2;
      return 1;
    }
    //update possibly the ptr of a fused level
    if (upper_lv!=0 && (vLevel[upper_lv-1]->type&1)) {
      assert(vLevel[upper_lv-1]->type&2); //it must be a fused level
      int cur_lv = upper_lv-1;
      assert(vLevel[cur_lv]->ptr.size() == vLevel[cur_lv]->crd.size()+1);
      int saved_st_point = 0;
      assert(vLevel[cur_lv]->ptr[0] == 0);
      for (int i = 0; i < vLevel[cur_lv]->crd.size(); ++i) {
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
       *  1) should update: crd, pos(if any), same_path(must have)
       *  2) should not have ptr
       */
      assert(vLevel[cur_lv]->crd.size() == vLevel[lv]->crd.size());
      assert(vLevel[cur_lv]->ptr.size() == 0);
      assert(vLevel[cur_lv]->crd.size() == vLevel[cur_lv]->same_path.size());
      std::vector<int> new_crd;
      std::vector<bool> new_same_path;
      std::vector<long long> new_pos;
      if (vLevel[cur_lv]->pos.size()) assert(vLevel[cur_lv]->pos.size() == vLevel[cur_lv]->crd.size());
      bool is_same_path = 1; /** !!!!!!! **/
      for (int i = 0; i < vLevel[cur_lv]->crd.size(); ++i) {
        is_same_path &= vLevel[cur_lv]->same_path[i];
        if (!crd_deleted[i]) {
          new_crd.push_back(vLevel[cur_lv]->crd[i]);
          new_same_path.push_back(is_same_path);
          is_same_path = 1;
          if (vLevel[cur_lv]->pos.size()) new_pos.push_back(vLevel[cur_lv]->pos[i]);
        }
      }
      vLevel[cur_lv]->crd = new_crd;
      vLevel[cur_lv]->same_path = new_same_path;
      if (vLevel[cur_lv]->pos.size()) vLevel[cur_lv]->pos = new_pos;
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
    vLevel[lv]->type |= 2;
  } else {
    vLevel[lv]->type |= 2;
  }
  return 1;
}

//TODO:
bool SparlayStorage::defuse(int level) {
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
  std::cerr << "Reading to COO, must be row major." << std::endl;
  float tic = TI;
  auto sparT = readFromFile(fin);
  std::cerr << "Read done, time is " << std::setprecision(2) << std::fixed << TI-tic << "(s)." << std::endl;
#ifdef PRINT
  sparT->Print(std::cerr);
#endif
  tic = TI;
  sparT->fuse(0);
#ifdef PRINT
  sparT->Print(std::cerr);
#endif
  sparT->grow(0);
#ifdef PRINT
  sparT->Print(std::cerr);
#endif
  std::cerr << std::endl << "Conversion to CSR with trivial assertation(checking) passed, time is " << std::setprecision(2) << std::fixed << TI-tic << "(s)." << std::endl << std::endl;
  return 0;
}