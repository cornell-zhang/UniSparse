#ifndef VEC_HPP
#define VEC_HPP

#include <memory>
#include <cstdlib>
#include <iostream>
#include <vector>

template<typename V>
class victor {
public:
  V* obj;
  size_t len;
  size_t cap;

  ~victor() { free(obj); obj = nullptr, len = cap = 0; }

  victor() {
    obj = new V; obj[0] = 0; len = 0; cap = 1;
  }

  victor(size_t _cap) { //init with zero
    (_cap == 0 ? _cap = 1 : _cap = _cap);
    cap = _cap;
    obj = (V*)calloc(cap, sizeof(V));
    len = 0;
  }

  victor(const victor<V>& other) {
    //copy!
    obj = (V*)malloc(other.cap*sizeof(V));
    memcpy(obj, other.obj, other.len*sizeof(V));
    len = other.len;
    cap = other.cap;
  }

  victor& operator = (const victor<V>& other) {
    //copy
    if (obj != nullptr) { //destruct first
      free(obj), obj = nullptr;
    }
    obj = (V*)malloc(other.cap * sizeof(V));
    memcpy(obj, other.obj, other.len*sizeof(V));
    len = other.len;
    cap = other.cap;
    return *this;
  }

  victor& operator = (victor<V>&& other) {
    //swap
    std::swap(obj, other.obj);
    len = other.len;
    cap = other.cap;
    return *this;
  }

  void clear() {
    //clear do not free the space
    len = 0;
  }

  size_t size() const { return len; }

  inline void reserve(size_t new_cap) {
    if (new_cap < cap) return;
    V* new_obj = (V*)malloc(new_cap * sizeof(V));
    memcpy(new_obj, obj, len * sizeof(V));
    cap = new_cap;
    std::swap(obj, new_obj);
    free(new_obj);
  }

  void push_back(const V& a) {
    if (len == cap) reserve(2 * cap);
    obj[len++] = a;
  }

  void push_back(V&& a) {
    if (len == cap) reserve(2 * cap);
    obj[len++] = a;
  }

  void resize0(size_t new_cap) {
    if (new_cap < cap) {
      memset(obj, 0, new_cap * sizeof(V));
      len = new_cap;
      return;
    }
    V* new_obj = (V*)calloc(new_cap, sizeof(V));
    cap = len = new_cap;
    std::swap(obj, new_obj);
    free(new_obj);
  }

  void insert0_end(size_t num) {
    size_t tgt_cap = cap;
    while (len + num < tgt_cap) tgt_cap *= 2;
    reserve(tgt_cap);
    memset(obj+len, 0, (cap-len)*sizeof(V));
    len = cap;
  }

  bool empty() const { return len == 0; }

  V& operator [](size_t idx) {
    // assert(idx < len);
    return obj[idx];
  }

  V operator [](size_t idx) const {
    // assert(idx < len);
    return obj[idx];
  }

  V* data() { return obj; }

  void dump() {
    for (size_t i = 0; i < len; ++i) {
      std::cerr << obj[i] << ' ';
    }
    std::cerr << std::endl;
  }

  std::vector<V>&& to_vector() const {
    return std::move(std::vector<V>(obj, obj+len));
  }

  std::vector<V>&& to_vector() {
    return std::move(std::vector<V>(obj, obj+len));
  }
};

#endif