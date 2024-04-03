
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for TAPA High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <ap_int.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <tapa.h>
#include "spmv.h"
#include <iostream>

using namespace std;


template <typename T, typename R>
inline void async_read(tapa::async_mmap<T> & A,
                       tapa::ostream<T> & fifo_A,
                       const R A_len,
                       R & i_req,
                       R & i_resp) {
#pragma HLS inline
    if ((i_req < A_len) &
        !A.read_addr.full()) {
        A.read_addr.try_write(i_req);
        ++i_req;
    }
    if (!fifo_A.full() & !A.read_data.empty()) {
        T tmp;
        A.read_data.try_read(tmp);
        fifo_A.try_write(tmp);
        ++i_resp;
    }
}

template <typename T, typename R>
inline void async_write(tapa::istream<T>& fifo_A,
                        tapa::async_mmap<T> & A,
                        const R A_len,
                        R & i_req,
                        R & i_resp) {
#pragma HLS inline
    if ((i_req < A_len) &
        !fifo_A.empty() &
        !A.write_addr.full() &
        !A.write_data.full()) {
        A.write_addr.try_write(i_req);
        T tmp;
        fifo_A.try_read(tmp);
        A.write_data.try_write(tmp);
        ++i_req;
    }
    uint8_t n_resp;
    if (A.write_resp.try_read(n_resp)) {
        i_resp += int(n_resp) + 1;
    }
}

void read_sparse0_ptr0(
    tapa::async_mmap<SPARSE0_PTR0_T>& sparse0_ptr0_mmapIn,
    tapa::ostream<SPARSE0_PTR0_T>& sparse0_ptr0_fifoOut,
    const int size_d0) {
    for (int i_req = 0, i_resp = 0; i_resp < size_d0;) {
#pragma HLS pipeline II=1
        async_read(sparse0_ptr0_mmapIn, 
        sparse0_ptr0_fifoOut, 
        size_d0, 
        i_req, i_resp);
    }
}
void read_sparse0_crd1(
    tapa::async_mmap<SPARSE0_CRD1_T>& sparse0_crd1_mmapIn,
    tapa::ostream<SPARSE0_CRD1_T>& sparse0_crd1_fifoOut,
    const int size_crd1) {
    for (int i_req = 0, i_resp = 0; i_resp < size_crd1;) {
#pragma HLS pipeline II=1
        async_read(sparse0_crd1_mmapIn, 
        sparse0_crd1_fifoOut, 
        size_crd1, 
        i_req, i_resp);
    }
}
void read_sparse0_val(
    tapa::async_mmap<SPARSE0_VAL_T>& sparse0_val_mmapIn,
    tapa::ostream<SPARSE0_VAL_T>& sparse0_val_fifoOut,
    const int size_val) {
    for (int i_req = 0, i_resp = 0; i_resp < size_val;) {
#pragma HLS pipeline II=1
        async_read(sparse0_val_mmapIn, 
        sparse0_val_fifoOut, 
        size_val, 
        i_req, i_resp);
    }
}
void read_dense1(
    tapa::async_mmap<DENSE1_T>& dense1_mmapIn,
    tapa::ostream<DENSE1_T>& dense1_fifoOut,
    const int size_d1) {
    for (int i_req = 0, i_resp = 0; i_resp < size_d1;) {
#pragma HLS pipeline II=1
        async_read(dense1_mmapIn, 
        dense1_fifoOut, 
        size_d1, 
        i_req, i_resp);
    }
}
void repeater_sparse0_ptr0_fifoOut(
    tapa::istream<SPARSE0_PTR0_T>& sparse0_ptr0_fifoOut,
    tapa::ostream<SPARSE0_PTR0_T>& sparse0_ptr0_fifoOut_crd,
    const int size_d0) {
    auto start = sparse0_ptr0_fifoOut.read();
    for (int i = 0; i < size_d0; i++) {
        auto end = sparse0_ptr0_fifoOut.read();
        for (int j = start; j < end;) {
#pragma HLS pipeline II=1
            if (!sparse0_ptr0_fifoOut_crd.full()) {
                sparse0_ptr0_fifoOut_crd.try_write(i);
                j++;
            }
        }
        start = end;
    }
}
void index_calc(
    tapa::istream<SPARSE0_CRD1_T>& sparse0_crd1_fifoOut,
    tapa::istream<SPARSE0_PTR0_T>& sparse0_ptr0_fifoOut_crd,
    tapa::ostream<SPARSE0_CRD1_T>& out_sparse0_crd1_fifoOut,
    tapa::ostream<SPARSE0_PTR0_T>& out_sparse0_ptr0_fifoOut_crd
)}
    for (;;) {
#pragma HLS pipeline II=1
        bool enable = 
            !sparse0_crd1_fifoOut.empty() &
            !sparse0_ptr0_fifoOut_crd.empty() &
            !out_sparse0_crd1_fifoOut.full() &
            !out_sparse0_ptr0_fifoOut_crd.full() ;

        SPARSE0_CRD1_T sparse0_crd1_fifoOut_v, out_sparse0_crd1_fifoOut_v;
        SPARSE0_PTR0_T sparse0_ptr0_fifoOut_crd_v, out_sparse0_ptr0_fifoOut_crd_v;
        if (enable) {
            sparse0_crd1_fifoOut.try_read(sparse0_crd1_fifoOut_v);
            sparse0_ptr0_fifoOut_crd.try_read(sparse0_ptr0_fifoOut_crd_v);
            out_sparse0_crd1_fifoOut.try_read(out_sparse0_crd1_fifoOut_v);
            out_sparse0_ptr0_fifoOut_crd.try_read(out_sparse0_ptr0_fifoOut_crd_v);
        }
    }
}

void top(
      tapa::mmap<DENSE1_T> dense1_mmapIn,
      tapa::mmap<SPARSE0_CRD1_T> sparse0_crd1_mmapIn,
      tapa::mmap<SPARSE0_PTR0_T> sparse0_ptr0_mmapIn,
      tapa::mmap<SPARSE0_VAL_T> sparse0_val_mmapIn,
      const int size_d0,
      const int size_d1,
      const int sparse0_ptr0_fifoOut_crd_size
){
      tapa::stream<DENSE1_T, FIFO_DEPTH> dense1_fifoOut("dense1_fifoOut");
      tapa::stream<SPARSE0_CRD1_T, FIFO_DEPTH> out_sparse0_crd1_fifoOut("out_sparse0_crd1_fifoOut");
      tapa::stream<SPARSE0_PTR0_T, FIFO_DEPTH> out_sparse0_ptr0_fifoOut_crd("out_sparse0_ptr0_fifoOut_crd");
      tapa::stream<SPARSE0_CRD1_T, FIFO_DEPTH> sparse0_crd1_fifoOut("sparse0_crd1_fifoOut");
      tapa::stream<SPARSE0_PTR0_T, FIFO_DEPTH> sparse0_ptr0_fifoOut("sparse0_ptr0_fifoOut");
      tapa::stream<SPARSE0_PTR0_T, FIFO_DEPTH> sparse0_ptr0_fifoOut_crd("sparse0_ptr0_fifoOut_crd");
      tapa::stream<SPARSE0_VAL_T, FIFO_DEPTH> sparse0_val_fifoOut("sparse0_val_fifoOut");
  tapa::task()
      .invoke<tapa::join>(
        dense1,
        dense1_mmapIn,
        dense1_fifoOut,
        size_d1
        )
      .invoke<tapa::detach>(
        index_calc,
        sparse0_crd1_fifoOut,
        sparse0_ptr0_fifoOut_crd,
        out_sparse0_crd1_fifoOut
        )
      .invoke<tapa::join>(
        repeater_sparse0_ptr0_fifoOut,
        sparse0_ptr0_fifoOut,
        sparse0_ptr0_fifoOut_crd,
        size_d0
        )
      .invoke<tapa::join>(
        sparse0_crd1,
        sparse0_crd1_mmapIn,
        sparse0_crd1_fifoOut,
        size_crd1
        )
      .invoke<tapa::join>(
        sparse0_ptr0,
        sparse0_ptr0_mmapIn,
        sparse0_ptr0_fifoOut,
        size_d0
        )
      .invoke<tapa::join>(
        sparse0_val,
        sparse0_val_mmapIn,
        sparse0_val_fifoOut,
        size_val
        )
  ;
}
