//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The UniSparse Authors.
//
//===----------------------------------------------------------------------===//

#ifndef UNISPARSE_TRANSLATION_EMITHLS_H
#define UNISPARSE_TRANSLATION_EMITHLS_H

#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <tuple>
#include <queue>
#include <map>

#include "IR/UniSparseDialect.h"
#include "IR/UniSparseOps.h"
#include "IR/UniSparseTypes.h"

using namespace std;

namespace mlir {
namespace unisparse {

LogicalResult emitTapaHLS(Operation *op, llvm::raw_ostream &os);
void registerEmitTapaHLSTranslation();

class StreamAttr {
public:
    AffineMap indMap;
    CrdMap crdMap;
    CompressMap compressMap;
    int level;
    bool val;
    StreamAttr(): indMap(nullptr), crdMap(nullptr), compressMap(nullptr), level(0), val(false) {};
    StreamAttr(AffineMap indMap, CrdMap crdMap, CompressMap compressMap, int level, bool val): 
        indMap(indMap), crdMap(crdMap), compressMap(compressMap), level(level), val(val) {};
};

class FixModules {
public:
    std::string async_read = R"(
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
)";

    std::string async_write = R"(
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
)";

    std::string device_header = R"(
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

)";

    std::string stream_read(const std::string &funcName, 
                            const std::string &dataType,
                            const std::string &mmapIn,
                            const std::string &fifoOut,
                            const std::string &size) {
        return R"(
void read_)"+funcName+R"((
    tapa::async_mmap<)"+dataType+R"(>& )"+mmapIn+R"(,
    tapa::ostream<)"+dataType+R"(>& )"+fifoOut+R"(,
    const int )"+size+R"() {
    for (int i_req = 0, i_resp = 0; i_resp < )"+size+R"(;) {
#pragma HLS pipeline II=1
        async_read()"+mmapIn+R"(, 
        )"+fifoOut+R"(, 
        )"+size+R"(, 
        i_req, i_resp);
    }
})";
};

    std::string repeater(const std::string &funcName, 
                         const std::string &inFifoType,
                         const std::string &inFifoName,
                         const std::string &outFifoType,
                         const std::string &outFifoName,
                         const std::string &size) {
        return R"(
void )"+funcName+R"((
    tapa::istream<)"+inFifoType+R"(>& )"+inFifoName+R"(,
    tapa::ostream<)"+outFifoType+R"(>& )"+outFifoName+R"(,
    const int )"+size+R"() {
    auto start = )"+inFifoName+R"(.read();
    for (int i = 0; i < )"+size+R"(; i++) {
        auto end = )"+inFifoName+R"(.read();
        for (int j = start; j < end;) {
#pragma HLS pipeline II=1
            if (!)"+outFifoName+R"(.full()) {
                )"+outFifoName+R"(.try_write(i);
                j++;
            }
        }
        start = end;
    }
})";
};

    std::string index_calc(const std::string &funcName,
                           std::vector<pair<string, string>> &fifos) {
        std::string index_calc_program = "";
        index_calc_program += R"(
void )"+funcName+R"(()";
    for (auto &fifo : fifos) {
        index_calc_program += R"(
    tapa::istream<)"+fifo.second+R"(>& )"+fifo.first+R"(,)"; 
    }
    for (auto &fifo : fifos) {
        index_calc_program += R"(
    tapa::ostream<)"+fifo.second+R"(>& out_)"+fifo.first+R"(,)"; 
    }
    /// remove the last `,`
    index_calc_program.pop_back();
    index_calc_program += R"(
){
    for (;;) {
#pragma HLS pipeline II=1
        bool enable = )";
    for (auto &fifo : fifos) {[]
        index_calc_program += R"(
            !)"+fifo.first+R"(.empty() &)"; 
    }
    for (auto &fifo : fifos) {
        index_calc_program += R"(
            !out_)"+fifo.first+R"(.full() &)"; 
    }
    /// remove the last `&`
    index_calc_program.pop_back();
    index_calc_program += R"(;
    )";
    for (auto &fifo : fifos) {
        index_calc_program += R"(
        )" + fifo.second + R"( )" + fifo.first + R"(_v, out_)" + fifo.first + R"(_v;)";
    }
    index_calc_program += R"(
        if (enable) {)";
    for (auto &fifo : fifos) {
        index_calc_program += R"(
            )"+fifo.first+R"(.try_read()"+fifo.first+R"(_v);)";
    }
    for (auto &fifo : fifos) {
        index_calc_program += R"(
            out_)"+fifo.first+R"(.try_read(out_)"+fifo.first+R"(_v);)";
    }
    index_calc_program += R"(
        }
    }
}
)";
    return index_calc_program;
};

    std::string PEMul(const std::string &funcName,
                      std::vector<tuple</*name*/string, /*type*/string, StreamAttr>> &inFifos,
                      pair</*name*/string,/*type*/string> &outFifoCrd,
                      pair</*name*/string,/*type*/string> &outFifoData,
                      std::vector<std::string> &sizes) {
        return R"(

}; // class FixModules

} // namespace unisparse
} // namespace mlir

#endif // UNISPARSE_TRANSLATION_EMITHLS_H
