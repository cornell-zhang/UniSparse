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
#include <queue>
#include <map>

namespace mlir {
namespace unisparse {

LogicalResult emitTapaHLS(Operation *op, llvm::raw_ostream &os);
void registerEmitTapaHLSTranslation();

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

};
} // namespace unisparse
} // namespace mlir

#endif // UNISPARSE_TRANSLATION_EMITHLS_H
