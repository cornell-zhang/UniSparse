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

struct Stream {
public:
    string name; // SSA style name for each stream
    string dataType;
    string size;
    AffineMap indMap;
    CrdMap crdMap;
    CompressMap compressMap;
    int level;
    bool val;
    int stage; // 1-read, 2-decompress, 3-indexCalc, 4-PEMul, 5-PEAcc, 6-write
    bool IO; // true for IO, false for inter module stream
Stream(): name(""), dataType(""), size(""), indMap(nullptr), 
    crdMap(), compressMap(), level(-1), val(false), stage(-1), IO(false) {};
Stream(string name, string dataType, string size, AffineMap indMap, 
            CrdMap crdMap, CompressMap compressMap, int level, bool val, 
            int stage, bool IO):
    name(name), dataType(dataType), size(size), indMap(indMap), 
    crdMap(crdMap), compressMap(compressMap), level(level), val(val), 
    stage(stage), IO(IO) {};
Stream(const Stream &stream):
    name(stream.name), dataType(stream.dataType), size(stream.size), 
    indMap(stream.indMap), crdMap(stream.crdMap), 
    compressMap(stream.compressMap), level(stream.level), val(stream.val), 
    stage(stream.stage), IO(stream.IO) {};
Stream& operator=(const Stream &stream) {
    name = stream.name;
    dataType = stream.dataType;
    size = stream.size;
    indMap = stream.indMap;
    crdMap = stream.crdMap;
    compressMap = stream.compressMap;
    level = stream.level;
    val = stream.val;
    stage = stream.stage;
    IO = stream.IO;
    return *this;
};
}; // struct Stream

struct Module {
public:
    string name;
    vector<Stream> inStreams;
    vector<Stream> outStreams;
    vector<string> sizes;
    bool join;
Module(): name(""), inStreams(), outStreams(), sizes(), join(true) {};
Module(string name, vector<Stream> inStreams, vector<Stream> outStreams, 
        vector<string> sizes, bool join):
    name(name), inStreams(inStreams.begin(), inStreams.end()),
    outStreams(outStreams.begin(), outStreams.end()), 
    sizes(sizes.begin(), sizes.end()), join(join) {};
Module(const Module &module):
    name(module.name), inStreams(module.inStreams.begin(), module.inStreams.end()),
    outStreams(module.outStreams.begin(), module.outStreams.end()), 
    sizes(module.sizes.begin(), module.sizes.end()), join(module.join) {};

}; // struct Module

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

    std::string stream_read(Module readInModule) {
        string funcName = readInModule.name; 
        string dataType = readInModule.inStreams[0].dataType;
        string mmapIn = readInModule.inStreams[0].name;
        string fifoOut = readInModule.outStreams[0].name;
        string size = readInModule.sizes[0];
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
}
)";
}; // stream_read

    std::string repeater(Module decompModule) {
        string funcName = decompModule.name;
        string inFifoType = decompModule.inStreams[0].dataType;
        string inFifoName = decompModule.inStreams[0].name;
        string outFifoType = decompModule.outStreams[0].dataType;
        string outFifoName = decompModule.outStreams[0].name;
        string size = decompModule.sizes[0];
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

    std::string index_calc(Module indexCalcModule) {
        std::string funcName = indexCalcModule.name;
        std::vector<Stream> inStreams = indexCalcModule.inStreams;
        std::vector<Stream> outStreams = indexCalcModule.outStreams;
        std::string index_calc_program = "";
        index_calc_program += R"(
void )"+funcName+R"(()";
    for (auto &fifo : inStreams) {
        index_calc_program += R"(
    tapa::istream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)"; 
    }
    for (auto &fifo : outStreams) {
        index_calc_program += R"(
    tapa::ostream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)"; 
    }
    /// remove the last `,`
    index_calc_program.pop_back();
    index_calc_program += R"(
){
    for (;;) {
#pragma HLS pipeline II=1
        bool enable = )";
    for (auto &fifo : inStreams) {
        index_calc_program += R"(
            !)"+fifo.name+R"(.empty() &)"; 
    }
    for (auto &fifo : outStreams) {
        index_calc_program += R"(
            !)"+fifo.name+R"(.full() &)"; 
    }
    /// remove the last `&`
    index_calc_program.pop_back();
    index_calc_program += R"(;
    )";
    std::string valVarName;
    for (auto &fifo : inStreams) {
        index_calc_program += R"(
        )" + fifo.dataType + R"( )" + fifo.name + R"(_v;)";
        if (fifo.val) {
            valVarName = fifo.name+"_v";
        }
    }
    for (auto &fifo : outStreams) {
        index_calc_program += R"(
        )" + fifo.dataType + R"( )" + fifo.name + R"(_v;)";
    }
    index_calc_program += R"(
        if (enable) {)";
    for (auto &fifo : inStreams) {
        index_calc_program += R"(
            )"+fifo.name+R"(.try_read()"+fifo.name+R"(_v);)";
    }
    unsigned outDimNum = 0;
    for (auto &fifo : outStreams) {
        if (fifo.val) {
            index_calc_program += R"(
            )"+fifo.name+R"(_v = )"+valVarName+R"(;)";
        } else {
            index_calc_program += R"(
            )" + fifo.name + R"(_v = )";
            unsigned inDimNum = 0;
            for (auto &inFifo: inStreams) {
                if (!inFifo.val) {
                    index_calc_program += R"(a)"+to_string(outDimNum) 
                    +to_string(inDimNum)+R"( * )"
                        +inFifo.name+R"(_v + )";
                    inDimNum++;
                }
            }
            /// remove the last ` + `
            index_calc_program.pop_back();
            index_calc_program.pop_back();
            index_calc_program.pop_back();
            index_calc_program += R"(;)";
        }
        outDimNum++;
    }
    for (auto &fifo : outStreams) {
        index_calc_program += R"(
            )"+fifo.name+R"(.try_write()"+fifo.name+R"(_v);)";
    }
    index_calc_program += R"(
        }
    }
}
)";
    return index_calc_program;
};

    std::string PE_Mul(Module PEMulModule) {
        std::string funcName = PEMulModule.name;
        std::string index_calc_program = "";
        std::vector<string> sizes;
        string sparse_size, sparse_val, dense_val, dense_ind;
        string out_val, out_ind;
        int out_ind_level;

        index_calc_program += R"(
void )"+funcName+R"(()";
    for (auto &fifo : PEMulModule.inStreams) {
        index_calc_program += R"(
    tapa::istream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)";
        if (find(sizes.begin(), sizes.end(), fifo.size) == sizes.end()) {
            sizes.push_back(fifo.size);
        }
    }
    for (auto &fifo : PEMulModule.outStreams) {
        index_calc_program += R"(
    tapa::ostream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)";
    }
    for (auto &size : sizes) {
        index_calc_program += R"(
    const int )"+size+R"(,)";
    }
    /// remove the last `,`
    index_calc_program.pop_back();
    index_calc_program += R"(
){)";
    for (auto &fifo : PEMulModule.inStreams) {
        if (fifo.val && fifo.compressMap == CompressMap()) { // dense val
            dense_val = fifo.name + "_buffer";
            index_calc_program += R"(
    )" + fifo.dataType + R"( )" + fifo.name + R"(_buffer[X_BUFFER_SIZE];)";
            index_calc_program += R"(
#pragma HLS bind_storage variable=)"+fifo.name+R"(_buffer type=ram_2p impl=bram)";
            index_calc_program += R"(
    for (int i = 0; i < )"+fifo.size+R"(;) {
#pragma HLS pipeline II=1
        if (!)" + fifo.name + R"(.empty()){
            )" + fifo.dataType + R"( )" + fifo.name + R"(_v;
            )" + fifo.name + R"(.try_read()" + fifo.name + R"(_v);
            )" + fifo.name + R"(_buffer[i] = )" + fifo.name + R"(_v;
            i++;
        }
    }
)";
        }
    }
    for (auto &fifo : PEMulModule.inStreams) {
        if (fifo.compressMap != CompressMap()) { // sparse data structures
            sparse_size = fifo.size;
            index_calc_program += R"(
    )" + fifo.dataType + R"( )" + fifo.name + R"(_v;)";
            if (fifo.val)
                sparse_val = fifo.name + "_v";
        }
    }
    for (auto &fifo : PEMulModule.outStreams) {
        if (fifo.val) { // sparse data structures
            index_calc_program += R"(
    )" + fifo.dataType + R"( )" + fifo.name + R"(_v;)";
            out_val = fifo.name + "_v";
        } else {
            out_ind_level = fifo.level;
        }
    }
    index_calc_program += R"(
    for (int i = 0; i < )"+sparse_size+R"(;) {
#pragma HLS pipeline II=1
        if ()";
    for (auto &fifo : PEMulModule.inStreams) {
        if (fifo.compressMap != CompressMap()) { // sparse data structures
            index_calc_program += R"(!)" + fifo.name + R"(.empty() & )";
            if (fifo.level == out_ind_level)
                out_ind = fifo.name + "_v";
            else if (!fifo.val)
                dense_ind = fifo.name + "_v";
        }
    }
    index_calc_program.pop_back();
    index_calc_program.pop_back();
    index_calc_program += R"(){)";
    for (auto &fifo : PEMulModule.inStreams) {
        if (fifo.compressMap != CompressMap()) { // sparse data structures
            index_calc_program += R"(
            )" + fifo.name + R"(.try_read()" + fifo.name + R"(_v);)";
        }
    }
    index_calc_program += R"(
            )" + out_val + R"( = )" + sparse_val + R"( * )" + dense_val + R"([)" + dense_ind + R"(];)";
    for (auto &fifo : PEMulModule.outStreams) {
        if (fifo.val) { // sparse data structures
            index_calc_program += R"(
            )" + fifo.name + R"(.try_write()" + out_val + R"();)";
        } else {
            index_calc_program += R"(
            )" + fifo.name + R"(.try_write()" + out_ind + R"();)";
        }
    }
    index_calc_program += R"(
            i++;
        }
    }
}
)";
    return index_calc_program;
}; // PE_Mul

    std::string PE_Sum(Module PESumModule) {
        std::string funcName = PESumModule.name;
        std::string index_calc_program = "";

        index_calc_program += R"(
void )"+funcName+R"(()";
    for (auto &fifo : PESumModule.inStreams) {
        index_calc_program += R"(
    tapa::istream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)";
    }
    for (auto &fifo : PESumModule.outStreams) {
        index_calc_program += R"(
    tapa::ostream<)"+fifo.dataType+R"(>& )"+fifo.name+R"(,)";
    }
    for (auto &size : PESumModule.sizes) {
        index_calc_program += R"(
    const int )"+size+R"(,)";
    }
    /// remove the last `,`
    index_calc_program.pop_back();
    index_calc_program += R"(
){)";
    
    /// Assume the output is stored as a buffer
    for (auto &fifo : PESumModule.outStreams) {
        index_calc_program += R"(
    float )" + fifo.name + R"(_buffer[Y_BUFFER_SIZE];
#pragma HLS bind_storage variable=)" + fifo.name + R"(_buffer type=ram_2p impl=URAM latency=1)";
        index_calc_program += R"(
    for (int i = 0; i < )"+fifo.size+R"(; i++) {
#pragma HLS pipeline II=1
        )" + fifo.name + R"(_buffer[i] = 0;
    })";
    }

    /// check if all input streams are of equal lengths
    string in_val_size = PESumModule.inStreams[0].size;
    for (auto &fifo : PESumModule.inStreams) {
        assert (in_val_size == fifo.size);
        index_calc_program += R"(
    )" + fifo.dataType + R"( )" + fifo.name + R"(_v;)";
    }

    index_calc_program += R"(
    for (int i = 0; i < )"+in_val_size+R"(;) {
#pragma HLS pipeline II=1
        if ()";
    for (auto &fifo : PESumModule.inStreams) {
        index_calc_program += R"(!)" + fifo.name + R"(.empty() & )";
    }
    index_calc_program.pop_back();
    index_calc_program.pop_back();
    index_calc_program += R"(){)";
    string in_val, in_ind;
    for (auto &fifo : PESumModule.inStreams) {
        index_calc_program += R"(
            )" + fifo.name + R"(.try_read()" + fifo.name + R"(_v);)";
        if (fifo.val)
            in_val = fifo.name + "_v";
        else
            in_ind = fifo.name + "_v";
    }
    for (auto &fifo : PESumModule.outStreams) {
        index_calc_program += R"(
            )" + fifo.name + R"(_buffer[)" + in_ind + R"(] += )" + in_val + R"(;)";
    }
    index_calc_program += R"(
            i++;
        }
    })";
    
    for (auto &fifo : PESumModule.outStreams) {
        index_calc_program += R"(
    for (int i = 0; i < )"+fifo.size+R"(; i++) {
#pragma HLS pipeline II=1
        )" + fifo.name + R"(.write()" + fifo.name + R"(_buffer[i]);
    })";
    }
    index_calc_program += R"(
}
)";
    return index_calc_program;
}; // PE_Sum

    string stream_write(Module writeOutModule) {
        string funcName = writeOutModule.name; 
        string dataType = writeOutModule.inStreams[0].dataType;
        string fifoIn = writeOutModule.inStreams[0].name;
        string mmapOut = writeOutModule.outStreams[0].name;
        string size = writeOutModule.sizes[0];
        return R"(
void read_)"+funcName+R"((
    tapa::istream<)"+dataType+R"(>& )"+fifoIn+R"(,
    tapa::async_mmap<)"+dataType+R"(>& )"+mmapOut+R"(,
    const int )"+size+R"() {
    for (int i_req = 0, i_resp = 0; i_resp < )"+size+R"(;) {
#pragma HLS pipeline II=1
        async_write()"+fifoIn+R"(, 
        )"+mmapOut+R"(, 
        )"+size+R"(, 
        i_req, i_resp);
    }
}
)";
}; // stream_write

}; // class FixModules

} // namespace unisparse
} // namespace mlir

#endif // UNISPARSE_TRANSLATION_EMITHLS_H
