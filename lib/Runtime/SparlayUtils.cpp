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

#ifdef MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

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
        printf("indices_read size = %zu, getRank = %lu\n", indices_read.size() , getRank());
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

static void readMTXHeader(FILE* file, char* fileName, uint64_t* metaData) {
    char line[1025];
    char header[64];
    char object[64];
    char format[64];
    char field[64];
    char symmetry[64];
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
        strcmp(toLower(format), "coordinate") || strcmp(toLower(field), "real") ||
        strcmp(toLower(symmetry), "general")) {
        fprintf(stderr,
                "Cannot find a general sparse matrix with type real in %s\n", fileName);
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

extern "C" {
    // refactor into a swiss army knife function in the future
    void* readSparseCoordinate(void* ptr) {
        char* fileName = static_cast<char *>(ptr);                                                  
                                                                                                    
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];                                                                     
        if (strstr(fileName, ".mtx")) {                                                             
            readMTXHeader(file, fileName, metaData);                                                
        } else if (strstr(fileName, ".tns")) {                                                      
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {                                                                                    
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        } 

        printf("in getTensorIndices  :\n");
        for (unsigned i = 0; i < 4; i++)
            printf("metaData[%u] = %lu \n", i, metaData[i]);                                                                                          
                                                                                                    
        uint64_t rank = metaData[0];    
        uint64_t nnz = metaData[1];                                                                 
        
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
            if (fscanf(file, "%lg\n", &val) != 1) {
                fprintf(stderr, "Cannot find next value in %s\n", fileName);
                exit(1);
            }
            tensor.insert(indices, val);
        }

        fclose(file);
        return &tensor;
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

        printf("ref->basePtr: %x\n", ref->basePtr);
        printf("ref->size: %zu\n", index->size());
        printf("ref->data: ");
        for (unsigned i = 0; i < index->size(); i++) {
            printf("%lu  ", *(ref->data + ref->offset + i * ref->strides[0]));
        }
        printf("\n");
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

        printf("value->basePtr: %x\n", ref->basePtr);
        printf("value->size: %zu\n", value->size());
        printf("value->data: ");
        for (unsigned i = 0; i < value->size(); i++) {
            printf("%f  ", *(ref->data + ref->offset + i * ref->strides[0]));
        }
        printf("\n");

    }

    void _mlir_ciface_calculateCSRSpMV(StridedMemRefType<double, 1> *out, StridedMemRefType<uint64_t, 1> *ptr, StridedMemRefType<uint64_t, 1> *col, StridedMemRefType<double, 1> *value, StridedMemRefType<double, 1> *input) {
      uint64_t row = ptr->sizes[0] - 1;
//      printf("row size is: %d\n", row);
      for(uint64_t i = 0; i < row; i++) {
	float temp = 0;
	for(uint64_t j = ptr->data[i]; j < ptr->data[i+1]; j++) {
	  temp += value->data[j] * input->data[col->data[j]];
//	  printf("value->data[%d] is: %f, col->data[%d] is: %d, input->data[%d] is: %f\n", j, value->data[j], j, col->data[j], col->data[j], input->data[col->data[j]]);
	}
        out->data[i] = temp;
//        printf("outdata[%d] is %f\n", i, out->data[i]);
      }
    }  

    // void delSparseCoordinate(void *tensor) {
    //     delete static_cast<SparseCoordinate<uint64_t, double> *>(tensor);
    // }

}

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
