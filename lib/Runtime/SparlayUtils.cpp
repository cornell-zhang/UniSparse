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

// #define GETINDICES(TYPE)
    void _mlir_ciface_getTensorIndices(StridedMemRefType<uint64_t, 1> *ref, void *ptr, uint64_t dim) {  
        char* fileName = static_cast<char *>(ptr);                                                  
                                                                                                    
        FILE *file = fopen(fileName, "r");                                                          
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
                                                                                                    
        uint64_t size = metaData[0];                                                                
        uint64_t nnz = metaData[1];                                                                 
        std::vector<uint64_t> indices(nnz);                                                         
        for (unsigned i = 0; i < nnz; i++) {                                                        
            uint64_t idx = -1;                                                                      
            for (uint64_t r = 0; r < size; r++) {                                                   
                if (fscanf(file, "%" PRIu64, &idx) != 1) {                                          
                    fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                    exit(1);                                                                        
                }
                // Add 0-based index.
                if (r == dim)
                    indices.push_back(idx - 1);
            }
            double val;
            if (fscanf(file, "%lg\n", &val) != 1) {
                fprintf(stderr, "Cannot find next value in %s\n", fileName);
                exit(1);
            }
        }

        ref->basePtr = ref->data = indices.data();    
        ref->offset = 0;  
        ref->sizes[0] = indices.size();  
        ref->strides[0] = 1; 

        fclose(file);
    }

// #define GETVALUES(TYPE)
    void _mlir_ciface_getTensorValues(StridedMemRefType<double, 1> *ref, void *ptr) {
        char* fileName = static_cast<char *>(ptr);

        FILE *file = fopen(fileName, "r");
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

        // generate calls to initiate crd memrefs
        uint64_t size = metaData[0];
        uint64_t nnz = metaData[1];
        std::vector<double> values(nnz);
        for (unsigned i = 0; i < nnz; i++) {
            uint64_t idx = -1;
            for (uint64_t r = 0; r < size; r++) {
                if (fscanf(file, "%" PRIu64, &idx) != 1) {
                    fprintf(stderr, "Cannot find next index in %s\n", fileName);
                    exit(1);
                }
            }
            double val;
            if (fscanf(file, "%lg\n", &val) != 1) {
                fprintf(stderr, "Cannot find next value in %s\n", fileName);
                exit(1);
            }
            values.push_back(val);
        }

        ref->basePtr = ref->data = values.data();    
        ref->offset = 0;  
        ref->sizes[0] = values.size();  
        ref->strides[0] = 1; 

        fclose(file);
    }

}

#endif // MLIR_CRUNNERUTILS_DEFINE_FUNCTIONS
