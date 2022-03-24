/*******************************************************************************
* Read .mtx and .tns file format
*******************************************************************************/

#ifndef _MTX_READ_H_
#define _MTX_READ_H_

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h> 
#include <cstring>
#include <cinttypes>

#include "mkl_spblas.h"

using namespace std;

static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

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
                "Cannot find a coordinate format sparse matrix in %s\n", fileName);
        exit(1);
    }
    // if (strcmp(toLower(field), "pattern"))
    // strcmp(toLower(symmetry), "general")

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

template <typename valueTp>
class parse_CSC {
public:
    parse_CSC(char* fileName) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
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
                                                                                                    
        uint64_t nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];

        cscColPtr = (MKL_INT*)malloc((num_cols + 1) * sizeof(MKL_INT));
        cscRowInd = (MKL_INT*)malloc(nnz * sizeof(MKL_INT));
        cscValue = (valueTp*)malloc(nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        MKL_INT lastRowInd = 0;
        // cscColPtr[0] = 0;
        for (unsigned i = 0; i < nnz; i++) {
            MKL_INT rowInd = -1;                                                                      
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%" PRIu64, &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                exit(1);                                                                        
            }
            cscRowInd[i] = rowInd;
            if (fscanf(file, "%" PRIu64, &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                exit(1);                                                                        
            }
            while (colInd > lastRowInd) {
                cscColPtr[lastRowInd++] = i;
            }
            if (!isFieldPattern) {
                // Field is Pattern
                cscValue[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%" PRIu64, &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                    exit(1);                                                                        
                }
                cscValue[i] = value;
            }
        }
    }

    ~parse_CSC() {
        free(cscColPtr);
        free(cscRowInd);
        free(cscValue);
    }

    int num_rows, num_cols;
    MKL_INT* cscColPtr;
    MKL_INT* cscRowInd;
    valueTp* cscValue;
};

template <typename valueTp>
class parse_COO {
public:
    parse_COO(char* fileName) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
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
                                                                                                    
        num_nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];

        cooRowInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        cooColInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        cooValue = (valueTp*)malloc(num_nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT rowInd = -1;
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%d", &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                exit(1);                                                                        
            }
            cooRowInd[i] = rowInd;
            if (fscanf(file, "%d", &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                exit(1);                                                                        
            }
            cooColInd[i] = colInd;
            if (!isFieldPattern) {
                // Field is Pattern
                cooValue[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%f", &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next index in %s\n", fileName);                    
                    exit(1);                                                                        
                }
                cooValue[i] = value;
            }
        }
    }

    ~parse_COO() {
        free(cooRowInd);
        free(cooColInd);
        free(cooValue);
    }

    int num_rows, num_cols;
    int num_nnz;
    MKL_INT* cooRowInd;
    MKL_INT* cooColInd;
    valueTp* cooValue;
};

#endif
