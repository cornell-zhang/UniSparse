// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | \
// RUN: mlir-opt -convert-vector-to-scf --convert-scf-to-std --tensor-constant-bufferize \
// RUN:     --tensor-bufferize --std-bufferize --finalizing-bufferize --convert-vector-to-llvm \
// RUN:     --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-cast | \
// RUN: mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 | as - -o |
// RUN: clang++ -L${YOUR_PROJECT_PATH}/sparlay/build/lib -lmlir_sparlay_runner_utils \
// RUN:     -L${LLVM_PROJECT_PATH}/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec
// RUN: ./exec

!Filename = type !llvm.ptr<i8>

module {
    // CHECK-LABEL: func @spmv()
    func private @getTensorFilename(index) -> (!Filename)
    func @main() {
        %i0 = constant 0: index
        %f0 = constant 0.0e+00: f64
         
        %B = constant dense<[1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00]> : tensor<6xf64>
        %B_mem = memref.buffer_cast %B : memref<6xf64>
 
        %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
        %A_input = sparlay.new (%fileName) : !Filename to
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                             memref<?xf64>>
        // Lower to :
        // %index_0 = call @getTensorIndices(%fileName, %i0) : !Filename, index to memref<?xindex>
        // %index_1 = call @getTensorIndices(%fileName, %i1) : !Filename, index to memref<?xindex>
        // %value = call @getTensorValues(%fileName) : !Filename to memref<?xf64>
        // %crd = sparlay.struct_construct (%index_0, %index_1) : memref<?xindex>, memref<?xindex> to
        //      !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>
        // %A_input = sparlay.struct_construct (%crd, %value) : 
        //      !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, memref<?xf64> to 
        //      !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
        //                      memref<?xf64>>
        %A_CSR = sparlay.compress (%A_input)
            { compress_dim = 1: index, storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                             memref<?xf64>> to 
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<?xindex>, "crd", (i,j)->(j)> , memref<?xf64> >
        
        %C = sparlay.multiply (%A_CSR, %B_mem) { target = "CPU", pattern = "inner" } : 
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<?xindex>, "crd", (i,j)->(j)> , memref<?xf64> >,
            memref<6xf64> to memref<4xf64>
        

        // // print result
        // // call @print_memref_f64(%C_mem) : (memref<4xf64>) -> ()
        %result = vector.transfer_read %C[%i0], %f0 : memref<4xf64>, vector<4xf64>
        vector.print %result : vector<4xf64>
        return
    }
}

// func private @print_memref_f64(%ptr : memref<4xf64>)
