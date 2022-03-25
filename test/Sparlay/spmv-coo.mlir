// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | \
// RUN: mlir-opt -convert-vector-to-scf --convert-scf-to-std --tensor-constant-bufferize \
// RUN:     --tensor-bufferize --std-bufferize --finalizing-bufferize --convert-vector-to-llvm \
// RUN:     --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 | as - |
// RUN: clang++ -L${YOUR_PROJECT_PATH}/sparlay/build/lib -lmlir_sparlay_runner_utils \
// RUN:     -L${LLVM_PROJECT_PATH}/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec
// RUN: ./exec

!Filename = type !llvm.ptr<i8>

module {
    // CHECK-LABEL: func @main()
    func private @getTensorFilename(index) -> (!Filename)
    func @main() {
        %i0 = constant 0: index
        %f0 = constant 0.0e+00: f64
         
        %B = constant dense<[1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00]> : tensor<6xf64>
        %B_mem = memref.buffer_cast %B : memref<6xf64>
 
        %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
        %A_COO = sparlay.new (%fileName) : !Filename to
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                             memref<?xf64>>
        
        %C = sparlay.multiply (%A_COO, %B_mem) { target = "CPU", pattern = "inner" } : 
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                memref<?xf64>>, memref<6xf64> to memref<4xf64>

        // // print result
        // // call @print_memref_f64(%C_mem) : (memref<4xf64>) -> ()
        %result = vector.transfer_read %C[%i0], %f0 : memref<4xf64>, vector<4xf64>
        vector.print %result : vector<4xf64>
        return
    }
}

// func private @print_memref_f64(%ptr : memref<4xf64>)
