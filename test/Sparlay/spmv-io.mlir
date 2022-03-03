// RUN: sparlay-opt %s | sparlay-opt | FileCheck %s

!Filename = type !llvm.ptr<i8>

module {
    // CHECK-LABEL: func @spmv()
    func private @getTensorFilename(index) -> (!Filename)
    func @spmv() {
        %i0 = constant 0: index
        %i1 = constant 1: index
        %i4 = constant 4: index
        %f0 = constant 0.0e+00 : f32
         
        %B = constant dense<[3.0e+00, 2.0e+00, 1.0e+00, 4.0e+00]> : tensor<4xf32>
        %C = constant dense<[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00]> : tensor<4xf32>

        %B_mem = memref.buffer_cast %B : memref<4xf32>
        %C_mem = memref.buffer_cast %C : memref<4xf32>
 
        %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
          
        %A_input = sparlay.new (%fileName) : !Filename to
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                             memref<?xf32>>
        // Lower to :
        // %index_1 = call @getTensorIndices(%fileName, %i0) : !Filename, index to memref<?xindex>
        // %index_2 = call @getTensorIndices(%fileName, %i1) : !Filename, index to memref<?xindex>
        // %value = call @getTensorValues(%fileName) : !Filename to memref<?xf32>
        // %crd = sparlay.struct_construct (%index_1, %index_2) : memref<?xindex>, memref<?xindex> to
        //      !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>
        // %A_input = sparlay.struct_construct (%crd, %value) : 
        //      !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>, memref<?xf32> to 
        //      !sparlay.struct<[4, 6], !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>, 
        //                      memref<7xf32>>
        %A_CSR = sparlay.compress (%A_input)
            { compress_dim = 0: index, storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
                             memref<?xf32>> to 
            !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<?xindex>, "crd", (i,j)->(j)> , memref<?xf32> >

        // /* Potential compute operation
        // %C_mem = linalg.generic {target = "CPU", pattern = "inner"}
        //     ins(%A_CSR, %B_mem: 
        //         !sparlay.struct<!sparlay.struct<memref<?xindex>>, 
        //                         !sparlay.struct<memref<?xindex>>,
        //                         memref<?xf32>>, 
        //         memref<4xf32>)
        //     outs(%C_mem: memref<4x4xf32>) {
        //     ^bb(%a: f64, %b: f64, %x: f64):
        //         %0 = mulf %a, %b : f64
        //         %1 = addf %x, %0 : f64
        //         linalg.yield %1 : f64
        //     } -> memref<4x4xf32>
        // */

        // %A_pointer = sparlay.struct_access %csr_ptr[0] : 
        //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>

        // %A_index = sparlay.struct_access %csr_crd[0] : 
        //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>
        // %A_ptr_arr = sparlay.struct_access %A_CSR[0] : 
        //     !sparlay.struct<!sparlay.struct<memref<?xindex>>, !sparlay.struct<memref<?xindex>>,
        //     memref<?xf32>> to
        //     !sparlay.struct<memref<?xindex>>
        // %A_pointer = sparlay.struct_access %A_ptr_arr[0] : 
        //     !sparlay.struct<memref<?xindex>> to memref<?xindex>

        // // %A_index = sparlay.struct_access %csr_crd[0] : 
        // //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>
        // %A_idx_arr = sparlay.struct_access %A_CSR[1] : 
        //     !sparlay.struct<!sparlay.struct<memref<?xindex>>, !sparlay.struct<memref<?xindex>>,
        //     memref<?xf32>> to
        //     !sparlay.struct<memref<?xindex>> 
        // %A_index = sparlay.struct_access %A_idx_arr[0] : 
        //     !sparlay.struct<memref<?xindex>> to memref<?xindex>
        
        // %A_value = sparlay.struct_access %A_CSR[2] : 
        //     !sparlay.struct<!sparlay.struct<memref<?xindex>>, !sparlay.struct<memref<?xindex>>,
        //     memref<?xf32>> to memref<?xf32>

        // %A_ptr_mem = memref.buffer_cast %A_pointer : memref<?xindex>
        // %A_crd_mem = memref.buffer_cast %A_index : memref<?xindex>
        // %A_val_mem = memref.buffer_cast %A_value : memref<?xf32>
        // %B_mem = memref.buffer_cast %B : memref<4xf32>
        // %C_mem = memref.buffer_cast %C : memref<4xf32>

        // // SpMV compute kernel
        // scf.for %arg0 = %i0 to %i4 step %i1 {
        //     %A_crd_start = memref.load %A_pointer[%arg0] : memref<?xindex>
        //     %arg0_1 = addi %arg0, %i1 : index
        //     %A_crd_end = memref.load %A_pointer[%arg0_1] : memref<?xindex>
        //     scf.for %arg1 = %A_crd_start to %A_crd_end step %i1 {
        //         %A_op = memref.load %A_value[%arg1] : memref<?xf32>
        //         %col_idx = memref.load %A_index[%arg1] : memref<?xindex>
        //         %B_op = memref.load %B_mem[%col_idx] : memref<4xf32>
        //         %C_op = memref.load %C_mem[%arg0] : memref<4xf32>
        //         %prod = mulf %A_op, %B_op : f32
        //         %sum = addf %prod, %C_op : f32
        //         memref.store %sum, %C_mem[%arg0] : memref<4xf32>
        //     }
        // }

        // // print result
        // // call @print_memref_f32(%C_mem) : (memref<4xf32>) -> ()
        // %result = vector.transfer_read %C_mem[%i0], %f0 : memref<4xf32>, vector<4xf32>
        // vector.print %result : vector<4xf32>

        // // memref.dealloc %A_ptr_mem : memref<?xindex>
        // // memref.dealloc %A_crd_mem : memref<?xindex>
        // // memref.dealloc %A_val_mem : memref<?xf32>
        // memref.dealloc %A_mem : memref<4x4xf32>
        memref.dealloc %B_mem : memref<4xf32>
        memref.dealloc %C_mem : memref<4xf32>
        

        return
    }
}

// func private @print_memref_f32(%ptr : memref<4xf32>)
