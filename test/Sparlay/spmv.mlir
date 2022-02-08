// RUN: sparlay-opt %s | sparlay-opt | FileCheck %s

module {
    // CHECK-LABEL: func @spmv()
    func @spmv() {
        
        // %A_1d = tensor.from_elements %f0, %f1, %f0, %f1, %f1, %f0, %f0, %f0, %f0, %f0, %f0, %f1, %f0, %f1, %f0, %f0 : tensor<16xf32>
        // %shape = tensor.from_elements %i4, %i4 : tensor<2xi32>
        // %A = tensor.reshape %A_1d(%shape) : (tensor<16xf32> , tensor<2xi32>) -> tensor<4x4xf32>
        %i0 = constant 0: index
        %i1 = constant 1: index
        %i4 = constant 4: index
        %f0 = constant 0.0e+00 : f32
        %A = constant dense<[[0.0e+00, 1.0e+00, 0.0e+00, 1.0e+00],
                             [1.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                             [0.0e+00, 0.0e+00, 0.0e+00, 1.0e+00],
                             [0.0e+00, 1.0e+00, 0.0e+00, 0.0e+00]]> : tensor<4x4xf32>
        // %B = tensor.from_elements %f3, %f2, %f1, %f4 : tensor<4xf32>
        %B = constant dense<[3.0e+00, 2.0e+00, 1.0e+00, 4.0e+00]> : tensor<4xf32>
        %C = constant dense<[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00]> : tensor<4xf32>
        
        // %A_crd, %A_val = sparlay.pack (%A) 
        //     { reduce_dim = "j", padding = "none", 
        //     storage_order = affine_map<(i,j) -> (i,j)> } :
        //     tensor<4x4xf32> to 
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
        %A_COO = sparlay.pack (%A) 
            { reduce_dim = "j", padding = "none", 
            storage_order = affine_map<(i,j) -> (i,j)> } :
            tensor<4x4xf32> to 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>>
        // %csr_ptr, %csr_crd, %csr_val = sparlay.compress (%A_crd, %A_val)
        //     { compress_dim = "i", storage_order = affine_map<(i,j)->(i,j)> } :
        //     !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
        //     tensor<?xf32> to 
        //     !sparlay.struct<tensor<?xindex>>,
        //     !sparlay.struct<tensor<?xindex>>,
        //     tensor<?xf32>  
        %A_CSR = sparlay.compress (%A_COO)
            { compress_dim = "i", storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<!sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>> to 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>>, !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>>

        // %A_pointer = sparlay.struct_access %csr_ptr[0] : 
        //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>

        // %A_index = sparlay.struct_access %csr_crd[0] : 
        //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>
        %A_ptr_arr = sparlay.struct_access %A_CSR[0] : 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>>, !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>> to
            !sparlay.struct<tensor<?xindex>>
        %A_pointer = sparlay.struct_access %A_ptr_arr[0] : 
            !sparlay.struct<tensor<?xindex>> to tensor<?xindex>

        // %A_index = sparlay.struct_access %csr_crd[0] : 
        //     !sparlay.struct<tensor<?xindex>> to tensor<?xindex>
        %A_idx_arr = sparlay.struct_access %A_CSR[1] : 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>>, !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>> to
            !sparlay.struct<tensor<?xindex>> 
        %A_index = sparlay.struct_access %A_idx_arr[0] : 
            !sparlay.struct<tensor<?xindex>> to tensor<?xindex>
        
        %A_value = sparlay.struct_access %A_CSR[2] : 
            !sparlay.struct<!sparlay.struct<tensor<?xindex>>, !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>> to tensor<?xf32>

        %A_ptr_mem = memref.buffer_cast %A_pointer : memref<?xindex>
        %A_crd_mem = memref.buffer_cast %A_index : memref<?xindex>
        %A_val_mem = memref.buffer_cast %A_value : memref<?xf32>
        %B_mem = memref.buffer_cast %B : memref<4xf32>
        %C_mem = memref.buffer_cast %C : memref<4xf32>

        // SpMV compute kernel
        scf.for %arg0 = %i0 to %i4 step %i1 {
            %A_crd_start = memref.load %A_ptr_mem[%arg0] : memref<?xindex>
            %arg0_1 = addi %arg0, %i1 : index
            %A_crd_end = memref.load %A_ptr_mem[%arg0_1] : memref<?xindex>
            scf.for %arg1 = %A_crd_start to %A_crd_end step %i1 {
                %A_op = memref.load %A_val_mem[%arg1] : memref<?xf32>
                %col_idx = memref.load %A_crd_mem[%arg1] : memref<?xindex>
                %B_op = memref.load %B_mem[%col_idx] : memref<4xf32>
                %C_op = memref.load %C_mem[%arg0] : memref<4xf32>
                %prod = mulf %A_op, %B_op : f32
                %sum = addf %prod, %C_op : f32
                memref.store %sum, %C_mem[%arg0] : memref<4xf32>
            }
        }

        // print result
        // call @print_memref_f32(%C_mem) : (memref<4xf32>) -> ()
        %result = vector.transfer_read %C_mem[%i0], %f0 : memref<4xf32>, vector<4xf32>
        vector.print %result : vector<4xf32>

        memref.dealloc %A_ptr_mem : memref<?xindex>
        memref.dealloc %A_crd_mem : memref<?xindex>
        memref.dealloc %A_val_mem : memref<?xf32>
        memref.dealloc %B_mem : memref<4xf32>
        memref.dealloc %C_mem : memref<4xf32>
        

        return
    }
}

// func private @print_memref_f32(%ptr : memref<4xf32>)
