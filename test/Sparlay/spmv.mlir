// RUN: sparlay-opt %s | sparlay-opt | FileCheck %s

module {
    // CHECK-LABEL: func @spmv()
    func @spmv() {
        %f0 = constant 0.0e+00 : f32
        %f1 = constant 1.0e+00 : f32
        %f2 = constant 2.0e+00 : f32
        %f3 = constant 3.0e+00 : f32
        %f4 = constant 4.0e+00 : f32
        %i4 = constant 4 : i32
        %i0 = constant 0 : index
        // %A_1d = tensor.from_elements %f0, %f1, %f0, %f1, %f1, %f0, %f0, %f0, %f0, %f0, %f0, %f1, %f0, %f1, %f0, %f0 : tensor<16xf32>
        // %shape = tensor.from_elements %i4, %i4 : tensor<2xi32>
        // %A = tensor.reshape %A_1d(%shape) : (tensor<16xf32> , tensor<2xi32>) -> tensor<4x4xf32>
        %A = constant dense<[[0.0e+00, 1.0e+00, 0.0e+00, 1.0e+00],
                             [1.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                             [0.0e+00, 0.0e+00, 0.0e+00, 1.0e+00],
                             [0.0e+00, 1.0e+00, 0.0e+00, 0.0e+00]]> : tensor<4x4xf32>
        // %B = tensor.from_elements %f3, %f2, %f1, %f4 : tensor<4xf32>
        %B = constant dense<[3.0e+00, 2.0e+00, 1.0e+00, 4.0e+00]> : tensor<4xf32>
        
        %A_crd, %A_val = sparlay.pack (%A) 
            { reduce_dim = "j", padding = "none", 
            storage_order = affine_map<(i,j) -> (i,j)> } :
            tensor<4x4xf32> to 
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
        %csr_ptr, %csr_crd, %csr_val = sparlay.compress (%A_crd, %A_val)
            { compress_dim = "i", storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
            tensor<?xf32> to 
            !sparlay.struct<tensor<?xindex>>,
            !sparlay.struct<tensor<?xindex>>,
            tensor<?xf32>   

        %A_pointer = sparlay.struct_access %csr_ptr[%i0] : 
            !sparlay.struct<tensor<?xindex>> to tensor<?xindex>

        %A_index = sparlay.struct_access %csr_crd[%i0] : 
            !sparlay.struct<tensor<?xindex>> to tensor<?xindex>

        // %A_ptr = memref.buffer_cast %csr_p

        return
    }
}
