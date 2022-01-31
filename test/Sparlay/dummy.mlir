// RUN: sparlay-opt %s | sparlay-opt | FileCheck %s

module {
    // CHECK-LABEL: func @ops()
    func @ops() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = sparlay.foo %{{.*}} : i32
        %res = sparlay.foo %0 : i32
        %A_crd, %A_val = sparlay.pack (%A) 
            { reduce_dim = "j", padding = "none", 
            storage_order = affine_map<(i,j) -> (i,j)> } :
            tensor<4x4xf32> to 
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>, tensor<?xf32>
        %csr_ptr, %csr_crd, %csr_val = sparlay.compress (%in_crd, %in_val)
            { compress_dim = "i", storage_order = affine_map<(i,j)->(i,j)> } :
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
            tensor<?xf32> to 
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
            !sparlay.struct<tensor<?xindex>, tensor<?xindex>>,
            tensor<?xf32>
        return
    }
}
