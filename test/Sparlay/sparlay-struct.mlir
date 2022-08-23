// RUN: sparlay-opt %s -lower-struct -dce | FileCheck %s

// CHECK-LABEL: func @main(
func.func @main() {
    %A_ptr = arith.constant dense<[0, 2, 5, 5, 7]> : tensor<5xindex>
    %A_crd_i = arith.constant dense<[0, 1, 0, 1, 3, 1, 3]> : tensor<7xindex>
    %A_crd_j = arith.constant dense<[0, 0, 1, 3, 3, 4, 4]> : tensor<7xindex>
    %A_val = arith.constant dense<[1.0e+00, 5.0e+00, 7.0e+00, 3.0e+00, 4.0e+00, 2.0e+00, 6.0e+00]> : tensor<7xf32>
    %A_ptr_mem = bufferization.to_memref %A_ptr : memref<5xindex>
    %A_crd_i_mem= bufferization.to_memref %A_crd_i : memref<7xindex>
    %A_crd_j_mem = bufferization.to_memref %A_crd_j : memref<7xindex>
    %A_val_mem = bufferization.to_memref %A_val : memref<7xf32>
    // CHECK: {{.*}} bufferization.to_memref {{.*}}
    // CHECK-NEXT: {{.*}} bufferization.to_memref {{.*}}
    // CHECK-NEXT: {{.*}} bufferization.to_memref {{.*}}
    // CHECK-NEXT: {{.*}} bufferization.to_memref {{.*}}
    // CHECK-NEXT: return

    // Inst 0:
    %A_COO_crd = sparlay.struct_construct (%A_crd_i_mem, %A_crd_j_mem) : memref<7xindex>, memref<7xindex> to    
        !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>
    
    // Inst 1:
    %A_COO = sparlay.struct_construct (%A_COO_crd, %A_val_mem) : 
        !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>, memref<7xf32> to
        !sparlay.struct< [4,6], !sparlay.struct<memref<7xindex>, memref<7xindex>, "crd", (i,j)->(i,j)>, memref<7xf32> >
    
    // Inst 2:
    %A_CSR_ptr = sparlay.struct_construct (%A_ptr_mem) : memref<5xindex> to    
        !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>
    
    // Inst 3:
    %A_CSR_crd = sparlay.struct_construct (%A_crd_j_mem) : memref<7xindex> to    
        !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)>
    
    // Inst 4:
    %A_CSR = sparlay.struct_construct (%A_CSR_ptr, %A_CSR_crd, %A_val_mem) : 
        !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>,
        !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)>, memref<7xf32> to
        !sparlay.struct< [4,6], !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)> , memref<7xf32> >
    
    // Inst 5: fold with Inst 4, replace result with %A_CSR_ptr
    %access_A_CSR_ptr = sparlay.struct_access %A_CSR[0] : 
        !sparlay.struct< [4,6], !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)> , memref<7xf32> > to
        !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>
    
    // Inst 6: fold with Inst 5 and Inst 2, replace result with %A_ptr_mem
    %access_A_CSR_ptr_mem = sparlay.struct_access %access_A_CSR_ptr[0] :
        !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)> to memref<5xindex>
    
    // Inst 7: fold with Inst 4, replace result with %A_CSR_crd
    %access_A_CSR_crd = sparlay.struct_access %A_CSR[1] : 
        !sparlay.struct< [4,6], !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)> , memref<7xf32> > to
        !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)>
    
    // Inst 8: fold with Inst 7 and Inst 3, replace result with %A_crd_j_mem
    %access_A_CSR_crd_mem = sparlay.struct_access %access_A_CSR_crd[0] :
        !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)> to memref<7xindex>
    
    // Inst 9: fold with Inst 4, replace result with %A_val_mem
    %access_A_val_mem = sparlay.struct_access %A_CSR[2] : 
        !sparlay.struct< [4,6], !sparlay.struct<memref<5xindex>, "ptr", (i,j)->(j)>,
                         !sparlay.struct<memref<7xindex>, "crd", (i,j)->(j)> , memref<7xf32> > to
        memref<7xf32>
           
    return
}

