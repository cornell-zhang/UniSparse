// sparlay-opt ./decompose-BDIA.mlir -lower-struct-convert -lower-struct -dce -sparlay-codegen -lower-format-conversion | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o bdia_spmv.o
    
// clang++ bdia_spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o bdia_spmv

// ./bdia_spmv

// RUN: sparlay-opt %s -lower-struct-convert -lower-struct -dce -lower-format-conversion | FileCheck %s


!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#BDIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i floordiv 50, j minus i, i mod 50)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#amap_0 = affine_map<(i,j) -> (i, j)>

#trait1 = {
indexing_maps = [
    #amap_0,  // A
    affine_map<(i,j) -> (j)>,  // B
    affine_map<(i,j) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) =+ A(i,j) * B(j)"
}

module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @kernel_csr_spmv(%arg0: tensor<?x?xf32, #CSR>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSR>, tensor<?xf32>)
    outs(%argx: tensor<?xf32>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  func.func @main() {
    %c0 = arith.constant 0: index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0: f32
    %f05 = arith.constant 0.5: f32
    %i1 = arith.constant 1: i32
    %blockSize = arith.constant 100: i32
    %thres_1 = arith.constant 0.5: f32
    %c1000 = arith.constant 1000 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<?x?xf32, #COO>
    %dim1 = tensor.dim %A_1, %c1 : tensor<?x?xf32, #COO>
    %dim0 = tensor.dim %A_1, %c0 : tensor<?x?xf32, #COO>
    // %thres_1 = arith.constant dense<[0.5]>: tensor<1xf32>
    // %thres_2 = bufferization.alloc_tensor () copy(%thres_1): tensor<1xf32>
    // %thres = bufferization.to_memref %thres_2: memref<1xf32>

    %t_start0 = call @rtclock() : () -> f64
    %S_1 = sparlay.decompose_BDIA %A_1, %blockSize, %thres_1 : tensor<?x?xf32, #COO>, i32, f32 to 
          !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#BDIA> >
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64
    
    %B_0 = sparlay.struct_access %S_1[0]: 
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#BDIA> >
          to  tensor<?x?xf32, #COO>
    %B_1 = sparlay.struct_access %S_1[1]:
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#BDIA> >
          to  tensor<?x?xf32, #BDIA>

    %D_0 = sparlay.convert(%B_0) : tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    
    %init_256_4 = bufferization.alloc_tensor(%dim1, %c1000) : tensor<?x?xf32>
    %b = scf.for %i = %c0 to %dim1 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf32> {
      %b2 = scf.for %j = %c0 to %c1000 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %k0 = arith.muli %i, %c1000 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f32
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %b2 : tensor<?x?xf32>
    }
    // %init_256_4 = bufferization.alloc_tensor(%dim1, %c1000) : tensor<?x?xf32>
    // %tensor_B = tensor.insert %f05 into %init_256_4[%c0] : tensor<?xf32>
    // %dim1_1 = arith.subi %dim1, %c1 : index
    // %i_dim1_1 = arith.index_cast %dim1_1 : index to i32
    // %f_dim1_1 = arith.sitofp %i_dim1_1 : i32 to f32
    // %elm = arith.divf %f05, %f_dim1_1 : f32
    // %b = scf.for %i = %c1 to %dim1 step %c1 iter_args(%t = %tensor_B) -> tensor<?xf32> {
    //    %b2 = scf.for %j = %c0 to %c1000 step %c1 iter_args(%t2 = %t) -> memref<?x?xf32> {
    //     %t3 = tensor.insert %elm into %t[%i] : tensor<?xf32>
    //     scf.yield %t3 : tensor<?xf32>
    //    }
    // }
    
    // %o0 = bufferization.alloc_tensor(%dim0) : tensor<?xf32>
    // %o00 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o0) -> tensor<?xf32> {
    //   %t3 = tensor.insert %f0 into %t[%i] : tensor<?xf32>
    //   scf.yield %t3 : tensor<?xf32>
    // }
    // %o0_4_4 = memref.alloc(%dim0, %c4) : memref<?x?xf32>
    // %o0 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o0_4_4) -> memref<?x?xf32> {
    //   %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> memref<?x?xf32> {
    //     memref.store %i0, %t2[%i, %j] : memref<?x?xf32>
    //     scf.yield %t2 : memref<?x?xf32>
    //   }
    //   scf.yield %x2 : memref<?x?xf32>
    // }
    %o1 = bufferization.alloc_tensor(%dim0, %c1000) : tensor<?x?xf32>
    %o11 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o1) -> tensor<?x?xf32> {
      %x2 = scf.for %j = %c0 to %c1000 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %t3 = tensor.insert %f0 into %t[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %x2 : tensor<?x?xf32>
    }
    // %o2 = bufferization.alloc_tensor(%dim0) : tensor<?xf32>
    // %o22 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o2) -> tensor<?xf32> {
    //   %t3 = tensor.insert %f0 into %t[%i] : tensor<?xf32>
    //   scf.yield %t3 : tensor<?xf32>
    // }
    
    %t_start4 = call @rtclock() : () -> f64
    // CSR SpMV
    // %result0 = call @kernel_csr_spmv(%D_0, %b, %o00) : (tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    // %t_end1 = call @rtclock() : () -> f64
    // block DIA SpMV
    %result1 = sparlay.bdia_spmm %D_0, %B_1, %b, %o1: 
      tensor<?x?xf32, #CSR>, tensor<?x?xf32,#BDIA>, tensor<?x?xf32>, tensor<?x?xf32> to memref<?x?xf32>
    // %t_end2 = call @rtclock() : () -> f64
    // %output = linalg.elemwise_binary ins(%result0, %result1: tensor<?xf32>, tensor<?xf32>)
    //                           outs(%o2: tensor<?xf32>) -> tensor<?xf32>
    %t_end4 = call @rtclock() : () -> f64
    // %t_1 = arith.subf %t_end1, %t_start4: f64
    // %t_2 = arith.subf %t_end2, %t_end1: f64
    // %t_4 = arith.subf %t_end4, %t_end2: f64
    %t_5 = arith.subf %t_end4, %t_start4: f64
    // vector.print %t_1 : f64
    // vector.print %t_2 : f64
    // vector.print %t_4 : f64
    vector.print %t_5 : f64
    // %v0 = vector.transfer_read %result0[%c0], %f0: tensor<?xf32>, vector<4xf32>
    // vector.print %v0 : vector<4xf32>
    %v1 = vector.transfer_read %result1[%c0, %c0], %f0: memref<?x?xf32>, vector<4x4xf32>
    vector.print %v1 : vector<4x4xf32>
    // %v2 = vector.transfer_read %output[%c0], %f0: tensor<?xf32>, vector<4xf32>
    // vector.print %v2 : vector<4xf32>
    bufferization.dealloc_tensor %A_1 : tensor<?x?xf32, #COO>
    bufferization.dealloc_tensor %B_1 : tensor<?x?xf32, #BDIA>
    sparlay.release %S_1: !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#BDIA> >
    // bufferization.dealloc_tensor %B_0 : tensor<?x?xf32, #COO>
    // bufferization.dealloc_tensor %o1 : tensor<?xf32>
    // bufferization.dealloc_tensor %result0 : tensor<?xf32>
    // bufferization.dealloc_tensor %output : tensor<?xf32>

    return
  }
}