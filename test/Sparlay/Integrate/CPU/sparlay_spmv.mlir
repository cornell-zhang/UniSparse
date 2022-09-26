// sparlay-opt test/Sparlay/Integrate/CPU/sparlay_spmv.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
//     mlir-opt --arith-bufferize --tensor-bufferize --scf-bufferize --func-bufferize --vector-bufferize \
//     --finalizing-bufferize --convert-vector-to-scf --lower-affine --convert-scf-to-cf \
//     --convert-vector-to-llvm --convert-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
//     --convert-arith-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec_all

// ./exec_all

// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#CSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j, i)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DCSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#DCSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j, i)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#trait1 = {
indexing_maps = [
    affine_map<(i,j) -> (i, j)>,  // A
    affine_map<(i,j) -> (j)>,  // B
    affine_map<(i,j) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) =+ A(i,j) * B(j)"
}

module {
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

  func.func @kernel_csc_spmv(%arg0: tensor<?x?xf32, #CSC>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSC>, tensor<?xf32>)
    outs(%argx: tensor<?xf32>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  func.func @kernel_dcsr_spmv(%arg0: tensor<?x?xf32, #DCSR>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #DCSR>, tensor<?xf32>)
    outs(%argx: tensor<?xf32>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

//  func.func @kernel_dcsc_spmv(%arg0: tensor<?x?xf32, #DCSC>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
//    %0 = linalg.generic #trait1
//    ins(%arg0, %arg1 : tensor<?xf32, #DCSC>, tensor<?xf32>)
//   outs(%argx: tensor<?xf32>) {
//    ^bb0(%a: f32, %b: f32, %x: f32):
//      %2 = arith.mulf %a, %b : f32
//      %3 = arith.addf %x, %2 : f32
//      linalg.yield %3 : f32
//    } -> tensor<?xf32>
//    return %0 : tensor<?xf32>
//  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %A_ori = sparlay.copy (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    %a0 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    %a1 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSC>
    %a2 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSR>
//    %a3 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSC>

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf32> {
      %k0 = arith.muli %i, %c4 : index
      %k1 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k1 : i32 to f32
      %t3 = tensor.insert %k into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %o0_4_4 = bufferization.alloc_tensor(%c4) : tensor<?xf32>
    %o0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o0_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %o1_4_4 = bufferization.alloc_tensor(%c4) : tensor<?xf32>
    %o1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o1_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %o2_4_4 = bufferization.alloc_tensor(%c4) : tensor<?xf32>
    %o2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o2_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

//    %o3_4_4 = bufferization.alloc_tensor(%c4) : tensor<?xf32>
//    %o3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o3_4_4) -> tensor<?xf32> {
//      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
//      scf.yield %t3 : tensor<?xf32>
//    }


    %0 = call @kernel_csr_spmv(%a0, %b, %o0) : (tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %v0 = vector.transfer_read %0[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v0 : vector<4xf32>

    %1 = call @kernel_csc_spmv(%a1, %b, %o1) : (tensor<?x?xf32, #CSC>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %v1 = vector.transfer_read %1[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v1 : vector<4xf32>

    %2 = call @kernel_dcsr_spmv(%a2, %b, %o2) : (tensor<?x?xf32, #DCSR>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %v2 = vector.transfer_read %2[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v2 : vector<4xf32>

//    %3 = call @kernel_dcsc_spmv(%a3, %b, %o3) : (tensor<?x?xf32, #DCSC>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
//    %v3 = vector.transfer_read %3[%c0], %i0: tensor<?xf32>, vector<4xf32>
//    vector.print %v3 : vector<4xf32>

    //Release the resources 
//    bufferization.dealloc_tensor %a0 : tensor<?x?xf32, #CSR>
//    bufferization.dealloc_tensor %a1 : tensor<?x?xf32, #CSC>
//    bufferization.dealloc_tensor %a2 : tensor<?x?xf32, #DCSR>
//    bufferization.dealloc_tensor %a3 : tensor<?x?xf32, #DCSC>
    bufferization.dealloc_tensor %init_256_4 : tensor<?xf32>
    bufferization.dealloc_tensor %o0_4_4 : tensor<?xf32>
    bufferization.dealloc_tensor %o1_4_4 : tensor<?xf32>
    bufferization.dealloc_tensor %o2_4_4 : tensor<?xf32>
//    bufferization.dealloc_tensor %o3_4_4 : tensor<?xf32>
    return
  }
}
