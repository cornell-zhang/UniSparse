// sparlay-opt test/Sparlay/Integrate/CPU/sparlay_spmm.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
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
    affine_map<(i,j,k) -> (i, k)>,  // A
    affine_map<(i,j,k) -> (k, j)>,  // B
    affine_map<(i,j,k) -> (i, j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) =+ A(i,k) * B(k, j)"
}

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @kernel_csr_spmm(%arg0: tensor<?x?xf64, #CSR>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSR>, tensor<?x?xf64>)
    outs(%argx: tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @kernel_csc_spmm(%arg0: tensor<?x?xf64, #CSC>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSC>, tensor<?x?xf64>)
    outs(%argx: tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @kernel_dcsr_spmm(%arg0: tensor<?x?xf64, #DCSR>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #DCSR>, tensor<?x?xf64>)
    outs(%argx: tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

//  func.func @kernel_dcsc_spmm(%arg0: tensor<?x?xf64, #DCSC>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
//    %0 = linalg.generic #trait1
//    ins(%arg0, %arg1 : tensor<?x?xf64, #DCSC>, tensor<?x?xf64>)
//   outs(%argx: tensor<?x?xf64>) {
//    ^bb0(%a: f64, %b: f64, %x: f64):
//      %2 = arith.mulf %a, %b : f64
//      %3 = arith.addf %x, %2 : f64
//      linalg.yield %3 : f64
//    } -> tensor<?x?xf64>
//    return %0 : tensor<?x?xf64>
//  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf64, #COO>
    %A_2 = sparlay.copy (%A_1): tensor<?x?xf64, #COO> to tensor<?x?xf64, #COO>
    %A_3 = sparlay.copy (%A_1): tensor<?x?xf64, #COO> to tensor<?x?xf64, #COO>
    %a0 = sparlay.convert (%A_1): tensor<?x?xf64, #COO> to tensor<?x?xf64, #CSR>
    %a1 = sparlay.convert (%A_2): tensor<?x?xf64, #COO> to tensor<?x?xf64, #CSC>
    %a2 = sparlay.convert (%A_3): tensor<?x?xf64, #COO> to tensor<?x?xf64, #DCSR>
//    %a3 = sparlay.convert (%A_1): tensor<?x?xf64, #COO> to tensor<?x?xf64, #DCSC>

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf64>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf64> {
      %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %k0 = arith.muli %i, %c4 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f64
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %b2 : tensor<?x?xf64>
    }

    %o0_4_4 = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf64>
    %o0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o0_4_4) -> tensor<?x?xf64> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %x2 : tensor<?x?xf64>
    }

    %o1_4_4 = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf64>
    %o1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o1_4_4) -> tensor<?x?xf64> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %x2 : tensor<?x?xf64>
    }

    %o2_4_4 = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf64>
    %o2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o2_4_4) -> tensor<?x?xf64> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %x2 : tensor<?x?xf64>
    }

//    %o3_4_4 = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf64>
//    %o3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %o3_4_4) -> tensor<?x?xf64> {
//      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
//        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf64>
//        scf.yield %t3 : tensor<?x?xf64>
//      }
//      scf.yield %x2 : tensor<?x?xf64>
//    }

    %0 = call @kernel_csr_spmm(%a0, %b, %o0) : (tensor<?x?xf64, #CSR>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %v0 = vector.transfer_read %0[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v0 : vector<4x4xf64>

    %1 = call @kernel_csc_spmm(%a1, %b, %o1) : (tensor<?x?xf64, #CSC>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %v1 = vector.transfer_read %1[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v1 : vector<4x4xf64>

    %2 = call @kernel_dcsr_spmm(%a2, %b, %o2) : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %v2 = vector.transfer_read %2[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v2 : vector<4x4xf64>

//    %3 = call @kernel_dcsc_spmm(%a3, %b, %o3) : (tensor<?x?xf64, #DCSC>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
//    %v3 = vector.transfer_read %3[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
//    vector.print %v3 : vector<4x4xf64>

    //Release the resources 
//    bufferization.dealloc_tensor %a0 : tensor<?x?xf64, #CSR>
//    bufferization.dealloc_tensor %a1 : tensor<?x?xf64, #CSC>
//    bufferization.dealloc_tensor %a2 : tensor<?x?xf64, #DCSR>
//    bufferization.dealloc_tensor %a3 : tensor<?x?xf64, #DCSC>
    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf64>
    bufferization.dealloc_tensor %o0_4_4 : tensor<?x?xf64>
    bufferization.dealloc_tensor %o1_4_4 : tensor<?x?xf64>
    bufferization.dealloc_tensor %o2_4_4 : tensor<?x?xf64>
//    bufferization.dealloc_tensor %o3_4_4 : tensor<?x?xf64>
    return
  }
}
