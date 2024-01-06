// mlir-opt ./sparse_tensor_csc_spmm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o

// clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmm

// ./spmm

// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
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
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @kernel_csc_spmm(%arg0: tensor<?x?xf32, #CSC>, %arg1: tensor<?x?xf32>, %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSC>, tensor<?x?xf32>)
    outs(%argx: tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1000 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start1 = call @rtclock() : () -> f64
    %a1 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSC>
    %c256 = tensor.dim %a1, %c1 : tensor<?x?xf32, #CSC>
    %t_end1 = call @rtclock() : () -> f64
    %t_1 = arith.subf %t_end1, %t_start1: f64
    vector.print %t_1 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf32> {
      %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %k0 = arith.muli %i, %c4 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f32
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %b2 : tensor<?x?xf32>
    }

    %o1_4_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    %o1 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o1_4_4) -> tensor<?x?xf32> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %x2 : tensor<?x?xf32>
    }

    %t_start5 = call @rtclock() : () -> f64
    %1 = call @kernel_csc_spmm(%a1, %b, %o1) : (tensor<?x?xf32, #CSC>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %t_end5 = call @rtclock() : () -> f64
    %t_5 = arith.subf %t_end5, %t_start5: f64
    vector.print %t_5 : f64
    %v1 = vector.transfer_read %1[%c0, %c0], %i0: tensor<?x?xf32>, vector<4x4xf32>
    vector.print %v1 : vector<4x4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %a1 : tensor<?x?xf32, #CSC>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o1_4_4 : tensor<?x?xf32>
    return
  }
}
