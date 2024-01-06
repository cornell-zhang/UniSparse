// mlir-opt ./sparse_tensor_csr_to_csc.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_to_csc.o

// clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmm

// ./spmm

// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
}>

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

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a0 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSR>
    %t_start1 = call @rtclock() : () -> f64
    %a1 = sparse_tensor.convert %a0 : tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    %t_end1 = call @rtclock() : () -> f64
    %t_1 = arith.subf %t_end1, %t_start1: f64
    vector.print %t_1 : f64

    //Release the resources 
    bufferization.dealloc_tensor %a1 : tensor<?x?xf32, #CSC>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o1_4_4 : tensor<?x?xf32>
    return
  }
}
