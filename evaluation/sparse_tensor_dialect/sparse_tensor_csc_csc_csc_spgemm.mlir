// mlir-opt ./sparse_tensor_csr_csr_spgemm.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o
// clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmv

// ./spgemm

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

  func.func @kernel_spgemm(%arg0: tensor<?x?xf32, #CSC>, %arg1: tensor<?x?xf32, #CSC>, %dim0: index, %dim1: index) -> tensor<?x?xf32, #CSC> {
    %0 = bufferization.alloc_tensor(%dim0, %dim1) : tensor<?x?xf32, #CSC>
    %1 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSC>, tensor<?x?xf32, #CSC>)
    outs(%0: tensor<?x?xf32, #CSC>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32, #CSC>
    return %1 : tensor<?x?xf32, #CSC>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %a0 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSC>
    %a1 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSC>
    %dim0 = tensor.dim %a0, %c0 : tensor<?x?xf32, #CSC>
    %dim1 = tensor.dim %a1, %c1 : tensor<?x?xf32, #CSC>

    // Initialize output sparse matrix.
    
    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_spgemm(%a0, %a1, %dim0, %dim1) : (tensor<?x?xf32, #CSC>, tensor<?x?xf32, #CSC>, index, index) -> tensor<?x?xf32, #CSC>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64

    %out_val = sparse_tensor.values %0 : tensor<?x?xf32, #CSC> to memref<?xf32>
    %v0 = vector.transfer_read %out_val[%c0], %i0: memref<?xf32>, vector<8xf32>
    %nnz = memref.dim %out_val, %c0 : memref<?xf32>
    vector.print %v0 : vector<8xf32>
    vector.print %nnz : index

    //Release the resources 
    bufferization.dealloc_tensor %a0 : tensor<?x?xf32, #CSC>
    bufferization.dealloc_tensor %a1 : tensor<?x?xf32, #CSC>
    return
  }
}
