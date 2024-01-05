// mlir-opt ./sparse_tensor_csr_csr_csr_spgemm_F64.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_csr_csr_spgemm_F64.o
// clang++ csr_csr_csr_spgemm_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVM_ROOT/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_csr_csr_spgemm_F64

// ./csr_csr_csr_spgemm_F64

!Filename = !llvm.ptr<i8>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
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

  func.func @kernel_csr_spgemm(%arg0: tensor<?x?xf64, #CSR>, %arg1: tensor<?x?xf64, #CSR>, %dim0: index, %dim1: index) -> tensor<?x?xf64, #CSR> {
    %0 = bufferization.alloc_tensor(%dim0, %dim1) : tensor<?x?xf64, #CSR>
    %1 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
    outs(%0: tensor<?x?xf64, #CSR>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64, #CSR>
    return %1 : tensor<?x?xf64, #CSR>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %a0 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR>
    %a1 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR>
    %dim0 = tensor.dim %a0, %c0 : tensor<?x?xf64, #CSR>
    %dim1 = tensor.dim %a1, %c1 : tensor<?x?xf64, #CSR>

    // Initialize output sparse matrix.
    
    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_csr_spgemm(%a0, %a1, %dim0, %dim1) : (tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>, index, index) -> tensor<?x?xf64, #CSR>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
     
    %out_val = sparse_tensor.values %0 : tensor<?x?xf64, #CSR> to memref<?xf64>
    %v0 = vector.transfer_read %out_val[%c0], %i0: memref<?xf64>, vector<8xf64>
    %nnz = memref.dim %out_val, %c0 : memref<?xf64>
    vector.print %v0 : vector<8xf64>
    vector.print %nnz : index

    //Release the resources 
    bufferization.dealloc_tensor %a0 : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %a1 : tensor<?x?xf64, #CSR>
    return
  }
}
