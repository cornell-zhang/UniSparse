// sparlay-opt ./sparlay_csr_spmm.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmm.o

// clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmm

// ./spmm

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
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

  func.func @kernel_csr_spgemm(%arg0: tensor<?x?xf32, #CSR>, %arg1: tensor<?x?xf32, #CSR>, %dim0: index, %dim1: index) -> tensor<?x?xf32, #CSR> {
    %0 = bufferization.alloc_tensor(%dim0, %dim1) : tensor<?x?xf32, #CSR>
    %1 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>)
    outs(%0: tensor<?x?xf32, #CSR>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32, #CSR>
    return %1 : tensor<?x?xf32, #CSR>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %dim0 = tensor.dim %A_0, %c0 : tensor<?x?xf32, #COO>
    %dim1 = tensor.dim %A_1, %c1 : tensor<?x?xf32, #COO>
    %a0 = sparlay.convert (%A_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    %a1 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    // Initialize output sparse matrix.
    
    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_csr_spgemm(%a0, %a1, %dim0, %dim1) : (tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>, index, index) -> tensor<?x?xf32, #CSR>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64

    %out_val = sparlay.value %0, %c0 : tensor<?x?xf32, #CSR> to memref<?xf32>
    %v0 = vector.transfer_read %out_val[%c0], %i0: memref<?xf32>, vector<8xf32>
    %nnz = memref.dim %out_val, %c0 : memref<?xf32>
    vector.print %v0 : vector<8xf32>
    vector.print %nnz : index

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
    bufferization.dealloc_tensor %A_1 : tensor<?x?xf32, #COO>
    return
  }
}
