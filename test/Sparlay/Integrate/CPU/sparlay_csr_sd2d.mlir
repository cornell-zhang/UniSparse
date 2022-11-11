// sparlay-opt ./sparlay_csr_sd2d.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o sd2d.o

// clang++ sd2d.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o sd2d

// ./sd2d

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
    affine_map<(i,j) -> (i, j)>,  // A
    affine_map<(i,j) -> (i, j)>,  // B
    affine_map<(i,j) -> (i, j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) + B(i, j)"
}

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @csr_sparse_dense_add(%arg0: tensor<?x?xf64, #CSR>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSR>, tensor<?x?xf64>)
    outs(%argx: tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %0 = arith.addf %a, %b : f64
      linalg.yield %0 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 259789 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf64, #COO>
    %A_ori = sparlay.copy (%A_1): tensor<?x?xf64, #COO> to tensor<?x?xf64, #COO>
    %a0 = sparlay.convert (%A_ori): tensor<?x?xf64, #COO> to tensor<?x?xf64, #CSR>

    // Initialize dense matrix.
    %init_4_256 = bufferization.alloc_tensor(%c256, %c256) : tensor<?x?xf64>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_4_256) -> tensor<?x?xf64> {
      %b2 = scf.for %j = %c0 to %c256 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %k0 = arith.muli %i, %c1 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f64
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %b2 : tensor<?x?xf64>
    }

    %o0 = bufferization.alloc_tensor(%c256, %c256) : tensor<?x?xf64>

    %0 = call @csr_sparse_dense_add(%a0, %b, %o0) : (tensor<?x?xf64, #CSR>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %v0 = vector.transfer_read %0[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v0 : vector<4x4xf64>

    //Release the resources 
    bufferization.dealloc_tensor %A_1 : tensor<?x?xf64, #COO>
//    bufferization.dealloc_tensor %init_4_256 : tensor<?x?xf64>
//    bufferization.dealloc_tensor %o0 : tensor<?x?xf64>
    
    return
  }
}
