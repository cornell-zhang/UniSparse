// unisparse-opt ./unisparse_csr_spmv_F64.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_spmv_F64.o

// clang++ csr_spmv_F64.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmv_F64

// ./csr_spmv_F64

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
}>

#CSR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(1,1)>
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
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @kernel_csr_spmv(%arg0: tensor<?x?xf64, #CSR>, %arg1: tensor<?xf64>, %argx: tensor<?xf64>) -> tensor<?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #CSR>, tensor<?xf64>)
    outs(%argx: tensor<?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf64, #COO>
    %c256 = tensor.dim %A_0, %c1 : tensor<?x?xf64, #COO>
    %a0 = unisparse.convert (%A_0): tensor<?x?xf64, #COO> to tensor<?x?xf64, #CSR>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256) : tensor<?xf64>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf64> {
      %k0 = arith.muli %i, %c1 : index
      %k1 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k1 : i32 to f64
      %t3 = tensor.insert %k into %t[%i] : tensor<?xf64>
      scf.yield %t3 : tensor<?xf64>
    }

    %o0_4_4 = bufferization.alloc_tensor(%c256) : tensor<?xf64>
    %o0 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o0_4_4) -> tensor<?xf64> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf64>
      scf.yield %t3 : tensor<?xf64>
    }

    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_csr_spmv(%a0, %b, %o0) : (tensor<?x?xf64, #CSR>, tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v0 = vector.transfer_read %0[%c0], %i0: tensor<?xf64>, vector<4xf64>
    vector.print %v0 : vector<4xf64>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf64, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?xf64>
//    bufferization.dealloc_tensor %o0_4_4 : tensor<?xf64>
    return
  }
}
