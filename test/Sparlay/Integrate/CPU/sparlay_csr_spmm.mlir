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

  func.func @kernel_csr_spmm(%arg0: tensor<?x?xf32, #CSR>, %arg1: tensor<?x?xf32>, %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #CSR>, tensor<?x?xf32>)
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
    %c256 = arith.constant 1005 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %a0 = sparlay.convert (%A_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

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

    %o0_4_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    %o0 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o0_4_4) -> tensor<?x?xf32> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %x2 : tensor<?x?xf32>
    }

    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_csr_spmm(%a0, %b, %o0) : (tensor<?x?xf32, #CSR>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v0 = vector.transfer_read %0[%c0, %c0], %i0: tensor<?x?xf32>, vector<4x4xf32>
    vector.print %v0 : vector<4x4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o0_4_4 : tensor<?x?xf32>
    return
  }
}
