\\ sparlay-opt ./sparlay_dcsc_spmv.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
\\ mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
\\ -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
\\ -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
\\ -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
\\ -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o

// clang++ spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmv

// ./spmv

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
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
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @kernel_dcsc_spmv(%arg0: tensor<?x?xf32, #DCSC>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #DCSC>, tensor<?xf32>)
    outs(%argx: tensor<?xf32>) {
    ^bb0(%a: f32, %b: f32, %x: f32):
      %2 = arith.mulf %a, %b : f32
      %3 = arith.addf %x, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 1005 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start3 = call @rtclock() : () -> f64
    %A_3 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %a3 = sparlay.convert (%A_3): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSC>
    %t_end3 = call @rtclock() : () -> f64
    %t_3 = arith.subf %t_end3, %t_start3: f64
    vector.print %t_3 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf32> {
      %k0 = arith.muli %i, %c1 : index
      %k1 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k1 : i32 to f32
      %t3 = tensor.insert %k into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %o3_4_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %o3 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o3_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %t_start7 = call @rtclock() : () -> f64
    %3 = call @kernel_dcsc_spmv(%a3, %b, %o3) : (tensor<?x?xf32, #DCSC>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %t_end7 = call @rtclock() : () -> f64
    %t_7 = arith.subf %t_end7, %t_start7: f64
    vector.print %t_7 : f64
    %v3 = vector.transfer_read %3[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v3 : vector<4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_3 : tensor<?x?xf32, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?xf32>
//    bufferization.dealloc_tensor %o3_4_4 : tensor<?xf32>
    return
  }
}
