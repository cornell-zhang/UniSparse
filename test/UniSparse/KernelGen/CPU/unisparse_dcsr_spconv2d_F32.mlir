// unisparse-opt ./unisparse_dcsr_spmm_F32.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o dcsr_spmm_F32.o

// clang++ dcsr_spmm_F32.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dcsr_spmm_F32

// ./dcsr_spmm_F32

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
}>

#DCSR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(0,1)>
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

  func.func @conv2d(%input:  tensor<8x8xi32>,
               %filter: tensor<3x3xi32, #DCSR>,
               %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32, #DCSR>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1000 : index

    %filter = arith.constant dense<[
      [  1.0,  0.0, -1.0 ],
      [  0.0,  0.0,  0.0 ],
      [ -1.0,  0.0,  1.0 ]
    ]> : tensor<3x3xf32>
    %sparse_filter = unisparse.convert (%filter) : tensor<3x3xf32> to tensor<3x3xf32, #DCSR>

    %input = arith.constant dense<[
      [  1.0,  2.0,  3.0,  4.0,  0.0,  6.0,  7.0,  8.0 ],
      [  2.0,  2.0,  4.0,  4.0,  0.0,  0.0,  6.0,  8.0 ],
      [  2.0,  2.0,  4.0,  4.0,  0.0,  0.0,  6.0,  8.0 ],
      [  2.0,  2.0,  3.0,  4.0,  0.0,  0.0,  7.0,  8.0 ],
      [  1.0,  3.0,  3.0,  4.0,  0.0,  0.0,  6.0,  8.0 ],
      [  3.0,  2.0,  3.0,  4.0,  0.0,  0.0,  7.0,  8.0 ],
      [  1.0,  3.0,  3.0,  4.0,  3.0,  6.0,  6.0,  8.0 ],
      [  1.0,  3.0,  3.0,  4.0,  3.0,  0.0,  7.0,  8.0 ]
    ]> : tensor<8x8xf32>

    // %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    // %t_start2 = call @rtclock() : () -> f64
    // %A_2 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    // %c256 = tensor.dim %A_2, %c1 : tensor<?x?xf32, #COO>
    // %a2 = unisparse.convert (%A_2): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSR>
    // %t_end2 = call @rtclock() : () -> f64
    // %t_2 = arith.subf %t_end2, %t_start2: f64
    // vector.print %t_2 : f64

    // // Initialize dense matrix.
    // %init_256_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    // %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf32> {
    //   %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
    //     %k0 = arith.muli %i, %c4 : index
    //     %k1 = arith.addi %j, %k0 : index
    //     %k2 = arith.index_cast %k1 : index to i32
    //     %k = arith.sitofp %k2 : i32 to f32
    //     %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf32>
    //     scf.yield %t3 : tensor<?x?xf32>
    //   }
    //   scf.yield %b2 : tensor<?x?xf32>
    // }

    // %o2_4_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    // %o2 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o2_4_4) -> tensor<?x?xf32> {
    //   %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
    //     %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf32>
    //     scf.yield %t3 : tensor<?x?xf32>
    //   }
    //   scf.yield %x2 : tensor<?x?xf32>
    // }

    %t_start6 = call @rtclock() : () -> f64
    %2 = call @conv2d(%input, %sparse_filter, %output)
          : (tensor<8x8xf32>,
          tensor<3x3xf32, #DCSR>, tensor<6x6xf32>) -> tensor<6x6xf32>
    // %2 = call @kernel_dcsr_spmm(%a2, %b, %o2) : (tensor<?x?xf32, #DCSR>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %t_end6 = call @rtclock() : () -> f64
    %t_6 = arith.subf %t_end6, %t_start6: f64
    vector.print %t_6 : f64
    %v2 = vector.transfer_read %2[%c0, %c0], %i0: tensor<?x?xf32>, vector<4x4xf32>
    vector.print %v2 : vector<4x4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %sparse_filter : tensor<3x3xf32, #DCSR>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o2_4_4 : tensor<?x?xf32>
    return
  }
}
