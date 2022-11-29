// sparlay-opt ./sparlay_dia_spmm.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o dia_spmm.o

// clang++ dia_spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dia_spmm

// ./spmm

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#DIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i,i)>,
  compressMap = #sparlay.compress<trim(0,0)>
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
  func.func private @getTensorDim(!Filename, index) -> (index)
  // func.func private @kernel_dia_spmm(tensor<?x?xf32, #DIA>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // func.func private @printU64(index) -> ()

  // func.func @kernel_dia_spmm(%arg0: tensor<?x?xf32, #DIA>, %arg1: tensor<?x?xf32>, %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //   %0 = linalg.generic #trait1
  //   ins(%arg0, %arg1 : tensor<?x?xf32, #DIA>, tensor<?x?xf32>)
  //   outs(%argx: tensor<?x?xf32>) {
  //   ^bb0(%a: f32, %b: f32, %x: f32):
  //     %2 = arith.mulf %a, %b : f32
  //     %3 = arith.addf %x, %2 : f32
  //     linalg.yield %3 : f32
  //   } -> tensor<?x?xf32>
  //   return %0 : tensor<?x?xf32>
  // }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // %c4 = arith.constant 1000 : index
    // %c256 = arith.constant 200200 : index !!

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %iSize = call @getTensorDim(%fileName, %c0) : (!Filename, index) -> (index)
    %jSize = call @getTensorDim(%fileName, %c1) : (!Filename, index) -> (index)
    // call @printU64(%iSize) : (index) -> ()
    // call @printU64(%jSize) : (index) -> ()

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %a0 = sparlay.convert (%A_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DIA>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    // Initialize dense matrix.
    %init_256_4 = memref.alloc(%jSize) : memref<?xf32>
    // %init_256_4 = bufferization.alloc_tensor(%jSize) : tensor<?xf32>

    %b = scf.for %i = %c0 to %jSize step %c1 iter_args(%t = %init_256_4) -> memref<?xf32> {
      %k0 = arith.muli %i, %c1 : index
      %k2 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k2 : i32 to f32
      memref.store %k, %t[%i] : memref<?xf32>
      // %t3 = tensor.insert %k into %t[%i] : tensor<?xf32>
      scf.yield %t : memref<?xf32>
    }

    %o0_4_4 = memref.alloc(%iSize) : memref<?xf32>
    // %o0_4_4 = bufferization.alloc_tensor(%iSize) : tensor<?xf32>
    %o0 = scf.for %i = %c0 to %iSize step %c1 iter_args(%t = %o0_4_4) -> memref<?xf32> {
      memref.store %i0, %t[%i] : memref<?xf32>
      // %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t : memref<?xf32>
    }

    %t_start4 = call @rtclock() : () -> f64
    // %0 = call @kernel_dia_spmm(%a0, %b, %o0) : (tensor<?x?xf32, #DIA>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %0 = sparlay.dia_spmv(%a0, %init_256_4, %o0_4_4) : tensor<?x?xf32, #DIA>, memref<?xf32>, memref<?xf32> to memref<?xf32>
    // %mem_b = bufferization.to_memref %b: memref<?x?xf32>
    // %mem_o0 = bufferization.to_memref %o0: memref<?x?xf32>
    // %mem_0 = call @kernel_dia_spmm(%a0, %mem_b, %mem_o0) : (tensor<?x?xf32, #DIA>, memref<?x?xf32>, memref<?x?xf32>) -> memref<?x?xf32>
    // %0 = bufferization.to_tensor
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    // %v0 = vector.transfer_read %mem_b[%c0, %c0], %i0: memref<?x?xf32>, vector<4x4xf32>
    // vector.print %v0 : vector<4x4xf32>
    // %v1 = vector.transfer_read %mem_o0[%c0, %c0], %i0: memref<?x?xf32>, vector<4x4xf32>
    // vector.print %v0 : vector<4x4xf32>
    // %v0 = vector.transfer_read %0[%c0], %i0: tensor<?xf32>, vector<4xf32>
    // vector.print %v0 : vector<4xf32>
    %v0 = vector.transfer_read %0[%c0], %i0: memref<?xf32>, vector<4xf32>
    vector.print %v0 : vector<4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
    return
  }
}
