// unisparse-opt ./unisparse_coo_spmv_F32.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o coo_spmv_F32.o

// clang++ coo_spmv_F32.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o coo_spmv_F32

// ./coo_spmv_F32

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
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
  func.func private @printU64(index) -> ()

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %A_0 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %dim0 = call @getTensorDim(%fileName, %c0) : (!Filename, index) -> (index)
    %dim1 = call @getTensorDim(%fileName, %c1) : (!Filename, index) -> (index)
    call @printU64(%dim0) : (index) -> ()
    call @printU64(%dim1) : (index) -> ()
    // %dim0 = tensor.dim %A_0, %c0 : tensor<?x?xf32, #COO>
    // %dim1 = tensor.dim %A_0, %c1 : tensor<?x?xf32, #COO>

    // Initialize vector matrix.
    %init_256_4 = memref.alloc(%dim1) : memref<?xf32>
    %b = scf.for %i = %c0 to %dim1 step %c1 iter_args(%t = %init_256_4) -> memref<?xf32> {
      %k0 = arith.muli %i, %c1 : index
      %k1 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k1 : i32 to f32
      memref.store %k, %t[%i] : memref<?xf32>
      scf.yield %t : memref<?xf32>
    }

    %o0_4_4 = memref.alloc(%dim0) : memref<?xf32>
    %o0 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o0_4_4) -> memref<?xf32> {
      memref.store %i0, %t[%i] : memref<?xf32>
      scf.yield %t : memref<?xf32>
    }

    %t_start4 = call @rtclock() : () -> f64
    %0 = unisparse.coo_spmv %A_0, %init_256_4, %o0_4_4: tensor<?x?xf32, #COO>, memref<?xf32>, memref<?xf32> to memref<?xf32>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v1 = vector.transfer_read %init_256_4[%c0], %i0: memref<?xf32>, vector<4xf32>
    vector.print %v1 : vector<4xf32>
    %v0 = vector.transfer_read %0[%c0], %i0: memref<?xf32>, vector<4xf32>
    vector.print %v0 : vector<4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?xf32>
//    bufferization.dealloc_tensor %o0_4_4 : tensor<?xf32>
    return
  }
}
