// unisparse-opt ./unisparse_coo_spmm.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o coo_spmm.o

// clang++ coo_spmm.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o coo_spmm

// ./coo_spmm

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
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
  func.func private @getTensorDim(!Filename, index) -> (index)

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1000 : index
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %A_0 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %dim0 = call @getTensorDim(%fileName, %c0) : (!Filename, index) -> (index)
    %dim1 = call @getTensorDim(%fileName, %c1) : (!Filename, index) -> (index)
    // %dim0 = tensor.dim %A_0, %c0 : tensor<?x?xf32, #COO>
    // %dim1 = tensor.dim %A_0, %c1 : tensor<?x?xf32, #COO>

    // Initialize vector matrix.
    %init_256_4 = memref.alloc(%dim1, %c4) : memref<?x?xf32>
    %b = scf.for %i = %c0 to %dim1 step %c1 iter_args(%t = %init_256_4) -> memref<?x?xf32> {
      %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> memref<?x?xf32> {
        %k0 = arith.muli %i, %c4 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f32
        memref.store %k, %t2[%i, %j] : memref<?x?xf32>
        scf.yield %t2 : memref<?x?xf32>
      }
      scf.yield %b2 : memref<?x?xf32>
    }

    %o0_4_4 = memref.alloc(%dim0, %c4) : memref<?x?xf32>
    %o0 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o0_4_4) -> memref<?x?xf32> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> memref<?x?xf32> {
        memref.store %i0, %t2[%i, %j] : memref<?x?xf32>
        scf.yield %t2 : memref<?x?xf32>
      }
      scf.yield %x2 : memref<?x?xf32>
    }

    %t_start4 = call @rtclock() : () -> f64
    %0 = unisparse.coo_spmm %A_0, %init_256_4, %o0_4_4: tensor<?x?xf32, #COO>, memref<?x?xf32>, memref<?x?xf32> to memref<?x?xf32>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v1 = vector.transfer_read %init_256_4[%c0, %c0], %i0: memref<?x?xf32>, vector<4x4xf32>
    vector.print %v1 : vector<4x4xf32>
    %v0 = vector.transfer_read %0[%c0, %c0], %i0: memref<?x?xf32>, vector<4x4xf32>
    vector.print %v0 : vector<4x4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
    return
  }
}
