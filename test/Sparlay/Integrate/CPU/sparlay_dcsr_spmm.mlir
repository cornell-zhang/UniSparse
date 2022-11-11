// sparlay-opt ./sparlay_dcsr_spmm.mlir -sparlay-codegen -lower-format-conversion -lower-struct -dce | \
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

#DCSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
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

  func.func @kernel_dcsr_spmm(%arg0: tensor<?x?xf64, #DCSR>, %arg1: tensor<?x?xf64>, %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf64, #DCSR>, tensor<?x?xf64>)
    outs(%argx: tensor<?x?xf64>) {
    ^bb0(%a: f64, %b: f64, %x: f64):
      %2 = arith.mulf %a, %b : f64
      %3 = arith.addf %x, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1000 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start2 = call @rtclock() : () -> f64
    %A_2 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf64, #COO>
    %c256 = tensor.dim %A_2, %c1 : tensor<?x?xf64, #COO>
    %a2 = sparlay.convert (%A_2): tensor<?x?xf64, #COO> to tensor<?x?xf64, #DCSR>
    %t_end2 = call @rtclock() : () -> f64
    %t_2 = arith.subf %t_end2, %t_start2: f64
    vector.print %t_2 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf64>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf64> {
      %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %k0 = arith.muli %i, %c4 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f64
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %b2 : tensor<?x?xf64>
    }

    %o2_4_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf64>
    %o2 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o2_4_4) -> tensor<?x?xf64> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf64> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf64>
        scf.yield %t3 : tensor<?x?xf64>
      }
      scf.yield %x2 : tensor<?x?xf64>
    }

    %t_start6 = call @rtclock() : () -> f64
    %2 = call @kernel_dcsr_spmm(%a2, %b, %o2) : (tensor<?x?xf64, #DCSR>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %t_end6 = call @rtclock() : () -> f64
    %t_6 = arith.subf %t_end6, %t_start6: f64
    vector.print %t_6 : f64
    %v2 = vector.transfer_read %2[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v2 : vector<4x4xf64>

    //Release the resources 
    bufferization.dealloc_tensor %A_2 : tensor<?x?xf64, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf64>
//    bufferization.dealloc_tensor %o2_4_4 : tensor<?x?xf64>
    return
  }
}
