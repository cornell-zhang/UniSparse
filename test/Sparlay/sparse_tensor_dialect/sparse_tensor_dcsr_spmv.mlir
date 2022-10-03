// mlir-opt ./sparse_tensor_dcsr_spmv.mlir -sparse-compiler | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o spmv.o

// clang++ spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o spmv

// ./spmv

!Filename = !llvm.ptr<i8>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>
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

  func.func @kernel_dcsr_spmv(%arg0: tensor<?x?xf32, #DCSR>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic #trait1
    ins(%arg0, %arg1 : tensor<?x?xf32, #DCSR>, tensor<?xf32>)
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

    %t_start1 = call @rtclock() : () -> f64
    %a2 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #DCSR>
    %t_end1 = call @rtclock() : () -> f64
    %t_1 = arith.subf %t_end1, %t_start1: f64
    vector.print %t_1 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf32> {
      %k0 = arith.muli %i, %c1 : index
      %k1 = arith.index_cast %k0 : index to i32
      %k = arith.sitofp %k1 : i32 to f32
      %t3 = tensor.insert %k into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %o1_4_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %o1 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o1_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %t_start5 = call @rtclock() : () -> f64
    %1 = call @kernel_dcsr_spmv(%a2, %b, %o1) : (tensor<?x?xf32, #DCSR>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %t_end5 = call @rtclock() : () -> f64
    %t_5 = arith.subf %t_end5, %t_start5: f64
    vector.print %t_5 : f64
    %v1 = vector.transfer_read %1[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v1 : vector<4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %a2 : tensor<?x?xf32, #DCSR>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o1_4_4 : tensor<?x?xf32>
    return
  }
}
