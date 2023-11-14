module {
  func.func private @getValueF64(!llvm.ptr<i8>, index) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @getCrdF64(!llvm.ptr<i8>, index) -> memref<?xi32> attributes {llvm.emit_c_interface}
  func.func private @getPtrF64(!llvm.ptr<i8>, index) -> memref<?xi32> attributes {llvm.emit_c_interface}
  func.func private @delUniSparseTensorF64(!llvm.ptr<i8>)
  func.func private @sptFuseF64(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF64(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwapF64(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sparseDimSizeF64(!llvm.ptr<i8>, index) -> index attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF64(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @kernel_dcsc_spmm(%arg0: !llvm.ptr<i8>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = call @getPtrF64(%arg0, %c0) : (!llvm.ptr<i8>, index) -> memref<?xi32>
    %1 = call @getCrdF64(%arg0, %c1) : (!llvm.ptr<i8>, index) -> memref<?xi32>
    %2 = call @getPtrF64(%arg0, %c1) : (!llvm.ptr<i8>, index) -> memref<?xi32>
    %3 = call @getCrdF64(%arg0, %c2) : (!llvm.ptr<i8>, index) -> memref<?xi32>
    %4 = call @getValueF64(%arg0, %c0) : (!llvm.ptr<i8>, index) -> memref<?xf64>
    %5 = bufferization.to_memref %arg1 : memref<?x?xf64>
    %6 = tensor.dim %arg2, %c1 : tensor<?x?xf64>
    %7 = bufferization.to_memref %arg2 : memref<?x?xf64>
    %8 = memref.load %0[%c0] : memref<?xi32>
    %9 = arith.extui %8 : i32 to i64
    %10 = arith.index_cast %9 : i64 to index
    %11 = memref.load %0[%c1] : memref<?xi32>
    %12 = arith.extui %11 : i32 to i64
    %13 = arith.index_cast %12 : i64 to index
    scf.for %arg3 = %10 to %13 step %c1 {
      %15 = memref.load %1[%arg3] : memref<?xi32>
      %16 = arith.extui %15 : i32 to i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = memref.load %2[%arg3] : memref<?xi32>
      %19 = arith.extui %18 : i32 to i64
      %20 = arith.index_cast %19 : i64 to index
      %21 = arith.addi %arg3, %c1 : index
      %22 = memref.load %2[%21] : memref<?xi32>
      %23 = arith.extui %22 : i32 to i64
      %24 = arith.index_cast %23 : i64 to index
      scf.for %arg4 = %20 to %24 step %c1 {
        %25 = memref.load %3[%arg4] : memref<?xi32>
        %26 = arith.extui %25 : i32 to i64
        %27 = arith.index_cast %26 : i64 to index
        %28 = memref.load %4[%arg4] : memref<?xf64>
        scf.for %arg5 = %c0 to %6 step %c1 {
          %29 = memref.load %7[%27, %arg5] : memref<?x?xf64>
          %30 = memref.load %5[%17, %arg5] : memref<?x?xf64>
          %31 = arith.mulf %28, %30 : f64
          %32 = arith.addf %29, %31 : f64
          memref.store %32, %7[%27, %arg5] : memref<?x?xf64>
        }
      }
    }
    %14 = bufferization.to_tensor %7 : memref<?x?xf64>
    return %14 : tensor<?x?xf64>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @rtclock() : () -> f64
    %2 = call @sptFromFileF64(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c1_0 = arith.constant 1 : index
    %3 = call @sparseDimSizeF64(%2, %c1_0) : (!llvm.ptr<i8>, index) -> index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %4 = call @sptSwapF64(%2, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %5 = call @sptMoveF64(%4, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptFuseF64(%5, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @rtclock() : () -> f64
    %8 = arith.subf %7, %1 : f64
    vector.print %8 : f64
    %9 = bufferization.alloc_tensor(%3, %c1000) : tensor<?x?xf64>
    %10 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %9) -> (tensor<?x?xf64>) {
      %18 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %19 = arith.muli %arg0, %c1000 : index
        %20 = arith.addi %arg2, %19 : index
        %21 = arith.index_cast %20 : index to i32
        %22 = arith.sitofp %21 : i32 to f64
        %23 = tensor.insert %22 into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %23 : tensor<?x?xf64>
      }
      scf.yield %18 : tensor<?x?xf64>
    }
    %11 = bufferization.alloc_tensor(%3, %c1000) : tensor<?x?xf64>
    %12 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %11) -> (tensor<?x?xf64>) {
      %18 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %19 = tensor.insert %cst into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %19 : tensor<?x?xf64>
      }
      scf.yield %18 : tensor<?x?xf64>
    }
    %13 = call @rtclock() : () -> f64
    %14 = call @kernel_dcsc_spmm(%6, %10, %12) : (!llvm.ptr<i8>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %15 = call @rtclock() : () -> f64
    %16 = arith.subf %15, %13 : f64
    vector.print %16 : f64
    %17 = vector.transfer_read %14[%c0, %c0], %cst : tensor<?x?xf64>, vector<4x4xf64>
    vector.print %17 : vector<4x4xf64>
    call @delUniSparseTensorF64(%2) : (!llvm.ptr<i8>) -> ()
    return
  }
}
