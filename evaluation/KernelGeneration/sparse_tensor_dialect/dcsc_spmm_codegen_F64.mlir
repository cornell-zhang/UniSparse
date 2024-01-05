module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @kernel_dcsc_spmm(%arg0: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.alloc_tensor() copy(%arg2) {bufferization.escape = [false]} : tensor<?x?xf64>
    %1 = sparse_tensor.pointers %arg0, %c0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %2 = sparse_tensor.indices %arg0, %c0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %3 = sparse_tensor.pointers %arg0, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %4 = sparse_tensor.indices %arg0, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %5 = sparse_tensor.values %arg0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
    %6 = bufferization.to_memref %arg1 : memref<?x?xf64>
    %7 = tensor.dim %0, %c1 : tensor<?x?xf64>
    %8 = bufferization.to_memref %0 : memref<?x?xf64>
    %9 = memref.load %1[%c0] : memref<?xindex>
    %10 = memref.load %1[%c1] : memref<?xindex>
    scf.for %arg3 = %9 to %10 step %c1 {
      %12 = memref.load %2[%arg3] : memref<?xindex>
      %13 = memref.load %3[%arg3] : memref<?xindex>
      %14 = arith.addi %arg3, %c1 : index
      %15 = memref.load %3[%14] : memref<?xindex>
      scf.for %arg4 = %13 to %15 step %c1 {
        %16 = memref.load %4[%arg4] : memref<?xindex>
        %17 = memref.load %5[%arg4] : memref<?xf64>
        scf.for %arg5 = %c0 to %7 step %c1 {
          %18 = memref.load %8[%16, %arg5] : memref<?x?xf64>
          %19 = memref.load %6[%12, %arg5] : memref<?x?xf64>
          %20 = arith.mulf %17, %19 : f64
          %21 = arith.addf %18, %20 : f64
          memref.store %21, %8[%16, %arg5] : memref<?x?xf64>
        }
      }
    }
    %11 = bufferization.to_tensor %8 : memref<?x?xf64>
    return %11 : tensor<?x?xf64>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @rtclock() : () -> f64
    %2 = sparse_tensor.new %0 {bufferization.escape = [false]} : !llvm.ptr<i8> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %3 = tensor.dim %2, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %4 = call @rtclock() : () -> f64
    %5 = arith.subf %4, %1 : f64
    vector.print %5 : f64
    %6 = bufferization.alloc_tensor(%3, %c1000) {bufferization.escape = [false]} : tensor<?x?xf64>
    %7 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %6) -> (tensor<?x?xf64>) {
      %15 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %16 = arith.muli %arg0, %c1000 : index
        %17 = arith.addi %arg2, %16 : index
        %18 = arith.index_cast %17 : index to i32
        %19 = arith.sitofp %18 : i32 to f64
        %20 = tensor.insert %19 into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %20 : tensor<?x?xf64>
      }
      scf.yield %15 : tensor<?x?xf64>
    }
    %8 = bufferization.alloc_tensor(%3, %c1000) {bufferization.escape = [false]} : tensor<?x?xf64>
    %9 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %8) -> (tensor<?x?xf64>) {
      %15 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %16 = tensor.insert %cst into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %16 : tensor<?x?xf64>
      }
      scf.yield %15 : tensor<?x?xf64>
    }
    %10 = call @rtclock() : () -> f64
    %11 = call @kernel_dcsc_spmm(%2, %7, %9) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %12 = call @rtclock() : () -> f64
    %13 = arith.subf %12, %10 : f64
    vector.print %13 : f64
    %14 = vector.transfer_read %11[%c0, %c0], %cst : tensor<?x?xf64>, vector<4x4xf64>
    vector.print %14 : vector<4x4xf64>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    return
  }
}

