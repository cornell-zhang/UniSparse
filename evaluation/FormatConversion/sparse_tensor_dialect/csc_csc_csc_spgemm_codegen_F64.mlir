module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @kernel_spgemm(%arg0: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, %arg1: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, %arg2: index, %arg3: index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %true = arith.constant true
    %0 = bufferization.alloc_tensor(%arg2, %arg3) {bufferization.escape = [false]} : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %1 = sparse_tensor.pointers %arg0, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %2 = sparse_tensor.indices %arg0, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %3 = sparse_tensor.values %arg0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
    %4 = sparse_tensor.pointers %arg1, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %5 = sparse_tensor.indices %arg1, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
    %6 = sparse_tensor.values %arg1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
    %7 = tensor.dim %0, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %8 = memref.alloca(%c2) : memref<?xindex>
    scf.for %arg4 = %c0 to %7 step %c1 {
      memref.store %arg4, %8[%c0] : memref<?xindex>
      %values, %filled, %added, %count = sparse_tensor.expand %0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
      %10 = memref.load %4[%arg4] : memref<?xindex>
      %11 = arith.addi %arg4, %c1 : index
      %12 = memref.load %4[%11] : memref<?xindex>
      %13 = scf.for %arg5 = %10 to %12 step %c1 iter_args(%arg6 = %count) -> (index) {
        %14 = memref.load %5[%arg5] : memref<?xindex>
        %15 = memref.load %6[%arg5] : memref<?xf64>
        %16 = memref.load %1[%14] : memref<?xindex>
        %17 = arith.addi %14, %c1 : index
        %18 = memref.load %1[%17] : memref<?xindex>
        %19 = scf.for %arg7 = %16 to %18 step %c1 iter_args(%arg8 = %arg6) -> (index) {
          %20 = memref.load %2[%arg7] : memref<?xindex>
          %21 = memref.load %values[%20] : memref<?xf64>
          %22 = memref.load %3[%arg7] : memref<?xf64>
          %23 = arith.mulf %22, %15 : f64
          %24 = arith.addf %21, %23 : f64
          %25 = memref.load %filled[%20] : memref<?xi1>
          %26 = arith.cmpi eq, %25, %false : i1
          %27 = scf.if %26 -> (index) {
            memref.store %true, %filled[%20] : memref<?xi1>
            memref.store %20, %added[%arg8] : memref<?xindex>
            %28 = arith.addi %arg8, %c1 : index
            scf.yield %28 : index
          } else {
            scf.yield %arg8 : index
          }
          memref.store %24, %values[%20] : memref<?xf64>
          scf.yield %27 : index
        }
        scf.yield %19 : index
      }
      sparse_tensor.compress %0, %8, %values, %filled, %added, %13 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index
    }
    %9 = sparse_tensor.load %0 hasInserts : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    return %9 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = sparse_tensor.new %0 {bufferization.escape = [false]} : !llvm.ptr<i8> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %2 = sparse_tensor.new %0 {bufferization.escape = [false]} : !llvm.ptr<i8> to tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %3 = tensor.dim %1, %c0 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %4 = tensor.dim %2, %c1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %5 = call @rtclock() : () -> f64
    %6 = call @kernel_spgemm(%1, %2, %3, %4) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>, index, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    %7 = call @rtclock() : () -> f64
    %8 = arith.subf %7, %5 : f64
    vector.print %8 : f64
    %9 = sparse_tensor.values %6 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
    %10 = vector.transfer_read %9[%c0], %cst : memref<?xf64>, vector<8xf64>
    %11 = memref.dim %9, %c0 : memref<?xf64>
    vector.print %10 : vector<8xf64>
    vector.print %11 : index
    bufferization.dealloc_tensor %1 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
    return
  }
}

