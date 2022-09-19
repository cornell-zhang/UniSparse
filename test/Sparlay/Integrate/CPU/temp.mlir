module {
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @kernel_spmm(%arg0: tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >>, %arg1: tensor<?x?xf64>, %arg2: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = sparlay.ptr %arg0, %c1 : tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >> to memref<?xi32>
    %1 = sparlay.crd %arg0, %c1 : tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >> to memref<?xi32>
    %2 = sparlay.value %arg0, %c0 : tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >> to memref<?xf64>
    %3 = bufferization.to_memref %arg1 : memref<?x?xf64>
    %4 = tensor.dim %arg2, %c0 : tensor<?x?xf64>
    %5 = tensor.dim %arg2, %c1 : tensor<?x?xf64>
    %6 = bufferization.to_memref %arg2 : memref<?x?xf64>
    scf.for %arg3 = %c0 to %4 step %c1 {
      %8 = memref.load %0[%arg3] : memref<?xi32>
      %9 = arith.extui %8 : i32 to i64
      %10 = arith.index_cast %9 : i64 to index
      %11 = arith.addi %arg3, %c1 : index
      %12 = memref.load %0[%11] : memref<?xi32>
      %13 = arith.extui %12 : i32 to i64
      %14 = arith.index_cast %13 : i64 to index
      scf.for %arg4 = %10 to %14 step %c1 {
        %15 = memref.load %1[%arg4] : memref<?xi32>
        %16 = arith.extui %15 : i32 to i64
        %17 = arith.index_cast %16 : i64 to index
        %18 = memref.load %2[%arg4] : memref<?xf64>
        scf.for %arg5 = %c0 to %5 step %c1 {
          %19 = memref.load %6[%arg3, %arg5] : memref<?x?xf64>
          %20 = memref.load %3[%17, %arg5] : memref<?x?xf64>
          %21 = arith.mulf %18, %20 : f64
          %22 = arith.addf %19, %21 : f64
          memref.store %22, %6[%arg3, %arg5] : memref<?x?xf64>
        }
      }
    }
    %7 = bufferization.to_tensor %6 : memref<?x?xf64>
    return %7 : tensor<?x?xf64>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = sparlay.fromFile(%0) : !llvm.ptr<i8> to tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(0,1), fuse_level(), bitWidth: 8, schedule: >>
    %2 = sparlay.convert(%1) : tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(0,1), fuse_level(), bitWidth: 8, schedule: >> to tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >>
    %3 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf64>
    %4 = scf.for %arg0 = %c0 to %c256 step %c1 iter_args(%arg1 = %3) -> (tensor<?x?xf64>) {
      %9 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %10 = arith.muli %arg0, %c4 : index
        %11 = arith.addi %arg2, %10 : index
        %12 = arith.index_cast %11 : index to i32
        %13 = arith.sitofp %12 : i32 to f64
        %14 = tensor.insert %13 into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %14 : tensor<?x?xf64>
      }
      scf.yield %9 : tensor<?x?xf64>
    }
    %5 = bufferization.alloc_tensor(%c4, %c4) : tensor<?x?xf64>
    %6 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %5) -> (tensor<?x?xf64>) {
      %9 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> (tensor<?x?xf64>) {
        %10 = tensor.insert %cst into %arg3[%arg0, %arg2] : tensor<?x?xf64>
        scf.yield %10 : tensor<?x?xf64>
      }
      scf.yield %9 : tensor<?x?xf64>
    }
    %7 = call @kernel_spmm(%2, %4, %6) : (tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>
    %8 = vector.transfer_read %7[%c0, %c0], %cst : tensor<?x?xf64>, vector<4x4xf64>
    vector.print %8 : vector<4x4xf64>
//    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #sparlay<encoding crdMap: { (d0, d1) -> (d0, d1); indirect_level() }, trim_level(1,1), fuse_level(0), bitWidth: 8, schedule: >>
    return
  }
}
