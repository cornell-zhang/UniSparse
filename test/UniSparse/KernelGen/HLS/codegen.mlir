module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @rtclock() : () -> f64
    %2 = unisparse.fromFile(%0) : !llvm.ptr<i8> to tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(0,1), fuse_level(), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >>
    %3 = tensor.dim %2, %c1 : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(0,1), fuse_level(), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >>
    %4 = unisparse.convert(%2) : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(0,1), fuse_level(), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >> to tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(1,1), fuse_level(0), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >>
    %5 = call @rtclock() : () -> f64
    %6 = arith.subf %5, %1 : f64
    vector.print %6 : f64
    %7 = bufferization.alloc_tensor(%3) : tensor<?xf32>
    %8 = arith.index_cast %3 : index to i32
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.divf %cst_0, %9 : f32
    %11 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %7) -> (tensor<?xf32>) {
      %25 = tensor.insert %10 into %arg1[%arg0] : tensor<?xf32>
      scf.yield %25 : tensor<?xf32>
    }
    %12 = bufferization.alloc_tensor(%3) : tensor<?xf32>
    %13 = scf.for %arg0 = %c0 to %3 step %c1 iter_args(%arg1 = %12) -> (tensor<?xf32>) {
      %25 = tensor.insert %cst into %arg1[%arg0] : tensor<?xf32>
      scf.yield %25 : tensor<?xf32>
    }
    %14 = call @rtclock() : () -> f64
    %15 = unisparse.ptr %4, %c1 : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(1,1), fuse_level(0), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >> to memref<?xi32>
    %16 = unisparse.crd %4, %c2 : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(1,1), fuse_level(0), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >> to memref<?xi32>
    %17 = unisparse.value %4, %c0 : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(1,1), fuse_level(0), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >> to memref<?xf32>
    %18 = bufferization.to_memref %11 : memref<?xf32>
    %19 = tensor.dim %13, %c0 : tensor<?xf32>
    %20 = bufferization.to_memref %13 : memref<?xf32>
    scf.for %arg0 = %c0 to %19 step %c1 {
      %25 = memref.load %20[%arg0] : memref<?xf32>
      %26 = memref.load %15[%arg0] : memref<?xi32>
      %27 = arith.extui %26 : i32 to i64
      %28 = arith.index_cast %27 : i64 to index
      %29 = arith.addi %arg0, %c1 : index
      %30 = memref.load %15[%29] : memref<?xi32>
      %31 = arith.extui %30 : i32 to i64
      %32 = arith.index_cast %31 : i64 to index
      %33 = scf.for %arg1 = %28 to %32 step %c1 iter_args(%arg2 = %25) -> (f32) {
        %34 = memref.load %16[%arg1] : memref<?xi32>
        %35 = arith.extui %34 : i32 to i64
        %36 = arith.index_cast %35 : i64 to index
        %37 = memref.load %17[%arg1] : memref<?xf32>
        %38 = memref.load %18[%36] : memref<?xf32>
        %39 = arith.mulf %37, %38 : f32
        %40 = arith.addf %arg2, %39 : f32
        scf.yield %40 : f32
      }
      memref.store %33, %20[%arg0] : memref<?xf32>
    }
    %21 = bufferization.to_tensor %20 : memref<?xf32>
    %22 = call @rtclock() : () -> f64
    %23 = arith.subf %22, %14 : f64
    vector.print %23 : f64
    %24 = vector.transfer_read %21[%c0], %cst : tensor<?xf32>, vector<4xf32>
    vector.print %24 : vector<4xf32>
    bufferization.dealloc_tensor %2 : tensor<?x?xf32, #unisparse<encoding crdMap: { (d0, d1) -> (d0, d1) }, trim_level(0,1), fuse_level(), bitWidth: 8, indirectFunc: { layout = < pack_level(), partition_level() > } >>
    return
  }
}
