module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptTileMergeF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplitF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopyF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %c0 = arith.constant 0 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFileF32(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopyF32(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %4 = call @sptTileSplitF32(%2, %c0_i32, %c1024_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %5 = call @sptMoveF32(%4, %c1_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptFuseF32(%5, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptFuseF32(%6, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @sptGrowF32(%7, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %9 = call @rtclock() : () -> f64
    %10 = arith.subf %9, %3 : f64
    vector.print %10 : f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %11 = call @sptTrimF32(%8, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %12 = call @sptSeparateF32(%11, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %13 = call @sptSeparateF32(%12, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %14 = call @sptMoveF32(%13, %c1_i32_1, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c1024_i32_6 = arith.constant 1024 : i32
    %15 = call @sptTileMergeF32(%14, %c0_i32_0, %c1024_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%15) : (!llvm.ptr<i8>) -> ()
    return
  }
}
