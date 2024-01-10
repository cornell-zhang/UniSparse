module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptTileMergeF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptDevectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplitF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwapF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopyF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %c0 = arith.constant 0 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFileF32(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopyF32(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %3 = call @sptSwapF32(%2, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %4 = call @sptMoveF32(%3, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %5 = call @sptFuseF32(%4, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %6 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %7 = call @sptSeparateF32(%5, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @sptSwapF32(%7, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %9 = call @sptMoveF32(%8, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c4_i32_6 = arith.constant 4 : i32
    %10 = call @sptTileSplitF32(%9, %c1_i32_1, %c4_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c4_i32_7 = arith.constant 4 : i32
    %11 = call @sptTileSplitF32(%10, %c0_i32_0, %c4_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @sptMoveF32(%11, %c2_i32_2, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %13 = call @sptMoveF32(%12, %c1_i32_1, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %14 = call @sptFuseF32(%13, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %15 = call @sptFuseF32(%14, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %16 = call @sptGrowF32(%15, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %17 = call @sptVectorizeF32(%16, %c2_i32_2) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %18 = call @rtclock() : () -> f64
    %19 = arith.subf %18, %6 : f64
    vector.print %19 : f64
    %20 = call @rtclock() : () -> f64
    %c0_i32_8 = arith.constant 0 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %c2_i32_10 = arith.constant 2 : i32
    %c3_i32_11 = arith.constant 3 : i32
    %c4_i32_12 = arith.constant 4 : i32
    %c5_i32_13 = arith.constant 5 : i32
    %21 = call @sptDevectorizeF32(%17, %c2_i32_10) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %22 = call @sptTrimF32(%21, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %23 = call @sptSeparateF32(%22, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %24 = call @sptSeparateF32(%23, %c1_i32_9) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %25 = call @sptMoveF32(%24, %c2_i32_10, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %26 = call @sptMoveF32(%25, %c3_i32_11, %c3_i32_11) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c4_i32_14 = arith.constant 4 : i32
    %27 = call @sptTileMergeF32(%26, %c2_i32_10, %c4_i32_14) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c4_i32_15 = arith.constant 4 : i32
    %28 = call @sptTileMergeF32(%27, %c0_i32_8, %c4_i32_15) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %29 = call @sptSwapF32(%28, %c0_i32_8, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %30 = call @sptMoveF32(%29, %c0_i32_8, %c0_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %31 = call @sptFuseF32(%30, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %32 = call @rtclock() : () -> f64
    %33 = arith.subf %32, %20 : f64
    // vector.print %33 : f64
    %c0_i32_16 = arith.constant 0 : i32
    %c1_i32_17 = arith.constant 1 : i32
    %c2_i32_18 = arith.constant 2 : i32
    %c3_i32_19 = arith.constant 3 : i32
    %c4_i32_20 = arith.constant 4 : i32
    %c5_i32_21 = arith.constant 5 : i32
    %34 = call @sptSeparateF32(%31, %c0_i32_16) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %35 = call @sptSwapF32(%34, %c0_i32_16, %c1_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %36 = call @sptMoveF32(%35, %c0_i32_16, %c0_i32_16) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%36) : (!llvm.ptr<i8>) -> ()
    return
  }
}
