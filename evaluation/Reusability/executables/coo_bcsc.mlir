module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptSwapF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileMergeF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptDevectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplitF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopySthThatMustGoWrong(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %c0 = arith.constant 0 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFileF32(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopySthThatMustGoWrong(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32_0 = arith.constant 3 : i32
    %4 = call @sptTileSplitF32(%2, %c1_i32, %c3_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_1 = arith.constant 2 : i32
    %5 = call @sptTileSplitF32(%4, %c0_i32, %c2_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptMoveF32(%5, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %7 = call @sptMoveF32(%6, %c2_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %8 = call @sptFuseF32(%7, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %9 = call @sptFuseF32(%8, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %10 = call @sptGrowF32(%9, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %11 = call @sptVectorizeF32(%10, %c2_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %12 = call @rtclock() : () -> f64
    %13 = arith.subf %12, %3 : f64
    vector.print %13 : f64
    %c0_i32_2 = arith.constant 0 : i32
    %c1_i32_3 = arith.constant 1 : i32
    %c2_i32_4 = arith.constant 2 : i32
    %c3_i32_5 = arith.constant 3 : i32
    %c4_i32_6 = arith.constant 4 : i32
    %c5_i32_7 = arith.constant 5 : i32
    %14 = call @sptDevectorizeF32(%11, %c2_i32_4) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %15 = call @sptTrimF32(%14, %c0_i32_2) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %16 = call @sptSeparateF32(%15, %c0_i32_2) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %17 = call @sptSeparateF32(%16, %c1_i32_3) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %18 = call @sptMoveF32(%17, %c3_i32_5, %c1_i32_3) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %19 = call @sptMoveF32(%18, %c3_i32_5, %c3_i32_5) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_8 = arith.constant 2 : i32
    %20 = call @sptTileMergeF32(%19, %c2_i32_4, %c2_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_9 = arith.constant 3 : i32
    %21 = call @sptTileMergeF32(%20, %c0_i32_2, %c3_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %22 = call @sptSwapF32(%21, %c0_i32_2, %c1_i32_3) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %23 = call @sptMoveF32(%22, %c0_i32_2, %c0_i32_2) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%23) : (!llvm.ptr<i8>) -> ()
    return
  }
}
