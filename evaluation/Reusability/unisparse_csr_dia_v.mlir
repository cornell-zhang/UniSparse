module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptDevectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptNegF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSubF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %3 = call @sptFuseF32(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %4 = call @sptGrowF32(%3, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %6 = call @sptTrimF32(%4, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptSeparateF32(%6, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @sptSubF32(%7, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %9 = call @sptNegF32(%8, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %10 = call @sptMoveF32(%9, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %11 = call @sptFuseF32(%10, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %12 = call @sptVectorizeF32(%11, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %13 = call @rtclock() : () -> f64
    %14 = arith.subf %13, %5 : f64
    vector.print %14 : f64
    %c0_i32_6 = arith.constant 0 : i32
    %c1_i32_7 = arith.constant 1 : i32
    %c2_i32_8 = arith.constant 2 : i32
    %c3_i32_9 = arith.constant 3 : i32
    %c4_i32_10 = arith.constant 4 : i32
    %c5_i32_11 = arith.constant 5 : i32
    %15 = call @sptDevectorizeF32(%12, %c1_i32_7) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %16 = call @sptSeparateF32(%15, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %17 = call @sptSubF32(%16, %c0_i32_6, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %18 = call @sptNegF32(%17, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %19 = call @sptMoveF32(%18, %c0_i32_6, %c0_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %20 = call @sptMoveF32(%19, %c1_i32_7, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %21 = call @sptFuseF32(%20, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %22 = call @sptGrowF32(%21, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %c3_i32_15 = arith.constant 3 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %c5_i32_17 = arith.constant 5 : i32
    %23 = call @sptSeparateF32(%22, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %24 = call @sptTrimF32(%23, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%24) : (!llvm.ptr<i8>) -> ()
    return
  }
}
