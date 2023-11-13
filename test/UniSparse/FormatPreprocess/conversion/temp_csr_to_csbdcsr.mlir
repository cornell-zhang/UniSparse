module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptTileMerge(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplit(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopy(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptPrint(!llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFile(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    call @sptPrint(%1) : (!llvm.ptr<i8>) -> ()
    %2 = call @sptCopy(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %3 = call @sptFuse(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %4 = call @sptGrow(%3, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @sptPrint(%4) : (!llvm.ptr<i8>) -> ()
    %5 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %6 = call @sptTrim(%4, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptSeparate(%6, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @sptTileSplit(%7, %c1_i32_1, %c3_i32_3) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %9 = call @sptTileSplit(%8, %c0_i32_0, %c2_i32_2) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %10 = call @sptMove(%9, %c2_i32_2, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %11 = call @sptMove(%10, %c1_i32_1, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @sptFuse(%11, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %13 = call @sptFuse(%12, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %14 = call @sptFuse(%13, %c2_i32_2) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %15 = call @sptGrow(%14, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @sptPrint(%15) : (!llvm.ptr<i8>) -> ()
    %16 = call @rtclock() : () -> f64
    %17 = arith.subf %16, %5 : f64
    vector.print %17 : f64
    %c0_i32_6 = arith.constant 0 : i32
    %c1_i32_7 = arith.constant 1 : i32
    %c2_i32_8 = arith.constant 2 : i32
    %c3_i32_9 = arith.constant 3 : i32
    %c4_i32_10 = arith.constant 4 : i32
    %c5_i32_11 = arith.constant 5 : i32
    %18 = call @sptTrim(%15, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %19 = call @sptSeparate(%18, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %20 = call @sptSeparate(%19, %c1_i32_7) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %21 = call @sptSeparate(%20, %c2_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %22 = call @sptMove(%21, %c2_i32_8, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %23 = call @sptMove(%22, %c3_i32_9, %c3_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %24 = call @sptTileMerge(%23, %c2_i32_8, %c3_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %25 = call @sptTileMerge(%24, %c0_i32_6, %c2_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %26 = call @sptFuse(%25, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %27 = call @sptGrow(%26, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @sptPrint(%27) : (!llvm.ptr<i8>) -> ()
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %c3_i32_15 = arith.constant 3 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %c5_i32_17 = arith.constant 5 : i32
    %28 = call @sptSeparate(%27, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %29 = call @sptTrim(%28, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @sptCheck(%29, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%29) : (!llvm.ptr<i8>) -> ()
    return
  }
}
