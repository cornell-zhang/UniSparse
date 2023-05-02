module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwap(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopy(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c916428 = arith.constant 916428 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFile(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopy(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %4 = call @sptFuse(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptGrow(%4, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %6 = call @rtclock() : () -> f64
    %7 = arith.subf %6, %3 : f64
    vector.print %7 : f64
    %8 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %9 = call @sptTrim(%5, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %10 = call @sptSeparate(%9, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %11 = call @sptSwap(%10, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @sptMove(%11, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %13 = call @sptFuse(%12, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %14 = call @sptGrow(%13, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %15 = call @rtclock() : () -> f64
    %16 = arith.subf %15, %8 : f64
    vector.print %16 : f64
    %17 = call @rtclock() : () -> f64
    %c0_i32_6 = arith.constant 0 : i32
    %c1_i32_7 = arith.constant 1 : i32
    %c2_i32_8 = arith.constant 2 : i32
    %c3_i32_9 = arith.constant 3 : i32
    %c4_i32_10 = arith.constant 4 : i32
    %c5_i32_11 = arith.constant 5 : i32
    %18 = call @sptTrim(%14, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %19 = call @sptSeparate(%18, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %20 = call @sptSwap(%19, %c0_i32_6, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %21 = call @sptMove(%20, %c0_i32_6, %c0_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %22 = call @sptFuse(%21, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %23 = call @sptGrow(%22, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %24 = call @rtclock() : () -> f64
    %25 = arith.subf %24, %17 : f64
    vector.print %25 : f64
    %26 = call @rtclock() : () -> f64
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %c3_i32_15 = arith.constant 3 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %c5_i32_17 = arith.constant 5 : i32
    %27 = call @sptSeparate(%23, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %28 = call @sptTrim(%27, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %29 = call @rtclock() : () -> f64
    %30 = arith.subf %29, %26 : f64
    vector.print %30 : f64
    call @sptCheck(%28, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    %31 = call @rtclock() : () -> f64
    %c0_i32_18 = arith.constant 0 : i32
    %c1_i32_19 = arith.constant 1 : i32
    %c2_i32_20 = arith.constant 2 : i32
    %c3_i32_21 = arith.constant 3 : i32
    %c4_i32_22 = arith.constant 4 : i32
    %c5_i32_23 = arith.constant 5 : i32
    %32 = call @sptSwap(%28, %c0_i32_18, %c1_i32_19) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %33 = call @sptMove(%32, %c0_i32_18, %c0_i32_18) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %34 = call @sptFuse(%33, %c0_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %35 = call @sptGrow(%34, %c0_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %36 = call @rtclock() : () -> f64
    %37 = arith.subf %36, %3 : f64
    vector.print %37 : f64
    %38 = call @rtclock() : () -> f64
    %c0_i32_24 = arith.constant 0 : i32
    %c1_i32_25 = arith.constant 1 : i32
    %c2_i32_26 = arith.constant 2 : i32
    %c3_i32_27 = arith.constant 3 : i32
    %c4_i32_28 = arith.constant 4 : i32
    %c5_i32_29 = arith.constant 5 : i32
    %39 = call @sptTrim(%35, %c0_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %40 = call @sptSeparate(%39, %c0_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %41 = call @sptSwap(%40, %c0_i32_24, %c1_i32_25) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %42 = call @sptMove(%41, %c0_i32_24, %c0_i32_24) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %43 = call @rtclock() : () -> f64
    %44 = arith.subf %43, %38 : f64
    vector.print %44 : f64
    call @sptCheck(%42, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%42) : (!llvm.ptr<i8>) -> ()
    return
  }
}
