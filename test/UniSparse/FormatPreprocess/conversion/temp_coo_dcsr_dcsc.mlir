module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwap(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %5 = call @rtclock() : () -> f64
    %6 = arith.subf %5, %3 : f64
    vector.print %6 : f64
    %7 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %8 = call @sptSeparate(%4, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %9 = call @sptSwap(%8, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %10 = call @sptMove(%9, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %11 = call @sptFuse(%10, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %12 = call @rtclock() : () -> f64
    %13 = arith.subf %12, %7 : f64
    vector.print %13 : f64
    %14 = call @rtclock() : () -> f64
    %c0_i32_6 = arith.constant 0 : i32
    %c1_i32_7 = arith.constant 1 : i32
    %c2_i32_8 = arith.constant 2 : i32
    %c3_i32_9 = arith.constant 3 : i32
    %c4_i32_10 = arith.constant 4 : i32
    %c5_i32_11 = arith.constant 5 : i32
    %15 = call @sptSeparate(%11, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %16 = call @sptSwap(%15, %c0_i32_6, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %17 = call @sptMove(%16, %c0_i32_6, %c0_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %18 = call @sptFuse(%17, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %19 = call @rtclock() : () -> f64
    %20 = arith.subf %19, %14 : f64
    vector.print %20 : f64
    %21 = call @rtclock() : () -> f64
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %c3_i32_15 = arith.constant 3 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %c5_i32_17 = arith.constant 5 : i32
    %22 = call @sptSeparate(%18, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %23 = call @rtclock() : () -> f64
    %24 = arith.subf %23, %21 : f64
    vector.print %24 : f64
    call @sptCheck(%22, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    %25 = call @rtclock() : () -> f64
    %c0_i32_18 = arith.constant 0 : i32
    %c1_i32_19 = arith.constant 1 : i32
    %c2_i32_20 = arith.constant 2 : i32
    %c3_i32_21 = arith.constant 3 : i32
    %c4_i32_22 = arith.constant 4 : i32
    %c5_i32_23 = arith.constant 5 : i32
    %26 = call @sptSwap(%22, %c0_i32_18, %c1_i32_19) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %27 = call @sptMove(%26, %c0_i32_18, %c0_i32_18) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %28 = call @sptFuse(%27, %c0_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %29 = call @rtclock() : () -> f64
    %30 = arith.subf %29, %3 : f64
    vector.print %30 : f64
    %31 = call @rtclock() : () -> f64
    %c0_i32_24 = arith.constant 0 : i32
    %c1_i32_25 = arith.constant 1 : i32
    %c2_i32_26 = arith.constant 2 : i32
    %c3_i32_27 = arith.constant 3 : i32
    %c4_i32_28 = arith.constant 4 : i32
    %c5_i32_29 = arith.constant 5 : i32
    %32 = call @sptSeparate(%28, %c0_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %33 = call @sptSwap(%32, %c0_i32_24, %c1_i32_25) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %34 = call @sptMove(%33, %c0_i32_24, %c0_i32_24) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %35 = call @rtclock() : () -> f64
    %36 = arith.subf %35, %31 : f64
    vector.print %36 : f64
    call @sptCheck(%34, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%34) : (!llvm.ptr<i8>) -> ()
    return
  }
}
