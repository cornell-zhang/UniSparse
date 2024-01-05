module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptTileMerge(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplit(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptDevectorize(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorize(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptNeg(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSub(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopy(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFile(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopy(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %3 = call @sptSub(%2, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %4 = call @sptNeg(%3, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptMove(%4, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptFuse(%5, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptVectorize(%6, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %9 = call @sptDevectorize(%7, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %10 = call @sptSeparate(%9, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %11 = call @sptSub(%10, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @sptNeg(%11, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %13 = call @sptMove(%12, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_6 = arith.constant 3 : i32
    %14 = call @sptTileSplit(%13, %c1_i32_1, %c3_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_7 = arith.constant 2 : i32
    %15 = call @sptTileSplit(%14, %c0_i32_0, %c2_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %16 = call @sptMove(%15, %c2_i32_2, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %17 = call @sptMove(%16, %c1_i32_1, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %18 = call @sptMove(%17, %c1_i32_1, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %19 = call @sptFuse(%18, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %20 = call @sptFuse(%19, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %21 = call @sptGrow(%20, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %22 = call @rtclock() : () -> f64
    %23 = arith.subf %22, %8 : f64
    vector.print %23 : f64
    %24 = call @rtclock() : () -> f64
    %c0_i32_8 = arith.constant 0 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %c2_i32_10 = arith.constant 2 : i32
    %c3_i32_11 = arith.constant 3 : i32
    %c4_i32_12 = arith.constant 4 : i32
    %c5_i32_13 = arith.constant 5 : i32
    %25 = call @sptTrim(%21, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %26 = call @sptSeparate(%25, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %27 = call @sptSeparate(%26, %c1_i32_9) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %28 = call @sptMove(%27, %c2_i32_10, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %29 = call @sptMove(%28, %c3_i32_11, %c3_i32_11) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_14 = arith.constant 3 : i32
    %30 = call @sptTileMerge(%29, %c2_i32_10, %c3_i32_14) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_15 = arith.constant 2 : i32
    %31 = call @sptTileMerge(%30, %c0_i32_8, %c2_i32_15) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %32 = call @sptSub(%31, %c0_i32_8, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %33 = call @sptNeg(%32, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %34 = call @sptMove(%33, %c0_i32_8, %c0_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %35 = call @sptFuse(%34, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %36 = call @sptVectorize(%35, %c1_i32_9) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %37 = call @rtclock() : () -> f64
    %38 = arith.subf %37, %24 : f64
    vector.print %38 : f64
    %c0_i32_16 = arith.constant 0 : i32
    %c1_i32_17 = arith.constant 1 : i32
    %c2_i32_18 = arith.constant 2 : i32
    %c3_i32_19 = arith.constant 3 : i32
    %c4_i32_20 = arith.constant 4 : i32
    %c5_i32_21 = arith.constant 5 : i32
    %39 = call @sptDevectorize(%36, %c1_i32_17) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %40 = call @sptSeparate(%39, %c0_i32_16) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %41 = call @sptSub(%40, %c0_i32_16, %c1_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %42 = call @sptNeg(%41, %c0_i32_16) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %43 = call @sptMove(%42, %c0_i32_16, %c0_i32_16) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %44 = call @sptMove(%43, %c1_i32_17, %c1_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @sptCheck(%44, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%44) : (!llvm.ptr<i8>) -> ()
    return
  }
}
