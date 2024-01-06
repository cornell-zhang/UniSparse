module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptCheckF32(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwapF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFusedTransposeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopyF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptPrintF32(!llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFileF32(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopyF32(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
//    call @sptPrint(%1) : (!llvm.ptr<i8>) -> ()
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %4 = call @sptFuseF32(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptGrowF32(%4, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %6 = call @rtclock() : () -> f64
    %7 = arith.subf %6, %3 : f64
    // vector.print %7 : f64
    %8 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %10 = call @sptFusedTransposeF32(%5, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
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
    %18 = call @sptFusedTransposeF32(%10, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %24 = call @rtclock() : () -> f64
    %25 = arith.subf %24, %17 : f64
    // vector.print %25 : f64
    %26 = call @rtclock() : () -> f64
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %c3_i32_15 = arith.constant 3 : i32
    %c4_i32_16 = arith.constant 4 : i32
    %c5_i32_17 = arith.constant 5 : i32
    %27 = call @sptSeparateF32(%18, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %28 = call @sptTrimF32(%27, %c0_i32_12) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %29 = call @rtclock() : () -> f64
    %30 = arith.subf %29, %26 : f64
    // vector.print %30 : f64
    // call @sptCheckF32(%28, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    %31 = call @rtclock() : () -> f64
    %c0_i32_18 = arith.constant 0 : i32
    %c1_i32_19 = arith.constant 1 : i32
    %c2_i32_20 = arith.constant 2 : i32
    %c3_i32_21 = arith.constant 3 : i32
    %c4_i32_22 = arith.constant 4 : i32
    %c5_i32_23 = arith.constant 5 : i32
    %32 = call @sptSwapF32(%28, %c0_i32_18, %c1_i32_19) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %33 = call @sptMoveF32(%32, %c0_i32_18, %c0_i32_18) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %34 = call @sptFuseF32(%33, %c0_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %35 = call @sptGrowF32(%34, %c0_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %36 = call @rtclock() : () -> f64
    %37 = arith.subf %36, %3 : f64
    // vector.print %37 : f64
    %38 = call @rtclock() : () -> f64
    %c0_i32_24 = arith.constant 0 : i32
    %c1_i32_25 = arith.constant 1 : i32
    %c2_i32_26 = arith.constant 2 : i32
    %c3_i32_27 = arith.constant 3 : i32
    %c4_i32_28 = arith.constant 4 : i32
    %c5_i32_29 = arith.constant 5 : i32
    %39 = call @sptTrimF32(%35, %c0_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %40 = call @sptSeparateF32(%39, %c0_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %41 = call @sptSwapF32(%40, %c0_i32_24, %c1_i32_25) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %42 = call @sptMoveF32(%41, %c0_i32_24, %c0_i32_24) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %43 = call @rtclock() : () -> f64
    %44 = arith.subf %43, %38 : f64
    // vector.print %44 : f64
    // call @sptCheckF32(%42, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%42) : (!llvm.ptr<i8>) -> ()
    return
  }
}
