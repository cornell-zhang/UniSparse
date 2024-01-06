module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptCheckF32(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptTileMergeF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplitF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptDevectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorizeF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptNegF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSubF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCopyF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFromFileF32(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %1 = call @sptFromFileF32(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = call @sptCopyF32(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %3 = call @sptSubF32(%2, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %4 = call @sptNegF32(%3, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptMoveF32(%4, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptFuseF32(%5, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptVectorizeF32(%6, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @rtclock() : () -> f64
    %c0_i32_0 = arith.constant 0 : i32
    %c1_i32_1 = arith.constant 1 : i32
    %c2_i32_2 = arith.constant 2 : i32
    %c3_i32_3 = arith.constant 3 : i32
    %c4_i32_4 = arith.constant 4 : i32
    %c5_i32_5 = arith.constant 5 : i32
    %9 = call @sptDevectorizeF32(%7, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %10 = call @sptSeparateF32(%9, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %11 = call @sptSubF32(%10, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @sptNegF32(%11, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %13 = call @sptMoveF32(%12, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_6 = arith.constant 3 : i32
    %14 = call @sptTileSplitF32(%13, %c1_i32_1, %c3_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_7 = arith.constant 2 : i32
    %15 = call @sptTileSplitF32(%14, %c0_i32_0, %c2_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %16 = call @sptMoveF32(%15, %c2_i32_2, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %17 = call @sptMoveF32(%16, %c1_i32_1, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %18 = call @sptMoveF32(%17, %c1_i32_1, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %19 = call @sptFuseF32(%18, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %20 = call @sptFuseF32(%19, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %21 = call @sptGrowF32(%20, %c1_i32_1) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
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
    %25 = call @sptTrimF32(%21, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %26 = call @sptSeparateF32(%25, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %27 = call @sptSeparateF32(%26, %c1_i32_9) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %28 = call @sptMoveF32(%27, %c2_i32_10, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %29 = call @sptMoveF32(%28, %c3_i32_11, %c3_i32_11) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_14 = arith.constant 3 : i32
    %30 = call @sptTileMergeF32(%29, %c2_i32_10, %c3_i32_14) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_15 = arith.constant 2 : i32
    %31 = call @sptTileMergeF32(%30, %c0_i32_8, %c2_i32_15) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %32 = call @sptSubF32(%31, %c0_i32_8, %c1_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %33 = call @sptNegF32(%32, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %34 = call @sptMoveF32(%33, %c0_i32_8, %c0_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %35 = call @sptFuseF32(%34, %c0_i32_8) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %36 = call @sptVectorizeF32(%35, %c1_i32_9) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %37 = call @rtclock() : () -> f64
    %38 = arith.subf %37, %24 : f64
    vector.print %38 : f64
    %c0_i32_16 = arith.constant 0 : i32
    %c1_i32_17 = arith.constant 1 : i32
    %c2_i32_18 = arith.constant 2 : i32
    %c3_i32_19 = arith.constant 3 : i32
    %c4_i32_20 = arith.constant 4 : i32
    %c5_i32_21 = arith.constant 5 : i32
    %39 = call @sptDevectorizeF32(%36, %c1_i32_17) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %40 = call @sptSeparateF32(%39, %c0_i32_16) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %41 = call @sptSubF32(%40, %c0_i32_16, %c1_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %42 = call @sptNegF32(%41, %c0_i32_16) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %43 = call @sptMoveF32(%42, %c0_i32_16, %c0_i32_16) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %44 = call @sptMoveF32(%43, %c1_i32_17, %c1_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @sptCheckF32(%44, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%44) : (!llvm.ptr<i8>) -> ()
    return
  }
}
