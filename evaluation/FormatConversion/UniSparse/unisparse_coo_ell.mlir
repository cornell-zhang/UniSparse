module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptCheckF32(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwapF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparateF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrimF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrowF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSumF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCustPadF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptEnumerateF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFusedEnumeratePadF32(!llvm.ptr<i8>, i32, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptPadF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %3 = call @rtclock() : () -> f64
    %6 = call @sptSumF32(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %9 = call @sptFusedEnumeratePadF32(%6, %c1_i32, %c1_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32, i32) -> !llvm.ptr<i8>
    %12 = call @rtclock() : () -> f64
    %13 = arith.subf %12, %3 : f64
    vector.print %13 : f64
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%9) : (!llvm.ptr<i8>) -> ()
    return
  }
}
