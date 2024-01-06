module {
  func.func private @delUniSparseTensorF32(!llvm.ptr<i8>)
  func.func private @sptCheckF32(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMoveF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuseF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSumF32(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptScheduleF32(!llvm.ptr<i8>, i32, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptPadF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptReorderF32(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 1024 : i32
    %4 = call @sptSumF32(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptScheduleF32(%4, %c0_i32, %c0_i32, %c5_i32) : (!llvm.ptr<i8>, i32, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptMoveF32(%5, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %7 = call @sptFuseF32(%6, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %8 = call @sptFuseF32(%7, %c1_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %14 = call @rtclock() : () -> f64
    %15 = arith.subf %14, %3 : f64
    vector.print %15 : f64
    call @delUniSparseTensorF32(%1) : (!llvm.ptr<i8>) -> ()
    call @delUniSparseTensorF32(%7) : (!llvm.ptr<i8>) -> ()
    return
  }
}
