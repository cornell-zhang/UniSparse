module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSwap(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptSum(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptEnumerate(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptPad(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptCustPad(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %4 = call @sptSeparate(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %5 = call @sptTrim(%4, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %6 = call @sptSum(%2, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %7 = call @sptEnumerate(%6, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %8 = call @sptMove(%7, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %9 = call @sptCustPad(%8, %c0_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
//    %10 = call @sptMove(%9, %c0_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
//    %11 = call @sptMove(%10, %c1_i32, %c1_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %12 = call @rtclock() : () -> f64
    %13 = arith.subf %12, %3 : f64
    vector.print %12 : f64
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%9) : (!llvm.ptr<i8>) -> ()
    return
  }
}
