module {
  func.func private @delSparlayTensor(!llvm.ptr<i8>)
  func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  func.func private @sptTileMerge(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptDevectorize(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptVectorize(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @sptTileSplit(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
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
    %3 = call @rtclock() : () -> f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32_0 = arith.constant 3 : i32
    %4 = call @sptTileSplit(%2, %c1_i32, %c3_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_1 = arith.constant 2 : i32
    %5 = call @sptTileSplit(%4, %c0_i32, %c2_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %6 = call @sptMove(%5, %c2_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %7 = call @sptMove(%6, %c1_i32, %c0_i32) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %8 = call @sptVectorize(%7, %c2_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    call @sptPrint(%8) : (!llvm.ptr<i8>) -> ()
    %9 = call @rtclock() : () -> f64
    %10 = arith.subf %9, %3 : f64
    vector.print %10 : f64
    %c0_i32_2 = arith.constant 0 : i32
    %c1_i32_3 = arith.constant 1 : i32
    %c2_i32_4 = arith.constant 2 : i32
    %c3_i32_5 = arith.constant 3 : i32
    %c4_i32_6 = arith.constant 4 : i32
    %c5_i32_7 = arith.constant 5 : i32
    %11 = call @sptDevectorize(%8, %c2_i32_4) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %12 = call @sptMove(%11, %c2_i32_4, %c1_i32_3) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %13 = call @sptMove(%12, %c3_i32_5, %c3_i32_5) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c3_i32_8 = arith.constant 3 : i32
    %14 = call @sptTileMerge(%13, %c2_i32_4, %c3_i32_8) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %c2_i32_9 = arith.constant 2 : i32
    %15 = call @sptTileMerge(%14, %c0_i32_2, %c2_i32_9) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    call @sptPrint(%15) : (!llvm.ptr<i8>) -> ()
    call @sptCheck(%15, %1) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%1) : (!llvm.ptr<i8>) -> ()
    call @delSparlayTensor(%15) : (!llvm.ptr<i8>) -> ()
    return
  }
}
