// sparlay-opt test/UniSparse/convert-perf-small.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-cf --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec

// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DCSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#CSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DCSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#CSB = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
  compressMap = #sparlay.compress<fuse(1), trim(2,3)>
}>

#DIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i,i)>,
  compressMap = #sparlay.compress<trim(0,0)>
}>

module {
  // CHECK-LABEL: func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptSwap(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptToc() attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptTic() attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptCopy(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  // CHECK-LABEL: func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  // CHECK-LABEL: func.func @main()
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @main() {
    %i0 = arith.constant 0: index
    // CHECK: %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    // CHECK: %1 = call @sptFromFile(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    // CHECK: %2 = call @sptCopy(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    // CHECK: call @sptTic() : () -> ()
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %A_ori = sparlay.copy (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    sparlay.tic()
    // CHECK: %3 = call @sptFuse(%1, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %4 = call @sptGrow(%3, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %A_2 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    sparlay.toc()
    sparlay.tic()
    // CHECK: %5 = call @sptTrim(%4, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %6 = call @sptSeparate(%5, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %7 = call @sptSwap(%6, %c0_i32_0, %c1_i32_1) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %8 = call @sptMove(%7, %c0_i32_0, %c0_i32_0) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %9 = call @sptFuse(%8, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %10 = call @sptGrow(%9, %c0_i32_0) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %A_3 = sparlay.convert (%A_2): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    sparlay.toc()
    // CHECK: %11 = call @sptTrim(%10, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %12 = call @sptSeparate(%11, %c0_i32_6) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %13 = call @sptSwap(%12, %c0_i32_6, %c1_i32_7) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    // CHECK-NEXT: %14 = call @sptMove(%13, %c0_i32_6, %c0_i32_6) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %A_4 = sparlay.convert (%A_3): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #COO>
    // CHECK: call @sptCheck(%14, %2) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    sparlay.check (%A_4, %A_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    return
  }
}