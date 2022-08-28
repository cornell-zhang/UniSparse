// sparlay-opt test/Sparlay/decompose.mlir -lower-struct-convert -lower-struct -dce -lower-format-conversion | \
// mlir-opt -convert-vector-to-scf --convert-scf-to-cf --arith-bufferize --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o
    
// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec

// RUN: sparlay-opt %s -lower-struct-convert -lower-struct -dce -lower-format-conversion | FileCheck %s


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

#NE_CSB = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
  compressMap = #sparlay.compress<fuse(1), trim(1,3)>
}>

#DIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i,i)>,
  compressMap = #sparlay.compress<trim(0,0)>
}>

module {
  //CHECK-LABEL: func.func private @sptCheck(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptTileMerge(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptNeg(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptAdd(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptDevectorize(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptTrim(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptSeparate(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptTileSplit(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptVectorize(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptMove(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptSub(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptSwap(!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptGrow(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptCopy(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @structAccess(!llvm.ptr<i8>, index) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptSplit(memref<2xf32>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @spwAssign(!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @spwTile(!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @spwNew() -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  //CHECK-LABEL: func.func @main()
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @main() {
    %i0 = arith.constant 0: index
    %i1 = arith.constant 1: i32
    //CHECK: %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    //CHECK-NEXT: %1 = call @sptFromFile(%0) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    //CHECK-NEXT: %cst = arith.constant dense<[5.000000e-01, 7.500000e-01]> : tensor<2xf32>
    //CHECK-NEXT: %2 = bufferization.alloc_tensor() copy(%cst) : tensor<2xf32>
    //CHECK-NEXT %3 = bufferization.to_memref %2 : memref<2xf32>
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<?x?xf32, #COO>
    %thres_1 = arith.constant dense<[0.5, 0.75]>: tensor<2xf32>
    %thres_2 = bufferization.alloc_tensor () copy(%thres_1): tensor<2xf32>
    %thres = bufferization.to_memref %thres_2: memref<2xf32>
    //CHECK: %4 = call @spwNew() : () -> !llvm.ptr<i8>
    //CHECK: %5 = call @spwTile(%4, %c0_0, %c0_1, %c2_i32) : (!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8>
    //CHECK: %6 = call @spwTile(%5, %c1, %c0_2, %c2_i32_3) : (!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8>
    //CHECK: %7 = call @spwAssign(%6, %c0_5, %c0_6, %c1_i32_4) : (!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8>
    //CHECK: %8 = call @spwAssign(%7, %c1_8, %c1_9, %c1_i32_7) : (!llvm.ptr<i8>, index, index, i32) -> !llvm.ptr<i8>
    //CHECK: %9 = call @sptSplit(%3, %1, %8) : (memref<2xf32>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %S_1 = sparlay.decompose (%A_1, %thres) {rmap = affine_map<(d0,d1)->(d0 floordiv 2, d1 floordiv 2)>}: 
              tensor<?x?xf32, #COO>, memref<2xf32>
          to  !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO> >
    %B_0 = sparlay.struct_access %S_1[0]: 
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO> >
          to  tensor<?x?xf32, #COO>
    %B_1 = sparlay.struct_access %S_1[1]:
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO> >
          to  tensor<?x?xf32, #COO>
    %B_2 = sparlay.struct_access %S_1[2]:
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO> >
          to  tensor<?x?xf32, #COO>
    %C_0 = sparlay.copy(%B_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    %C_1 = sparlay.copy(%B_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    %C_2 = sparlay.copy(%B_2): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    //CHECK: %16 = call @structAccess(%9, %c0_12) : (!llvm.ptr<i8>, index) -> !llvm.ptr<i8>
    //CHECK: %17 = call @structAccess(%9, %c1_13) : (!llvm.ptr<i8>, index) -> !llvm.ptr<i8>
    //CHECK: %18 = call @structAccess(%9, %c2_14) : (!llvm.ptr<i8>, index) -> !llvm.ptr<i8>
    //CHECK: %19 = call @sptFuse(%16, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %20 = call @sptGrow(%19, %c0_i32) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK: %21 = call @sptSwap(%17, %c0_i32_17, %c1_i32_18) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %22 = call @sptSub(%21, %c0_i32_17, %c1_i32_18) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %23 = call @sptMove(%22, %c0_i32_17, %c0_i32_17) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %24 = call @sptVectorize(%23, %c1_i32_18) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK: %25 = call @sptTileSplit(%18, %c1_i32_24, %c3_i32_26) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %26 = call @sptTileSplit(%25, %c0_i32_23, %c2_i32_25) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %27 = call @sptMove(%26, %c2_i32_25, %c0_i32_23) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %28 = call @sptMove(%27, %c1_i32_24, %c0_i32_23) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %29 = call @sptFuse(%28, %c1_i32_24) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %30 = call @sptGrow(%29, %c0_i32_23) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    %S_2 = sparlay.struct_convert (%S_1):
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO>, tensor<?x?xf32,#COO> >
          to  !sparlay.struct< tensor<?x?xf32,#CSR>, tensor<?x?xf32,#DIA>, tensor<?x?xf32,#NE_CSB> >
    %D_0 = sparlay.struct_access %S_2[0]: 
              !sparlay.struct< tensor<?x?xf32,#CSR>, tensor<?x?xf32,#DIA>, tensor<?x?xf32,#NE_CSB> >
          to  tensor<?x?xf32, #CSR>
    %D_1 = sparlay.struct_access %S_2[1]:
              !sparlay.struct< tensor<?x?xf32,#CSR>, tensor<?x?xf32,#DIA>, tensor<?x?xf32,#NE_CSB> >
          to  tensor<?x?xf32, #DIA>
    %D_2 = sparlay.struct_access %S_2[2]:
              !sparlay.struct< tensor<?x?xf32, #CSR>, tensor<?x?xf32,#DIA>, tensor<?x?xf32,#NE_CSB> >
          to  tensor<?x?xf32, #NE_CSB>
    //CHECK: %31 = call @sptSeparate(%20, %c0_i32_29) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %32 = call @sptTrim(%31, %c0_i32_29) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>          
    %E_0 = sparlay.convert(%D_0): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #COO>
    //CHECK: %33 = call @sptDevectorize(%24) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    //CHECK-NEXT: %34 = call @sptAdd(%33, %c1_i32_36, %c0_i32_35) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %35 = call @sptSub(%34, %c0_i32_35, %c1_i32_36) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %36 = call @sptNeg(%35, %c0_i32_35) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %37 = call @sptMove(%36, %c0_i32_35, %c0_i32_35) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %38 = call @sptMove(%37, %c1_i32_36, %c1_i32_36) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %E_1 = sparlay.convert(%D_1): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #COO>
    //CHECK: %39 = call @sptTrim(%30, %c0_i32_41) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %40 = call @sptSeparate(%39, %c1_i32_42) : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %41 = call @sptMove(%40, %c2_i32_43, %c1_i32_42) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %42 = call @sptMove(%41, %c3_i32_44, %c3_i32_44) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %43 = call @sptTileMerge(%42, %c2_i32_43, %c3_i32_44) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    //CHECK-NEXT: %44 = call @sptTileMerge(%43, %c0_i32_41, %c2_i32_43) : (!llvm.ptr<i8>, i32, i32) -> !llvm.ptr<i8>
    %E_2 = sparlay.convert(%D_2): tensor<?x?xf32, #NE_CSB> to tensor<?x?xf32, #COO>
    sparlay.check (%E_0, %C_0): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    sparlay.check (%E_1, %C_1): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    sparlay.check (%E_2, %C_2): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    return
  }
}