// unisparse-opt test/UniSparse/convert-perf-all.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-cf --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec_all

// ./exec_all

// RUN: unisparse-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
}>

#CSR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(1,1)>
}>

#DCSR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(0,1)>
}>

#CSC = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(j,i)>,
  compressMap = #unisparse.compress<fuse(0), trim(1,1)>
}>

#DCSC = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(j,i)>,
  compressMap = #unisparse.compress<fuse(0), trim(0,1)>
}>

#CSB = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
  compressMap = #unisparse.compress<fuse(1), trim(2,3)>
}>

#DIA = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(j minus i,i)>,
  compressMap = #unisparse.compress<trim(0,0)>
}>

module {
  func.func private @getTensorFilename(index) -> (!Filename)
  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_1 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %A_ori = unisparse.copy (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    unisparse.tic()
    %A_2 = unisparse.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    unisparse.toc()
    %A_3 = unisparse.convert (%A_2) : tensor<?x?xf32, #CSR> to tensor<?x?xf32, #COO>
    unisparse.tic()
    %A_4 = unisparse.convert (%A_3): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DIA>
    unisparse.toc()
    %A_5 = unisparse.convert (%A_4): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #CSR>
    unisparse.tic()
    %A_6 = unisparse.convert (%A_5): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #DIA>
    unisparse.toc()
    %A_7 = unisparse.convert (%A_6): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #CSR>
    unisparse.tic()
    %A_8 = unisparse.convert (%A_7): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    unisparse.toc()
    unisparse.tic()
    %A_9 = unisparse.convert (%A_8): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #DIA>
    unisparse.toc()
    %A_10 = unisparse.convert (%A_9): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #COO>
    unisparse.check (%A_10, %A_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    return
  }
}