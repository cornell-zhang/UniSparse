// unisparse-opt test/UniSparse/convert-correctness.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-cf --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec

// RUN: unisparse-opt %s -lower-format-conversion -lower-struct -dce | FileCheck %s

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
}>

// #CSR = #unisparse.encoding<{
//   crdMap = #unisparse.crd<(i,j)->(i,j)>,
//   compressMap = #unisparse.compress<fuse(0), trim(1,1)>
// }>

// #DCSR = #unisparse.encoding<{
//   crdMap = #unisparse.crd<(i,j)->(i,j)>,
//   compressMap = #unisparse.compress<fuse(0), trim(0,1)>
// }>

// #CSC = #unisparse.encoding<{
//   crdMap = #unisparse.crd<(i,j)->(j,i)>,
//   compressMap = #unisparse.compress<fuse(0), trim(1,1)>
// }>

// #CSB = #unisparse.encoding<{
//   crdMap = #unisparse.crd<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
//   compressMap = #unisparse.compress<fuse(1), trim(1,3)>
// }>

// #DIA = #unisparse.encoding<{
//   crdMap = #unisparse.crd<(i,j)->(j minus i,i)>,
//   compressMap = #unisparse.compress<trim(0,0)>
// }>

#CISR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(indirect(i), i, j)>,
  compressMap = #unisparse.compress<fuse(0,1), trim(1,2)>,
  indirectFunc = #unisparse.indirect<{
    sumVal = #unisparse.sum<groupBy (0), with val ne 0 -> 1 | otherwise -> 0>, // map: original matrix A -> output A' [0, 1]
    schedVal = #unisparse.schedule<traverseBy (0), sumVal, 2> //list[[]] -> list[]
    // sum(level = 1)
    // schedule(level = 1, sum, buckets = 2)
  }>,
  layout = #unisparse.layout<partition(0)>
}>

#CISR_plus = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(indirect(j), i, j)>,
  compressMap = #unisparse.compress<fuse(0), trim(0,0)>,
  indirectFunc = #unisparse.indirect<{
    sumVal = #unisparse.sum<groupBy (0), with val ne 0 -> 1 | otherwise -> 0>,
    reorderVal = #unisparse.reorder<traverseBy (0), sumVal, descend>, // map: original matrix A -> output A' [0, 1]
    schedVal = #unisparse.schedule<traverseBy (0), sumVal, 2> //list[[]] -> list[]
    // sum(level = 1)
    // enumerate(level = 1, sum)
  }>,
  layout = #unisparse.layout<partition(0)>
}>

#ELL = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(indirect(j), i, j)>,
  compressMap = #unisparse.compress<fuse(0), trim(0,0)>,
  indirectFunc = #unisparse.indirect<{
    sumVal = #unisparse.sum<groupBy (0), with val ne 0 -> 1 | otherwise -> 0>,
    enumVal = #unisparse.enumerate<groupBy (0), traverseBy (1), with val eq 0 -> sumVal | otherwise -> 0>
    // sum(level = 1)
    // enumerate(level = 1, sum)
  }>
}>

#hiSparse = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(1,1)>,
  layout = #unisparse.layout<pack(1,2)>
}>

module {
  func.func private @getTensorFilename(index) -> (!Filename)
  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    unisparse.tic ()
    %A_COO = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    // unisparse.printStorage (%A_COO) : tensor<?x?xf32, #COO>

    // %A_SV = unisparse.copy (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>

    // %A_CSR = unisparse.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    // // unisparse.printStorage (%A_CSR): tensor<?x?xf32, #CSR>

    // %A_CSC = unisparse.convert (%A_CSR): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    // // unisparse.printStorage (%A_CSC): tensor<?x?xf32, #CSC>

    // %A_CSB = unisparse.convert (%A_CSC): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #CSB>
    // // unisparse.printStorage (%A_CSB): tensor<?x?xf32, #CSB>

    // %A_DIA = unisparse.convert (%A_CSB): tensor<?x?xf32, #CSB> to tensor<?x?xf32, #DIA>
    // // unisparse.printStorage (%A_DIA): tensor<?x?xf32, #DIA>

    // %A_DCSR = unisparse.convert (%A_DIA): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #DCSR>
    // // unisparse.printStorage (%A_DCSR): tensor<?x?xf32, #DCSR>

    // %A_COO_1 = unisparse.convert (%A_DCSR): tensor<?x?xf32, #DCSR> to tensor<?x?xf32, #COO>
    // // unisparse.printStorage (%A_COO_1): tensor<?x?xf32, #COO>

    %A_CISR = unisparse.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CISR>

    %A_CISR_plus = unisparse.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CISR_plus>

    %A_ELL = unisparse.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #ELL>

    %A_hiSparse = unisparse.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #hiSparse>

    // unisparse.check (%A_COO_1, %A_SV): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>

    unisparse.toc()
    return
  }

}