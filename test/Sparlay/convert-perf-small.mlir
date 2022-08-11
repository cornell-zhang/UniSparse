// sparlay-opt test/Sparlay/convert-perf-small.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-cf --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DCSR = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#CSC = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DCSC = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#CSB = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
  compressMap = #sparlay.compress<fuse(1), trim(2,3)>
}>

#DIA = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(j-i,i)>,
  compressMap = #sparlay.compress<trim(0,0)>
}>

module {
  // CHECK-LABEL: func @convert()
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @main() {
    %i0 = arith.constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %A_ori = sparlay.copy (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    sparlay.tic()
    %A_2 = sparlay.convert (%A_1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    sparlay.toc()
    sparlay.tic()
    %A_3 = sparlay.convert (%A_2): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    sparlay.toc()
    %A_4 = sparlay.convert (%A_3): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #COO>
    sparlay.check (%A_4, %A_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    return
  }
}