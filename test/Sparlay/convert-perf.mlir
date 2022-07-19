// sparlay-opt convert-perf.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-std --tensor-constant-bufferize \
//     --tensor-bufferize --std-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 | tee 1.asm


// as 1.asm -o 1.o

// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec



!Filename = type !llvm.ptr<i8>

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
  func private @getTensorFilename(index) -> (!Filename)
  func @main() {
    %i0 = constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    sparlay.tic()
    %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    sparlay.toc()
    sparlay.tic()
    %A_SV = sparlay.copy (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    sparlay.toc()
    sparlay.tic()
    %A_CSR = sparlay.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    sparlay.toc()
    sparlay.tic()
    %A_DCSR = sparlay.convert (%A_DCSR) : tensor<?x?xf32, #CSR> to tensor<?x?xf32, #DCSR>
    sparlay.toc()
    sparlay.tic()
    %A_CSC = sparlay.convert (%A_DCSR): tensor<?x?xf32, #DCSR> to tensor<?x?xf32, #CSC>
    sparlay.toc()
    sparlay.tic()
    %A_DCSC = sparlay.convert (%A_DCSC): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #DCSC>
    sparlay.toc()
    sparlay.tic()
    %A_COO_1 = sparlay.convert (%A_DCSC): tensor<?x?xf32, #DCSC> to tensor<?x?xf32, #COO>
    sparlay.toc()
    sparlay.check (%A_COO_1, %A_SV): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    sparlay.toc()
    return
  }

}