// sparlay-opt convert.mlir -lower-format-conversion -lower-struct -dce | \
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
    sparlay.tic ()
    %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    sparlay.printStorage (%A_COO) : tensor<?x?xf32, #COO>

    %A_SV = sparlay.copy (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>

    %A_CSR = sparlay.convert (%A_COO): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    sparlay.printStorage (%A_CSR): tensor<?x?xf32, #CSR>

    %A_CSC = sparlay.convert (%A_CSR): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
    sparlay.printStorage (%A_CSC): tensor<?x?xf32, #CSC>

    %A_CSB = sparlay.convert (%A_CSC): tensor<?x?xf32, #CSC> to tensor<?x?xf32, #CSB>
    sparlay.printStorage (%A_CSB): tensor<?x?xf32, #CSB>

    %A_DIA = sparlay.convert (%A_CSB): tensor<?x?xf32, #CSB> to tensor<?x?xf32, #DIA>
    sparlay.printStorage (%A_DIA): tensor<?x?xf32, #DIA>

    %A_DCSR = sparlay.convert (%A_DIA): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #DCSR>
    sparlay.printStorage (%A_DCSR): tensor<?x?xf32, #DCSR>

    %A_COO_1 = sparlay.convert (%A_DCSR): tensor<?x?xf32, #DCSR> to tensor<?x?xf32, #COO>
    sparlay.printStorage (%A_COO_1): tensor<?x?xf32, #COO>

    sparlay.check (%A_COO_1, %A_SV): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>

    sparlay.toc()
    return
  }

}