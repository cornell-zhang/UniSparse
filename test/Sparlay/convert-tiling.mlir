// sparlay-opt sparlay-convert.mlir -lower-format-conversion -lower-struct -dce | \
//     mlir-opt -convert-vector-to-scf --convert-scf-to-std --tensor-constant-bufferize \
//     --tensor-bufferize --std-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 | tee sparlay-convert.asm

// as sparlay-convert.asm -o sparlay-convert.o

// clang++ sparlay-convert.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec



!Filename = type !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSB = #sparlay.encoding<{
  crdMap = affine_map<(i,j)->(i floordiv 2, j floordiv 3, i mod 2, j mod 3)>,
  compressMap = #sparlay.compress<fuse(1), trim(2,3)>
}>

module {
  // CHECK-LABEL: func @convert()
  func private @getTensorFilename(index) -> (!Filename)
  func @main() {
    %i0 = constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    sparlay.printStorage (%A_COO) : tensor<?x?xf32, #COO>

    %A_CSB = sparlay.convert (%A_COO) : tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSB>
    sparlay.printStorage (%A_CSB) : tensor<?x?xf32, #CSB>
    return
  }

}