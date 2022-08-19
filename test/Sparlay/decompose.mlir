// sparlay-opt test/Sparlay/decompose.mlir -lower-struct-convert -lower-struct -dce -lower-format-conversion | \
// mlir-opt -convert-vector-to-scf --convert-scf-to-cf --arith-bufferize --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o
    
// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec


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
  // CHECK-LABEL: func @convert()
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @main() {
    %i0 = arith.constant 0: index
    %i1 = arith.constant 1: i32
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<?x?xf32, #COO>
    %thres_1 = arith.constant dense<[0.5, 0.75]>: tensor<2xf32>
    %thres_2 = bufferization.alloc_tensor () copy(%thres_1): tensor<2xf32>
    %thres = bufferization.to_memref %thres_2: memref<2xf32>
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
    %E_0 = sparlay.convert(%D_0): tensor<?x?xf32, #CSR> to tensor<?x?xf32, #COO>
    %E_1 = sparlay.convert(%D_1): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #COO>
    %E_2 = sparlay.convert(%D_2): tensor<?x?xf32, #NE_CSB> to tensor<?x?xf32, #COO>
    sparlay.check (%E_0, %C_0): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    sparlay.check (%E_1, %C_1): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    sparlay.check (%E_2, %C_2): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>
    return
  }
}