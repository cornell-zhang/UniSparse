!Filename = type !llvm.ptr<i8>


#COO = #sparlay.encoding<{
  primaryMap = affine_map<(i,j)->()>,
  secondaryMap = #sparlay.affine<(i,j)->(trim i, j)>
}>

#CSR = #sparlay.encoding<{
  secondaryMap = #sparlay.affine<(j,i)->(fuse i, trim j)>,
  primaryMap = affine_map<(i,j)->()>
}>



module {
  // CHECK-LABEL: func @convert()
  func private @getTensorFilename(index) -> (!Filename)
  func @main() {
    %i0 = constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A_COO = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    //Lower to:
    //%1 = call @sptFromFile(%filename): !llvm.ptr<i8> -> !llvm.ptr<i8>
    %A_CSR = sparlay.convert (%A_COO) : tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    //Lower to:
    //%2 = call @sptFuse(%1,1): !llvm.ptr<i8> -> !llvm.ptr<i8>
    //%3 = call @sptGrow(%2,1): !llvm.ptr<i8> -> !llvm.ptr<i8>
    sparlay.printStorage (%A_CSR) : tensor<?x?xf32, #CSR>
    //Lower to:
    //call @sptPrint(%3,0)
    return
  }

}