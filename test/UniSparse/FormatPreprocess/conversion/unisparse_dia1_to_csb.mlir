
!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#DIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i, j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,0)>
}>

#CSB = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i, j)->(i floordiv 64, j floordiv 64, i mod 64, j mod 64)>,
  compressMap = #sparlay.compress<fuse(0,1), trim(2,3)>
}>

module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %a_ori = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
//    sparlay.printStorage (%a_ori): tensor<?x?xf32, #COO>
    %a1 = sparlay.copy (%a_ori): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>
    
    %a2 = sparlay.convert (%a1): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DIA>
//    sparlay.printStorage (%a2): tensor<?x?xf32, #DIA>
    
    %t_start0 = call @rtclock() : () -> f64
    %a3 = sparlay.convert (%a2): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #CSB>
//    sparlay.printStorage (%a3): tensor<?x?xf32, #CSB>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    %t_start1 = call @rtclock() : () -> f64
    %a4 = sparlay.convert (%a3): tensor<?x?xf32, #CSB> to tensor<?x?xf32, #DIA>
//    sparlay.printStorage (%a4): tensor<?x?xf32, #DIA>
    %t_end1 = call @rtclock() : () -> f64
    %t_1 = arith.subf %t_end1, %t_start1: f64
    vector.print %t_1 : f64

    %a5 = sparlay.convert (%a4): tensor<?x?xf32, #DIA> to tensor<?x?xf32, #COO>
//    sparlay.printStorage (%a5): tensor<?x?xf32, #COO>

    sparlay.check (%a5, %a_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>

    //Release the resources 
    bufferization.dealloc_tensor %a_ori : tensor<?x?xf32, #COO>
    bufferization.dealloc_tensor %a5 : tensor<?x?xf32, #COO>
    return
  }
}