
!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#DCSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

#DCSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1000 : index
    %c256 = arith.constant 916428 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %a_ori = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %a0 = sparlay.copy (%a_ori): tensor<?x?xf32, #COO> to tensor<?x?xf32, #COO>

    %t_start0 = call @rtclock() : () -> f64
    %a1 = sparlay.convert (%a0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSR>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    %t_start1 = call @rtclock() : () -> f64
    %a2 = sparlay.convert (%a1): tensor<?x?xf32, #DCSR> to tensor<?x?xf32, #DCSC>
    %t_end1 = call @rtclock() : () -> f64
    %t_1 = arith.subf %t_end1, %t_start1: f64
    vector.print %t_1 : f64

    %t_start2 = call @rtclock() : () -> f64
    %a3 = sparlay.convert (%a2): tensor<?x?xf32, #DCSC> to tensor<?x?xf32, #DCSR>
    %t_end2 = call @rtclock() : () -> f64
    %t_2 = arith.subf %t_end2, %t_start2: f64
    vector.print %t_2 : f64

    %t_start3 = call @rtclock() : () -> f64
    %a4 = sparlay.convert (%a3): tensor<?x?xf32, #DCSR> to tensor<?x?xf32, #COO>
    %t_end3 = call @rtclock() : () -> f64
    %t_3 = arith.subf %t_end3, %t_start3: f64
    vector.print %t_3 : f64

    sparlay.check (%a4, %a_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>

    %t_start4 = call @rtclock() : () -> f64
    %a5 = sparlay.convert (%a4): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DCSC>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start0: f64
    vector.print %t_4 : f64

    %t_start5 = call @rtclock() : () -> f64
    %a6 = sparlay.convert (%a5): tensor<?x?xf32, #DCSC> to tensor<?x?xf32, #COO>
    %t_end5 = call @rtclock() : () -> f64
    %t_5 = arith.subf %t_end5, %t_start5: f64
    vector.print %t_5 : f64

    sparlay.check (%a6, %a_ori): tensor<?x?xf32, #COO>, tensor<?x?xf32, #COO>

    //Release the resources 
    bufferization.dealloc_tensor %a_ori : tensor<?x?xf32, #COO>
    bufferization.dealloc_tensor %a6 : tensor<?x?xf32, #COO>
    return
  }
}
