// sparlay-opt test/Sparlay/to-ops.mlir -lower-struct-convert -lower-struct -dce -lower-format-conversion | \
// mlir-opt -convert-vector-to-scf --convert-scf-to-cf --arith-bufferize --tensor-bufferize \
//     --scf-bufferize --func-bufferize --finalizing-bufferize --convert-vector-to-llvm \
//     --convert-memref-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
//     mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o 1.o
    
// clang++ 1.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec

// ./exec

// RUN: sparlay-opt %s -lower-struct-convert -lower-struct -dce -lower-format-conversion | FileCheck %s

!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#DCSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(0,1)>
}>

module {
  //CHECK-LABEL: func.func private @getSize(!llvm.ptr<i8>, index) -> index attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @getValue(!llvm.ptr<i8>, index) -> memref<9xf32> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @getPtr(!llvm.ptr<i8>, index) -> memref<7xi32> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @getCrd(!llvm.ptr<i8>, index) -> memref<6xi32> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptFuse(!llvm.ptr<i8>, i32) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @sptFromFile(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  //CHECK-LABEL: func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func @main() {
    %i0 = arith.constant 0: index
    %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
    %A0 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<7x7xf32, #COO>
    %A1 = sparlay.convert (%A0): tensor<7x7xf32, #COO> to tensor<7x7xf32, #DCSR>
    //CHECK: %4 = call @getPtr(%2, %c0) : (!llvm.ptr<i8>, index) -> memref<7xi32>
    //CHECK: %5 = call @getValue(%2, %c0) : (!llvm.ptr<i8>, index) -> memref<9xf32>
    //CHECK: %6 = call @getSize(%2, %c0) : (!llvm.ptr<i8>, index) -> index
    %crd = sparlay.crd %A1, %i0: tensor<7x7xf32, #DCSR> to memref<6xi32>
    %ptr = sparlay.ptr %A1, %i0: tensor<7x7xf32, #DCSR> to memref<7xi32>
    %value = sparlay.value %A1, %i0: tensor<7x7xf32, #DCSR> to memref<9xf32>
    %size = sparlay.size %A1, %i0: tensor<7x7xf32, #DCSR> to index
    %vec_crd = vector.load %crd[%i0]: memref<6xi32>, vector<6xi32>
    %vec_ptr = vector.load %ptr[%i0]: memref<7xi32>, vector<7xi32>
    %vec_value = vector.load %value[%i0]: memref<9xf32>, vector<9xf32>
    vector.print %size: index
    vector.print %vec_crd: vector<6xi32>
    vector.print %vec_ptr: vector<7xi32>
    vector.print %vec_value: vector<9xf32>
    return
  }
}