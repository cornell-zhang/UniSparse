// sparlay-opt ./sparlay_ell_coo_spmv.mlir -lower-struct-convert -lower-struct -dce -sparlay-codegen -lower-format-conversion | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o bell_spmv.o
    
// clang++ bell_spmv.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o bell_spmv

// ./bell_spmv

// RUN: sparlay-opt %s -lower-struct-convert -lower-struct -dce -lower-format-conversion | FileCheck %s


!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#ELL = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i floordiv 50, j minus i, i mod 50)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#trait1 = {
indexing_maps = [
    affine_map<(i,j) -> (i, j)>,  // A
    affine_map<(i,j) -> (j)>,  // B
    affine_map<(i,j) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) =+ A(i,j) * B(j)"
}

module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)
  // func.func @kernel_coo_spmv(%arg0: tensor<?x?xf32, #COO>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
  //   %0 = linalg.generic #trait1
  //   ins(%arg0, %arg1 : tensor<?x?xf32, #COO>, tensor<?xf32>)
  //   outs(%argx: tensor<?xf32>) {
  //   ^bb0(%a: f32, %b: f32, %x: f32):
  //     %2 = arith.mulf %a, %b : f32
  //     %3 = arith.addf %x, %2 : f32
  //     linalg.yield %3 : f32
  //   } -> tensor<?xf32>
  //   return %0 : tensor<?xf32>
  // }

  func.func @main() {
    %c0 = arith.constant 0: index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0: f32
    %f05 = arith.constant 0.5: f32
    %i1 = arith.constant 1: i32
    %blockSize = arith.constant 32: i32
    %block_thres = arith.constant 0.2: f32
    %col_thres = arith.constant 0.7: f32

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A_1 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<?x?xf32, #COO>
    %dim1 = tensor.dim %A_1, %c1 : tensor<?x?xf32, #COO>
    %dim0 = tensor.dim %A_1, %c0 : tensor<?x?xf32, #COO>

    %t_start0 = call @rtclock() : () -> f64
    %S_1 = sparlay.decompose_BELL %A_1, %blockSize, %block_thres, %col_thres : tensor<?x?xf32, #COO>, i32, f32, f32 to 
          !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#ELL> >
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64
    
    %B_0 = sparlay.struct_access %S_1[0]: 
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#ELL> >
          to  tensor<?x?xf32, #COO>
    %B_1 = sparlay.struct_access %S_1[1]:
              !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#ELL> >
          to  tensor<?x?xf32, #ELL>

    // %init_256_4 = bufferization.alloc_tensor(%dim1) : tensor<?xf32>
    // %b = scf.for %i = %c0 to %dim1 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf32> {
    //   %k1 = arith.index_cast %i : index to i32
    //   %k = arith.sitofp %k1 : i32 to f32
    //   %t3 = tensor.insert %k into %t[%i] : tensor<?xf32>
    //   scf.yield %t3 : tensor<?xf32>
    // }
    
    // %o1 = bufferization.alloc_tensor(%dim0) : tensor<?xf32>
    // %o11 = scf.for %i = %c0 to %dim0 step %c1 iter_args(%t = %o1) -> tensor<?xf32> {
    //   %t3 = tensor.insert %f0 into %t[%i] : tensor<?xf32>
    //   scf.yield %t3 : tensor<?xf32>
    // }
    
    // %t_start4 = call @rtclock() : () -> f64
    // %result1 = sparlay.ell_spmv %B_0, %B_1, %b, %o1: 
    //   tensor<?x?xf32, #COO>, tensor<?x?xf32,#ELL>, tensor<?xf32>, tensor<?xf32> to memref<?xf32>
    // %t_end4 = call @rtclock() : () -> f64
    // %t_5 = arith.subf %t_end4, %t_start4: f64
    // vector.print %t_5 : f64
    // %v1 = vector.transfer_read %result1[%c0], %f0: memref<?xf32>, vector<4xf32>
    // vector.print %v1 : vector<4xf32>

    bufferization.dealloc_tensor %A_1 : tensor<?x?xf32, #COO>
    bufferization.dealloc_tensor %B_1 : tensor<?x?xf32, #ELL>
    sparlay.release %S_1: !sparlay.struct< tensor<?x?xf32,#COO>, tensor<?x?xf32,#ELL> >

    return
  }
}