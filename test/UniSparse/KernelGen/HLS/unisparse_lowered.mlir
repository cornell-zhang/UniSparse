// unisparse-opt ./unisparse_csr_spmv_F32.mlir -unisparse-codegen -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o csr_spmv_F32.o

// clang++ csr_spmv_F32.o -L$SPLHOME/build/lib -lmlir_unisparse_runner_utils \
//         -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o csr_spmv_F32

// ./csr_spmv_F32

!Filename = !llvm.ptr<i8>

#COO = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<trim(0,1)>
}>

#CSR = #unisparse.encoding<{
  crdMap = #unisparse.crd<(i,j)->(i,j)>,
  compressMap = #unisparse.compress<fuse(0), trim(1,1)>
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

//   func.func @kernel_csr_spmv(%arg0: tensor<?x?xf32, #CSR>, %arg1: tensor<?xf32>, %argx: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = linalg.generic #trait1
//     ins(%arg0, %arg1 : tensor<?x?xf32, #CSR>, tensor<?xf32>)
//     outs(%argx: tensor<?xf32>) {
//     ^bb0(%a: f32, %x: f32, %o: f32):
//       %2 = arith.mulf %a, %x : f32
//       %3 = arith.addf %o, %2 : f32
//       linalg.yield %3 : f32
//     } -> tensor<?xf32>
//     return %0 : tensor<?xf32>
//   }

  //CHECK-LABEL: func.func @main
  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %i1 = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = unisparse.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %c256 = tensor.dim %A_0, %c1 : tensor<?x?xf32, #COO>
    %a = unisparse.convert (%A_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #CSR>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %ts_dim_i = arith.index_cast %c256 : index to i32
    %ts_dim = arith.sitofp %ts_dim_i : i32 to f32
    %elm = arith.divf %i1, %ts_dim : f32
    %x = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?xf32> {
      // %k0 = arith.muli %i, %c1 : index
      // %k1 = arith.index_cast %k0 : index to i32
      // %k1 = arith.index_cast %i : index to i32
      // %k = arith.sitofp %k1 : i32 to f32
      %t3 = tensor.insert %elm into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    %y_4_4 = bufferization.alloc_tensor(%c256) : tensor<?xf32>
    %y = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %y_4_4) -> tensor<?xf32> {
      %t3 = tensor.insert %i0 into %t[%i] : tensor<?xf32>
      scf.yield %t3 : tensor<?xf32>
    }

    // %t_start4 = call @rtclock() : () -> f64
    %0 = unisparse.device (%a, %x, %y) {target = "HLS"}: 
      (tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32>) to tensor<?xf32> {
      // // read in
      // %a_d0_ptr = unisparse.read_ptr (%a) : tensor<?x?xf32> -> tensor<?xf32> 
      // %a_d1_crd = unisparse.read_crd (%a) : tensor<?x?xf32> -> tensor<?xf32>
      // %a_val = unisparse.read_val (%a) : tensor<?x?xf32> -> tensor<?xf32>
      // %x_val = unisparse.read_val (%x) : tensor<?xf32> -> tensor<?xf32>
      // // decode sparse matrix indices
      // %a_d0_crd = unisparse.repeater (%a_d0_ptr) : tensor<?xf32> -> tensor<?xf32> // subject to change
      // %a_ori_d0, %a_ori_d1 = unisparse.index_calc (%a_d0_crd, %a_d1_crd) : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>, tensor<?xf32>
      // // compute
      // %y_d0, %psum = unisparse.PE_mul (%a_ori_d0, %a_ori_d1, %a_val, %x_val) : 
      //   tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>, tensor<?xf32>
      // %y = unisparse.PE_accum (%y_d0, %psum):
      //   tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
      // // write out
      // unisparse.write_val (%y) : tensor<?xf32>
      // // generate top
      // unisparse.top (%a, %x, %y) : tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32>
      // // unisparse.terminator
    }
    // %0 = call @kernel_csr_spmv(%a, %x, %y) : (tensor<?x?xf32, #CSR>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    // %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v0 = vector.transfer_read %0[%c0], %i0: tensor<?xf32>, vector<4xf32>
    vector.print %v0 : vector<4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?xf32>
//    bufferization.dealloc_tensor %y_4_4 : tensor<?xf32>
    return
  }
}
