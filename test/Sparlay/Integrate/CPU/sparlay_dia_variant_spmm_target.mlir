// sparlay-opt sparlay_dia_variant_spmm_target.mlir -lower-format-conversion -lower-struct -dce | \
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" \
// -finalizing-bufferize -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -lower-affine \
// -convert-vector-to-llvm -convert-memref-to-llvm -convert-complex-to-standard -convert-math-to-llvm \
// -convert-math-to-libm -convert-complex-to-libm -convert-complex-to-llvm -convert-func-to-llvm \
// -reconcile-unrealized-casts  | mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 -relocation-model=pic -filetype=obj -o dia_v_spmm.o
    
// clang++ dia_v_spmm.o -L$SPLHOME/build/lib -lmlir_sparlay_runner_utils \
//     -L$LLVMHOME/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o dia_v_spmm

// ./dia_v_spmm

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

#DIA_variant = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i,j)>,
  compressMap = #sparlay.compress<trim(0,0)>
}>

#amap_0 = affine_map<(i,j) -> (i, j)>

// #trait1 = {
// indexing_maps = [
//     #amap_0,  // A
//     affine_map<(i,j) -> (j)>,  // B
//     affine_map<(i,j) -> (i)>   // X (out)
//   ],
//   iterator_types = ["parallel", "reduction"],
//   doc = "X(i) =+ A(i,j) * B(j)"
// }

#trait1 = {
indexing_maps = [
    affine_map<(i,j,k) -> (i, k)>,  // A
    affine_map<(i,j,k) -> (k, j)>,  // B
    affine_map<(i,j,k) -> (i, j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) =+ A(i,k) * B(k, j)"
}

module {
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> (!Filename)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
  func.func @kernel_dia_v_spmm(%arg0: tensor<?x?xf32, #DIA_variant>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // %0 = linalg.generic #trait1
    // ins(%arg0, %arg1 : tensor<?x?xf32, #DIA_variant>, tensor<?x?xf32>)
    // outs(%argx: tensor<?x?xf32>) {
    // ^bb0(%a: f32, %b: f32, %x: f32):
    //   %2 = arith.mulf %a, %b : f32
    //   %3 = arith.addf %x, %2 : f32
    //   linalg.yield %3 : f32
    // } -> tensor<?x?xf32>
    // return %0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32
    %crd0 = sparlay.crd %arg0, %c1 : tensor<?x?xf32, #DIA_variant> to memref<?xi32>
    %diag_num = memref.dim %crd0, %c0 : memref<?xi32>
    %crd0_print = memref.cast %crd0 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%crd0_print) : (memref<*xi32>) -> ()
    %diag_size = sparlay.size %arg0, %c1 : tensor<?x?xf32, #DIA_variant> to index
    vector.print %diag_size: index
    %val = sparlay.value %arg0, %c0: tensor<?x?xf32, #DIA_variant> to memref<?xf32>
    %val_print = memref.cast %val : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%val_print) : (memref<*xf32>) -> ()
    %mem_b = bufferization.to_memref %arg1 : memref<?x?xf32>
    %dim_d0 = tensor.dim %arg2, %c0 : tensor<?x?xf32>
    %dim_d2 = tensor.dim %arg2, %c1 : tensor<?x?xf32>
    %mem_c = bufferization.to_memref %arg2 : memref<?x?xf32>
    scf.for %arg3 = %c0 to %dim_d2 step %c1 {
      scf.for %arg4 = %c0 to %diag_num step %c1 {
        %diag_offset = memref.load %crd0[%arg4] : memref<?xi32>
        scf.for %arg5 = %c0 to %diag_size step %c1 {
          %d1 = arith.index_cast %arg5 : index to i32
          %dim_d0_i32 = arith.index_cast %dim_d0 : index to i32
          %d0 = arith.subi %d1, %diag_offset : i32
          %cond1 = arith.cmpi slt, %d0, %dim_d0_i32: i32
          %cond2 = arith.cmpi sge, %d0, %i0 : i32
          %cond = arith.andi %cond1, %cond2: i1
          scf.if %cond {
            %prod = arith.muli %arg4, %diag_size: index
            %sum = arith.addi %prod, %arg5: index
            %val_A = memref.load %val[%sum] : memref<?xf32>
            %val_B = memref.load %mem_b[%arg5, %arg3] : memref<?x?xf32>
            %d0_index = arith.index_cast %d0 : i32 to index
            %val_C = memref.load %mem_c[%d0_index, %arg3] : memref<?x?xf32>
            %val_prod = arith.mulf %val_A, %val_B : f32
            %val_sum = arith.addf %val_prod, %val_C : f32
            memref.store %val_sum, %mem_c[%d0_index, %arg3] : memref<?x?xf32>
          }
        }
      }
    }
    %result = bufferization.to_tensor %mem_c : memref<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  func.func @main() {
    %i0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %t_start0 = call @rtclock() : () -> f64
    %A_0 = sparlay.fromFile (%fileName) : !Filename to tensor<?x?xf32, #COO>
    %c256 = tensor.dim %A_0, %c1 : tensor<?x?xf32, #COO>
    %a0 = sparlay.convert (%A_0): tensor<?x?xf32, #COO> to tensor<?x?xf32, #DIA_variant>
    %t_end0 = call @rtclock() : () -> f64
    %t_0 = arith.subf %t_end0, %t_start0: f64
    vector.print %t_0 : f64

    // Initialize dense matrix.
    %init_256_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>

    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256_4) -> tensor<?x?xf32> {
      %b2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %k0 = arith.muli %i, %c4 : index
        %k1 = arith.addi %j, %k0 : index
        %k2 = arith.index_cast %k1 : index to i32
        %k = arith.sitofp %k2 : i32 to f32
        %t3 = tensor.insert %k into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %b2 : tensor<?x?xf32>
    }

    %o0_4_4 = bufferization.alloc_tensor(%c256, %c4) : tensor<?x?xf32>
    %o0 = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %o0_4_4) -> tensor<?x?xf32> {
      %x2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%t2 = %t) -> tensor<?x?xf32> {
        %t3 = tensor.insert %i0 into %t2[%i, %j] : tensor<?x?xf32>
        scf.yield %t3 : tensor<?x?xf32>
      }
      scf.yield %x2 : tensor<?x?xf32>
    }

    %t_start4 = call @rtclock() : () -> f64
    %0 = call @kernel_dia_v_spmm(%a0, %b, %o0) : (tensor<?x?xf32, #DIA_variant>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %t_end4 = call @rtclock() : () -> f64
    %t_4 = arith.subf %t_end4, %t_start4: f64
    vector.print %t_4 : f64
    %v0 = vector.transfer_read %0[%c0, %c0], %i0: tensor<?x?xf32>, vector<4x4xf32>
    vector.print %v0 : vector<4x4xf32>

    //Release the resources 
    bufferization.dealloc_tensor %A_0 : tensor<?x?xf32, #COO>
//    bufferization.dealloc_tensor %init_256_4 : tensor<?x?xf32>
//    bufferization.dealloc_tensor %o0_4_4 : tensor<?x?xf32>
    return
  }
}