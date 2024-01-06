module {
  func.func private @delSparseTensor(!llvm.ptr<i8>)
  func.func private @endInsert(!llvm.ptr<i8>)
  func.func private @expInsertF64(!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index) attributes {llvm.emit_c_interface}
  func.func private @sparseDimSize(!llvm.ptr<i8>, index) -> index
  func.func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @sparseIndices0(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @sparsePointers0(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @newSparseTensor(memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @getTensorFilename(index) -> !llvm.ptr<i8>
  func.func @kernel_csr_spgemm(%arg0: !llvm.ptr<i8>, %arg1: !llvm.ptr<i8>, %arg2: index, %arg3: index) -> !llvm.ptr<i8> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c2_0 = arith.constant 2 : index
    %0 = memref.alloca(%c2_0) : memref<?xi8>
    %c0_1 = arith.constant 0 : index
    memref.store %c0_i8, %0[%c0_1] : memref<?xi8>
    %c1_2 = arith.constant 1 : index
    memref.store %c1_i8, %0[%c1_2] : memref<?xi8>
    %c2_3 = arith.constant 2 : index
    %1 = memref.alloca(%c2_3) : memref<?xindex>
    %c0_4 = arith.constant 0 : index
    memref.store %arg2, %1[%c0_4] : memref<?xindex>
    %c1_5 = arith.constant 1 : index
    memref.store %arg3, %1[%c1_5] : memref<?xindex>
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %c2_8 = arith.constant 2 : index
    %2 = memref.alloca(%c2_8) : memref<?xindex>
    %c0_9 = arith.constant 0 : index
    memref.store %c0_6, %2[%c0_9] : memref<?xindex>
    %c1_10 = arith.constant 1 : index
    memref.store %c1_7, %2[%c1_10] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %3 = llvm.mlir.null : !llvm.ptr<i8>
    %4 = call @newSparseTensor(%0, %1, %2, %c0_i32, %c0_i32_11, %c1_i32, %c0_i32_12, %3) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c1_13 = arith.constant 1 : index
    %5 = call @sparseDimSize(%4, %c1_13) : (!llvm.ptr<i8>, index) -> index
    %6 = memref.alloc(%5) : memref<?xf64>
    %7 = memref.alloc(%5) : memref<?xi1>
    %8 = memref.alloc(%5) : memref<?xindex>
    %c0_14 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f64
    linalg.fill ins(%cst : f64) outs(%6 : memref<?xf64>)
    %false_15 = arith.constant false
    linalg.fill ins(%false_15 : i1) outs(%7 : memref<?xi1>)
    %9 = call @sparsePointers0(%arg0, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %10 = call @sparseIndices0(%arg0, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %11 = call @sparseValuesF64(%arg0) : (!llvm.ptr<i8>) -> memref<?xf64>
    %12 = call @sparsePointers0(%arg1, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %13 = call @sparseIndices0(%arg1, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
    %14 = call @sparseValuesF64(%arg1) : (!llvm.ptr<i8>) -> memref<?xf64>
    %c0_16 = arith.constant 0 : index
    %15 = call @sparseDimSize(%4, %c0_16) : (!llvm.ptr<i8>, index) -> index
    %16 = memref.alloca(%c2) : memref<?xindex>
    scf.for %arg4 = %c0 to %15 step %c1 {
      memref.store %arg4, %16[%c0] : memref<?xindex>
      %17 = memref.load %9[%arg4] : memref<?xindex>
      %18 = arith.addi %arg4, %c1 : index
      %19 = memref.load %9[%18] : memref<?xindex>
      %20 = scf.for %arg5 = %17 to %19 step %c1 iter_args(%arg6 = %c0_14) -> (index) {
        %21 = memref.load %10[%arg5] : memref<?xindex>
        %22 = memref.load %11[%arg5] : memref<?xf64>
        %23 = memref.load %12[%21] : memref<?xindex>
        %24 = arith.addi %21, %c1 : index
        %25 = memref.load %12[%24] : memref<?xindex>
        %26 = scf.for %arg7 = %23 to %25 step %c1 iter_args(%arg8 = %arg6) -> (index) {
          %27 = memref.load %13[%arg7] : memref<?xindex>
          %28 = memref.load %6[%27] : memref<?xf64>
          %29 = memref.load %14[%arg7] : memref<?xf64>
          %30 = arith.mulf %22, %29 : f64
          %31 = arith.addf %28, %30 : f64
          %32 = memref.load %7[%27] : memref<?xi1>
          %33 = arith.cmpi eq, %32, %false : i1
          %34 = scf.if %33 -> (index) {
            memref.store %true, %7[%27] : memref<?xi1>
            memref.store %27, %8[%arg8] : memref<?xindex>
            %35 = arith.addi %arg8, %c1 : index
            scf.yield %35 : index
          } else {
            scf.yield %arg8 : index
          }
          memref.store %31, %6[%27] : memref<?xf64>
          scf.yield %34 : index
        }
        scf.yield %26 : index
      }
      func.call @expInsertF64(%4, %16, %6, %7, %8, %20) : (!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index) -> ()
    }
    memref.dealloc %6 : memref<?xf64>
    memref.dealloc %7 : memref<?xi1>
    memref.dealloc %8 : memref<?xindex>
    call @endInsert(%4) : (!llvm.ptr<i8>) -> ()
    return %4 : !llvm.ptr<i8>
  }
  func.func @main() {
    %cst = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c2 = arith.constant 2 : index
    %1 = memref.alloca(%c2) : memref<?xi8>
    %c0_2 = arith.constant 0 : index
    memref.store %c0_i8, %1[%c0_2] : memref<?xi8>
    %c1_3 = arith.constant 1 : index
    memref.store %c1_i8, %1[%c1_3] : memref<?xi8>
    %c2_4 = arith.constant 2 : index
    %2 = memref.alloca(%c2_4) : memref<?xindex>
    %c0_5 = arith.constant 0 : index
    memref.store %c0_0, %2[%c0_5] : memref<?xindex>
    %c1_6 = arith.constant 1 : index
    memref.store %c0_1, %2[%c1_6] : memref<?xindex>
    %c0_7 = arith.constant 0 : index
    %c1_8 = arith.constant 1 : index
    %c2_9 = arith.constant 2 : index
    %3 = memref.alloca(%c2_9) : memref<?xindex>
    %c0_10 = arith.constant 0 : index
    memref.store %c0_7, %3[%c0_10] : memref<?xindex>
    %c1_11 = arith.constant 1 : index
    memref.store %c1_8, %3[%c1_11] : memref<?xindex>
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_13 = arith.constant 1 : i32
    %4 = call @newSparseTensor(%1, %2, %3, %c0_i32, %c0_i32_12, %c1_i32, %c1_i32_13, %0) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_14 = arith.constant 0 : index
    %c0_15 = arith.constant 0 : index
    %c0_i8_16 = arith.constant 0 : i8
    %c1_i8_17 = arith.constant 1 : i8
    %c2_18 = arith.constant 2 : index
    %5 = memref.alloca(%c2_18) : memref<?xi8>
    %c0_19 = arith.constant 0 : index
    memref.store %c0_i8_16, %5[%c0_19] : memref<?xi8>
    %c1_20 = arith.constant 1 : index
    memref.store %c1_i8_17, %5[%c1_20] : memref<?xi8>
    %c2_21 = arith.constant 2 : index
    %6 = memref.alloca(%c2_21) : memref<?xindex>
    %c0_22 = arith.constant 0 : index
    memref.store %c0_14, %6[%c0_22] : memref<?xindex>
    %c1_23 = arith.constant 1 : index
    memref.store %c0_15, %6[%c1_23] : memref<?xindex>
    %c0_24 = arith.constant 0 : index
    %c1_25 = arith.constant 1 : index
    %c2_26 = arith.constant 2 : index
    %7 = memref.alloca(%c2_26) : memref<?xindex>
    %c0_27 = arith.constant 0 : index
    memref.store %c0_24, %7[%c0_27] : memref<?xindex>
    %c1_28 = arith.constant 1 : index
    memref.store %c1_25, %7[%c1_28] : memref<?xindex>
    %c0_i32_29 = arith.constant 0 : i32
    %c0_i32_30 = arith.constant 0 : i32
    %c1_i32_31 = arith.constant 1 : i32
    %c1_i32_32 = arith.constant 1 : i32
    %8 = call @newSparseTensor(%5, %6, %7, %c0_i32_29, %c0_i32_30, %c1_i32_31, %c1_i32_32, %0) : (memref<?xi8>, memref<?xindex>, memref<?xindex>, i32, i32, i32, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %c0_33 = arith.constant 0 : index
    %9 = call @sparseDimSize(%4, %c0_33) : (!llvm.ptr<i8>, index) -> index
    %c1_34 = arith.constant 1 : index
    %10 = call @sparseDimSize(%8, %c1_34) : (!llvm.ptr<i8>, index) -> index
    %11 = call @rtclock() : () -> f64
    %12 = call @kernel_csr_spgemm(%4, %8, %9, %10) : (!llvm.ptr<i8>, !llvm.ptr<i8>, index, index) -> !llvm.ptr<i8>
    %13 = call @rtclock() : () -> f64
    %14 = arith.subf %13, %11 : f64
    vector.print %14 : f64
    %15 = call @sparseValuesF64(%12) : (!llvm.ptr<i8>) -> memref<?xf64>
    %16 = vector.transfer_read %15[%c0], %cst : memref<?xf64>, vector<8xf64>
    %17 = memref.dim %15, %c0 : memref<?xf64>
    vector.print %16 : vector<8xf64>
    vector.print %17 : index
    call @delSparseTensor(%4) : (!llvm.ptr<i8>) -> ()
    call @delSparseTensor(%8) : (!llvm.ptr<i8>) -> ()
    return
  }
}

