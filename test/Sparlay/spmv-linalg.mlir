// RUN: sparlay-opt %s -lower-format-conversion -lower-struct -dce | \
// RUN: mlir-opt -convert-vector-to-scf --convert-scf-to-std --tensor-constant-bufferize \
// RUN:     --tensor-bufferize --std-bufferize --finalizing-bufferize --convert-vector-to-llvm \
// RUN:     --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-cast | \
// RUN: mlir-translate -mlir-to-llvmir | opt -O3 -S | llc -O3 | as - -o |
// RUN: clang++ -L${YOUR_PROJECT_PATH}/sparlay/build/lib -lmlir_sparlay_runner_utils \
// RUN:     -L${LLVM_PROJECT_PATH}/build/lib -lmlir_runner_utils -lmlir_c_runner_utils -o exec
// RUN: ./exec

!Filename = type !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  secondaryMap = #sparlay.affine<(d0, d1)->(trim d0, d1)>
}>

#CSR = #sparlay.encoding<{
  secondaryMap = #sparlay.affine<(d0, d1)->(fuse d0, trim d1)>
}>

#spmv = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>, // B
    affine_map<(i,j) -> (i)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

module {
    // func @kernel_spmv(%arga: tensor<4x6xf32, #CSR>,
    //                 %argb: tensor<6xf32>,
    //                 %argx: tensor<4xf32> {linalg.inplaceable = true}) -> tensor<4xf32> {
    //     %0 = linalg.generic #spmv
    //         ins(%arga, %argb: tensor<4x6xf32, #CSR>, tensor<6xf32>)
    //         outs(%argx: tensor<4xf32>) {
    //         ^bb(%a: f32, %b: f32, %x: f32):
    //             %0 = mulf %a, %b : f32
    //             %1 = addf %x, %0 : f32
    //             linalg.yield %1 : f32
    //         } -> tensor<4xf32>
    //     return %0 : tensor<4xf32>
    // }

    func private @getTensorFilename(index) -> (!Filename)

    func @main() {
        %i0 = constant 0: index
        %i1 = constant 1: index
        %f0 = constant 0.0e+00: f32
         
        %B = constant dense<[1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00, 1.0e+00]> : tensor<6xf32>
        %C = constant dense<[0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00]> : tensor<4xf32>
 
        %fileName = call @getTensorFilename(%i0) : (index) -> (!Filename)
        %A_input = sparlay.new (%fileName) : !Filename to tensor<4x6xf32, #COO>
            // !sparlay.struct<[4, 6], !sparlay.struct<memref<?xindex>, memref<?xindex>, "crd", (i,j)->(i,j)>, 
            //                  memref<?xf64>>

        %A_CSR = sparlay.convert (%A_input) : tensor<4x6xf32, #COO> to tensor<4x6xf32, #CSR>
        // %A_CSR_col = sparlay.access (%A_CSR) {dim = 1: index} : tensor<4x6xf32, #CSR> to 
        //     !sparlay.struct<!sparlay.struct<memref<?xindex>, "ptr">,
        //                  !sparlay.struct<memref<?xindex>, "crd">>
        
        %0 = linalg.generic #spmv
            ins(%A_CSR, %B: tensor<4x6xf32, #CSR>, tensor<6xf32>)
            outs(%C: tensor<4xf32>) {
            ^bb(%a: f32, %b: f32, %x: f32):
                %0 = mulf %a, %b : f32
                %1 = addf %x, %0 : f32
                linalg.yield %1 : f32
            } -> tensor<4xf32>
        // %0 = call @kernel_spmv(%A_CSR, %B, %C)
        //     : (tensor<4x6xf32, #CSR>, tensor<6xf32>, tensor<4xf32>) -> tensor<4xf32>
        
        // %m = memref.buffer_cast %0 : memref<4xf32>
        // %v = vector.transfer_read %m[%i0], %f0: memref<?xf32>, vector<4xf32>
        // vector.print %v : vector<4xf32>
        return
    }
}

// func private @print_memref_f64(%ptr : memref<4xf64>)
