!Filename = !llvm.ptr<i8>

#COO = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<trim(0,1)>
}>

#CSR = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(i,j)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#CSC = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j,i)>,
  compressMap = #sparlay.compress<fuse(0), trim(1,1)>
}>

#DIA = #sparlay.encoding<{
  crdMap = #sparlay.crd<(i,j)->(j minus i,i)>,
  compressMap = #sparlay.compress<trim(0,0)>
}>

#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>,   // b
    affine_map<(i,j) -> (i)>    // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that multiplies a sparse matrix A with a dense vector b
  // into a dense vector x.
  //
  func.func @kernel_matvec(%arga: tensor<?x?xi32, #DIA>,
                           %argb: tensor<?xi32>,
                           %argx: tensor<?xi32>)
		      -> tensor<?xi32> {
    %0 = linalg.generic #matvec
      ins(%arga, %argb: tensor<?x?xi32, #DIA>, tensor<?xi32>)
      outs(%argx: tensor<?xi32>) {
      ^bb(%a: i32, %b: i32, %x: i32):
        %0 = arith.muli %a, %b : i32
        %1 = arith.addi %x, %0 : i32
        linalg.yield %1 : i32
    } -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %i0 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a_1 = sparlay.fromFile (%fileName): !llvm.ptr<i8> to tensor<?x?xi32, #COO>
    %a = sparlay.convert (%a_1): tensor<?x?xi32, #COO> to tensor<?x?xi32, #DIA>

    // Initialize dense vectors.
    %init_256 = bufferization.alloc_tensor(%c256) : tensor<?xi32>
    %b = scf.for %i = %c0 to %c256 step %c1 iter_args(%t = %init_256) -> tensor<?xi32> {
      %k = arith.addi %i, %c1 : index
      %j = arith.index_cast %k : index to i32
      %t2 = tensor.insert %j into %t[%i] : tensor<?xi32>
      scf.yield %t2 : tensor<?xi32>
    }
    %init_4 = bufferization.alloc_tensor(%c4) : tensor<?xi32>
    %x = scf.for %i = %c0 to %c4 step %c1 iter_args(%t = %init_4) -> tensor<?xi32> {
      %t2 = tensor.insert %i0 into %t[%i] : tensor<?xi32>
      scf.yield %t2 : tensor<?xi32>
    }



    // Call kernel.
    %0 = call @kernel_matvec(%a, %b, %x)
      : (tensor<?x?xi32, #DIA>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

    // Print the result for verification.
    //
    // CHECK: ( 889, 1514, -21, -3431 )
    //
    %v = vector.transfer_read %0[%c0], %i0: tensor<?xi32>, vector<4xi32>
    vector.print %v : vector<4xi32>

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xi32, #DIA>

    return
  }
}