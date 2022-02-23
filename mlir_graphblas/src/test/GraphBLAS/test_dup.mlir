// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    ///////////////
    // Test Matrix
    ///////////////

    %m = arith.constant sparse<[
      [0, 1], [0, 2],
      [1, 0], [1, 3], [1, 4],
      [3, 2]
    ], [1., 2., 3., 4., 5., 6.]> : tensor<4x5xf64>
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR>
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC>

    // CSR dup: make a duplicate, modify a value, print and check modified and original
    //
    // CHECK:      pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(33.33, 2, 3, 4, 5, 6)
    // CHECK-NEXT: values=(1, 2, 3, 4, 5, 6)
    //
    %0 = graphblas.dup %m_csr : tensor<?x?xf64, #CSR>
    %1 = sparse_tensor.values %0 : tensor<?x?xf64, #CSR> to memref<?xf64>
    %2 = arith.constant 33.33 : f64
    memref.store %2, %1[%c0] : memref<?xf64>
    graphblas.print_tensor %0 { level=3 } : tensor<?x?xf64, #CSR>
    graphblas.print_tensor %m_csr { level=1 } : tensor<?x?xf64, #CSR>

    // CSC dup: make a duplicate, modify a value, print and check modified and original
    //
    // CHECK:      pointers=(0, 1, 2, 4, 5, 6)
    // CHECK-NEXT: indices=(1, 0, 0, 3, 1, 1)
    // CHECK-NEXT: values=(3, 44.44, 2, 6, 4, 5)
    // CHECK-NEXT: values=(3, 1, 2, 6, 4, 5)
    //
    %10 = graphblas.dup %m_csc : tensor<?x?xf64, #CSC>
    %11 = sparse_tensor.values %10 : tensor<?x?xf64, #CSC> to memref<?xf64>
    %12 = arith.constant 44.44 : f64
    memref.store %12, %11[%c1] : memref<?xf64>
    graphblas.print_tensor %10 { level=3 } : tensor<?x?xf64, #CSC>
    graphblas.print_tensor %m_csc { level=1 } : tensor<?x?xf64, #CSC>


    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1, 2, 3, 4]> : tensor<9xi32>
    %v_cv = sparse_tensor.convert %v : tensor<9xi32> to tensor<?xi32, #CV>

    // CV dup: make a duplicate, modify an index, print and check modified and original
    //
    // CHECK:      indices=(1, 3, 4, 7)
    // CHECK-NEXT: values=(1, 2, 3, 4)
    // CHECK-NEXT: indices=(1, 2, 4, 7)
    // CHECK-NEXT: values=(1, 2, 3, 4)
    //
    %20 = graphblas.dup %v_cv : tensor<?xi32, #CV>
    %21 = sparse_tensor.indices %20, %c0 : tensor<?xi32, #CV> to memref<?xi64>
    %22 = arith.constant 3 : i64
    memref.store %22, %21[%c1] : memref<?xi64>
    graphblas.print_tensor %20 { level=2 } : tensor<?xi32, #CV>
    graphblas.print_tensor %v_cv { level=2 } : tensor<?xi32, #CV>

    return
  }
}