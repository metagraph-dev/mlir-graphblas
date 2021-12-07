// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @entry() {

    ///////////////
    // Test Matrix
    ///////////////

    %m = arith.constant sparse<[
      [0, 1], [0, 2],
      [1, 0], [1, 3], [1, 4],
      [3, 2]
    ], [1., 2., 3., 4., 5., 6.]> : tensor<4x5xf64>
    %m_csr = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSR64>
    %m_csc = sparse_tensor.convert %m : tensor<4x5xf64> to tensor<?x?xf64, #CSC64>

    // CSR uniform complement
    //
    // CHECK:      shape=(4, 5)
    // CHECK:      pointers=(0, 3, 5, 10, 14)
    // CHECK-NEXT: indices=(0, 3, 4, 1, 2, 0, 1, 2, 3, 4, 0, 1, 3, 4)
    // CHECK-NEXT: values=(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    //
    %cf2 = arith.constant 2.0 : f64
    %0 = graphblas.uniform_complement %m_csr, %cf2 : tensor<?x?xf64, #CSR64>, f64 to tensor<?x?xf64, #CSR64>
    graphblas.print_tensor %0 { level=4 } : tensor<?x?xf64, #CSR64>

    // CSC uniform_complement different element type
    //
    // CHECK:      rev=(1, 0)
    // CHECK:      shape=(4, 5)
    // CHECK:      pointers=(0, 3, 6, 8, 11, 14)
    // CHECK-NEXT: indices=(0, 2, 3, 1, 2, 3, 1, 2, 0, 2, 3, 0, 2, 3)
    // CHECK-NEXT: values=(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9)
    //
    %ci9 = arith.constant 9 : i64
    %10 = graphblas.uniform_complement %m_csc, %ci9 : tensor<?x?xf64, #CSC64>, i64 to tensor<?x?xi64, #CSC64>
    graphblas.print_tensor %10 { level=5 } : tensor<?x?xi64, #CSC64>

    ///////////////
    // Test Vector
    ///////////////

    %v = arith.constant sparse<[
      [1], [2], [4], [7]
    ], [1., 2., 3., 4.]> : tensor<9xf64>
    %v_cv = sparse_tensor.convert %v : tensor<9xf64> to tensor<?xf64, #CV64>

    // Vector uniform_complement
    //
    // CHECK:      shape=(9)
    // CHECK:      pointers=(0, 5)
    // CHECK-NEXT: indices=(0, 3, 5, 6, 8)
    // CHECK-NEXT: values=(1, 1, 1, 1, 1)
    //
    %ci1 = arith.constant 1 : i32
    %20 = graphblas.uniform_complement %v_cv, %ci1 : tensor<?xf64, #CV64>, i32 to tensor<?xi32, #CV64>
    graphblas.print_tensor %20 { level=4 } : tensor<?xi32, #CV64>

    return
  }
}