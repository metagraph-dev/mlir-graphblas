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

    // CSR -> CSC
    //
    // CHECK:      rev=(1, 0)
    // CHECK-NEXT: shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 1, 2, 4, 5, 6)
    // CHECK-NEXT: indices=(1, 0, 0, 3, 1, 1)
    // CHECK-NEXT: values=(3, 1, 2, 6, 4, 5)
    //
    %0 = graphblas.convert_layout %m_csr : tensor<?x?xf64, #CSR> to tensor<?x?xf64, #CSC>
    graphblas.print_tensor %0 { level=5 } : tensor<?x?xf64, #CSC>

    // CSC -> CSR
    //
    // CHECK:      rev=(0, 1)
    // CHECK-NEXT: shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 3, 4, 5, 6)
    //
    %10 = graphblas.convert_layout %m_csc : tensor<?x?xf64, #CSC> to tensor<?x?xf64, #CSR>
    graphblas.print_tensor %10 { level=5 } : tensor<?x?xf64, #CSR>

    // CSC -> CSC (should be unchanged)
    //
    // CHECK:      rev=(1, 0)
    // CHECK-NEXT: shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 1, 2, 4, 5, 6)
    // CHECK-NEXT: indices=(1, 0, 0, 3, 1, 1)
    // CHECK-NEXT: values=(3, 1, 2, 6, 4, 5)
    //
    %20 = graphblas.convert_layout %m_csc : tensor<?x?xf64, #CSC> to tensor<?x?xf64, #CSC>
    graphblas.print_tensor %20 { level=5 } : tensor<?x?xf64, #CSC>

    // CSR -> CSR (should be unchanged)
    //
    // CHECK:      rev=(0, 1)
    // CHECK-NEXT: shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 2, 5, 5, 6)
    // CHECK-NEXT: indices=(1, 2, 0, 3, 4, 2)
    // CHECK-NEXT: values=(1, 2, 3, 4, 5, 6)
    //
    %30 = graphblas.convert_layout %10 : tensor<?x?xf64, #CSR> to tensor<?x?xf64, #CSR>
    graphblas.print_tensor %30 { level=5 } : tensor<?x?xf64, #CSR>

    return
  }
}
