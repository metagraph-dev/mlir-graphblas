// RUN: graphblas-opt %s | graphblas-exec entry | FileCheck %s
// RUN: graphblas-opt %s | graphblas-linalg-exec entry | FileCheck %s

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

    // CSR select upper triangle
    //
    // CHECK:      shape=(4, 5)
    // CHECK-NEXT: pointers=(0, 2, 4, 4, 4)
    // CHECK-NEXT: indices=(1, 2, 3, 4)
    // CHECK-NEXT: values=(1, 2, 4, 5)
    //
    %0 = graphblas.select %m_csr { selector="triu" } : tensor<?x?xf64, #CSR> to tensor<?x?xf64, #CSR>
    graphblas.print_tensor %0 { level=4 } : tensor<?x?xf64, #CSR>

    return
  }
}