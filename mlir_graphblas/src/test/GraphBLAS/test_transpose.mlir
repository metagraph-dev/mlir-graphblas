// RUN: graphblas-opt %s | graphblas-exec functional_transpose | FileCheck %s

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

module {
  func @functional_transpose() -> () {
      %dense_mat = arith.constant dense<[
          [0, 1, 2, 0],
          [0, 0, 0, 3]
        ]> : tensor<2x4xi64>
      %mat = sparse_tensor.convert %dense_mat : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
      %mat_csc = sparse_tensor.convert %dense_mat : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>

      // CHECK: csr_to_csc_transpose [
      // CHECK-NEXT:   [_, _],
      // CHECK-NEXT:   [1, _],
      // CHECK-NEXT:   [2, _],
      // CHECK-NEXT:   [_, 3]
      // CHECK-NEXT: ]
      %csr_to_csc_transpose = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
      graphblas.print %csr_to_csc_transpose { strings = ["csr_to_csc_transpose "] } : tensor<?x?xi64, #CSC64>

      // CHECK: csc_to_csr_transpose [
      // CHECK-NEXT:   [_, _],
      // CHECK-NEXT:   [1, _],
      // CHECK-NEXT:   [2, _],
      // CHECK-NEXT:   [_, 3]
      // CHECK-NEXT: ]
      %csc_to_csr_transpose = graphblas.transpose %mat_csc : tensor<?x?xi64, #CSC64> to tensor<?x?xi64, #CSR64>
      graphblas.print %csc_to_csr_transpose { strings = ["csc_to_csr_transpose "] } : tensor<?x?xi64, #CSR64>

      // CHECK: csr_to_csr_transpose [
      // CHECK-NEXT:   [_, _],
      // CHECK-NEXT:   [1, _],
      // CHECK-NEXT:   [2, _],
      // CHECK-NEXT:   [_, 3]
      // CHECK-NEXT: ]
      %csr_to_csr_transpose = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>
      graphblas.print %csr_to_csr_transpose { strings = ["csr_to_csr_transpose "] } : tensor<?x?xi64, #CSR64>

      // CHECK: csc_to_csc_transpose [
      // CHECK-NEXT:   [_, _],
      // CHECK-NEXT:   [1, _],
      // CHECK-NEXT:   [2, _],
      // CHECK-NEXT:   [_, 3]
      // CHECK-NEXT: ]
      %csc_to_csc_transpose = graphblas.transpose %mat_csc : tensor<?x?xi64, #CSC64> to tensor<?x?xi64, #CSC64>
      graphblas.print %csc_to_csc_transpose { strings = ["csc_to_csc_transpose "] } : tensor<?x?xi64, #CSC64>

      return
  }
}
