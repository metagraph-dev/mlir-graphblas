// RUN: graphblas-opt %s | graphblas-exec functional_transpose | FileCheck %s
// RUN: graphblas-opt %s | graphblas-linalg-exec functional_transpose | FileCheck %s

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

      // CHECK:      pointers=(0, 2, 3)
      // CHECK-NEXT: indices=(1, 2, 3)
      // CHECK-NEXT: values=(1, 2, 3)
      %csr_to_csc_transpose = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
      graphblas.print_tensor %csr_to_csc_transpose { level=3 } : tensor<?x?xi64, #CSC64>

      // CHECK:      pointers=(0, 0, 1, 2, 3)
      // CHECK-NEXT: indices=(0, 0, 1)
      // CHECK-NEXT: values=(1, 2, 3)
      %csc_to_csr_transpose = graphblas.transpose %mat_csc : tensor<?x?xi64, #CSC64> to tensor<?x?xi64, #CSR64>
      graphblas.print_tensor %csc_to_csr_transpose { level=3 } : tensor<?x?xi64, #CSR64>

      // CHECK:      pointers=(0, 0, 1, 2, 3)
      // CHECK-NEXT: indices=(0, 0, 1)
      // CHECK-NEXT: values=(1, 2, 3)
      %csr_to_csr_transpose = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>
      graphblas.print_tensor %csr_to_csr_transpose { level=3 } : tensor<?x?xi64, #CSR64>

      // CHECK:      pointers=(0, 2, 3)
      // CHECK-NEXT: indices=(1, 2, 3)
      // CHECK-NEXT: values=(1, 2, 3)
      %csc_to_csc_transpose = graphblas.transpose %mat_csc : tensor<?x?xi64, #CSC64> to tensor<?x?xi64, #CSC64>
      graphblas.print_tensor %csc_to_csc_transpose { level=3 } : tensor<?x?xi64, #CSC64>

      return
  }
}
