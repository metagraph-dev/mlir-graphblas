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

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
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

      %different_compression_answer = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
      
      // CHECK: different_compression_answer [
      // CHECK:   [_, _],
      // CHECK:   [1, _],
      // CHECK:   [2, _],
      // CHECK:   [_, 3],
      // CHECK: ]
      graphblas.print %different_compression_answer { strings = ["different_compression_answer "] } : tensor<?x?xi64, #CSC64>

      %same_compression_answer = graphblas.transpose %mat : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>

      // CHECK: same_compression_answer [
      // CHECK:   [_, _],
      // CHECK:   [1, _],
      // CHECK:   [2, _],
      // CHECK:   [_, 3],
      // CHECK: ]
      graphblas.print %same_compression_answer { strings = ["same_compression_answer "] } : tensor<?x?xi64, #CSR64>
      
      return
  }
}
