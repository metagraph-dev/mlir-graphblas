// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

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
  func @main() -> () {
    %dense_mat = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3],
        [4, 0, 5, 0],
        [0, 6, 0, 7]
      ]> : tensor<4x4xi64>
    %mat_csr = sparse_tensor.convert %dense_mat : tensor<4x4xi64> to tensor<?x?xi64, #CSR64>
    %mat_csc = sparse_tensor.convert %dense_mat : tensor<4x4xi64> to tensor<?x?xi64, #CSC64>

    %answer_via_csr = graphblas.diag %mat_csr : tensor<?x?xi64, #CSR64> to tensor<?xi64, #CV64>
    // CHECK: %answer_via_csr [_, _, 5, 7]
    graphblas.print %answer_via_csr { strings = ["%answer_via_csr "] } : tensor<?xi64, #CV64>
    
    %answer_via_csc = graphblas.diag %mat_csc : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
    // CHECK: %answer_via_csc [_, _, 5, 7]
    graphblas.print %answer_via_csc { strings = ["%answer_via_csc "] } : tensor<?xi64, #CV64>
    
    return
  }
}
