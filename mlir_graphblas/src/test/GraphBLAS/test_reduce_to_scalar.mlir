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

func @main() -> () {
    %mat_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3]
      ]> : tensor<2x4xi64>
    %mat_csr = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %mat_csc = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>

    %dense_vec = arith.constant dense<[0, 7, 4, 0, 8, 0, 6, 5]> : tensor<8xi64>
    %vec = sparse_tensor.convert %dense_vec : tensor<8xi64> to tensor<?xi64, #CV64>
    
    %answer_1 = graphblas.reduce_to_scalar %mat_csr { aggregator = "plus" } : tensor<?x?xi64, #CSR64> to i64
    // CHECK: answer_1 6
    graphblas.print %answer_1 { strings = ["answer_1 "] } : i64
    
    %answer_2 = graphblas.reduce_to_scalar %mat_csc { aggregator = "max" } : tensor<?x?xi64, #CSC64> to i64
    // CHECK: answer_2 3
    graphblas.print %answer_2 { strings = ["answer_2 "] } : i64

    %answer_3 = graphblas.reduce_to_scalar %vec { aggregator = "argmax" } : tensor<?xi64, #CV64> to i64
    // CHECK: answer_3 4
    graphblas.print %answer_3 { strings = ["answer_3 "] } : i64

    return
}

// COM: TODO write tests for all tensor element types
