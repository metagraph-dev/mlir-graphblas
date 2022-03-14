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

// COM: linalg-lower doesn't has a bug which affects reduceToVector with lex_insert order
// COM: It should work like CSR@CSC, but it doesn't

func @main() -> () {
    %ci0 = arith.constant 0 : i64

    %mat_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 3, 4]
      ]> : tensor<2x4xi64>
    %mat_csr = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %mat_csc = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>
    
    %mask_2 = arith.constant sparse<[
      [1]
    ], [111]> : tensor<2xi64>
    %mask_2_cv = sparse_tensor.convert %mask_2 : tensor<2xi64> to tensor<?xi64, #CV64>
    
    %mask_4 = arith.constant sparse<[
      [0], [2]
    ], [0, 200]> : tensor<4xi64>
    %mask_4_cv = sparse_tensor.convert %mask_4 : tensor<4xi64> to tensor<?xi64, #CV64>
    
    %answer_1 = graphblas.reduce_to_vector_generic %mat_csr, %mask_2_cv { axis = 1, mask_complement = true } : tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64> to tensor<?xi64, #CV64> {
        graphblas.yield agg_identity %ci0 : i64
    },  {
      ^bb0(%a : i64, %b : i64):
        %result = arith.addi %a, %b : i64
        graphblas.yield agg %result : i64
    }
    // CHECK: answer_1 [3, _]
    graphblas.print %answer_1 { strings = ["answer_1 "] } : tensor<?xi64, #CV64>

    %answer_2 = graphblas.reduce_to_vector_generic %mat_csc, %mask_4_cv { axis = 0 } : tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64> to tensor<?xi64, #CV64> {
        graphblas.yield agg_identity %ci0 : i64
    },  {
      ^bb0(%a : i64, %b : i64):
        %result = arith.addi %a, %b : i64
        graphblas.yield agg %result : i64
    }
    // CHECK-NEXT: answer_2 [_, _, 5, _]
    graphblas.print %answer_2 { strings = ["answer_2 "] } : tensor<?xi64, #CV64>

    %answer_20 = graphblas.reduce_to_vector %mat_csc { aggregator = "max", axis = 0 } : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_20 [_, 1, 3, 4]
    graphblas.print %answer_20 { strings = ["answer_20 "] } : tensor<?xi64, #CV64>

    return
}

// COM: TODO write tests for all tensor element types
