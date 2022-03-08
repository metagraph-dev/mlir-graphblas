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
    %c0_i64 = arith.constant 0 : i64
    %c_min_i64 = arith.constant -9223372036854775808 : i64

    %mat_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3]
      ]> : tensor<2x4xi64>
    %mat_csr = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %mat_csc = sparse_tensor.convert %mat_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>

    %dense_vec = arith.constant dense<[0, 7, 4, 0, 8, 0, 6, 5]> : tensor<8xi64>
    %vec = sparse_tensor.convert %dense_vec : tensor<8xi64> to tensor<?xi64, #CV64>
    
    %answer_1 = graphblas.reduce_to_scalar_generic %mat_csr : tensor<?x?xi64, #CSR64> to i64 {
      graphblas.yield agg_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %13 = arith.addi %arg0, %arg1 : i64
      graphblas.yield agg %13 : i64
    }
    // CHECK: answer_1 6
    graphblas.print %answer_1 { strings = ["answer_1 "] } : i64
    
    %answer_2 = graphblas.reduce_to_scalar_generic %mat_csc : tensor<?x?xi64, #CSC64> to i64 {
      graphblas.yield agg_identity %c_min_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      %13 = arith.cmpi sgt, %arg0, %arg1 : i64
      %14 = arith.select %13, %arg0, %arg1 : i64
      graphblas.yield agg %14 : i64
    }
    // CHECK: answer_2 3
    graphblas.print %answer_2 { strings = ["answer_2 "] } : i64

    %answer_3 = graphblas.reduce_to_scalar_generic %vec : tensor<?xi64, #CV64> to i64 {
        graphblas.yield agg_identity %c0_i64 : i64
    },  {
      ^bb0(%a : i64, %b : i64):
        %result = arith.addi %a, %b : i64
        graphblas.yield agg %result : i64
    }
    // CHECK: answer_3 30
    graphblas.print %answer_3 { strings = ["answer_3 "] } : i64
    
    return
}

// COM: TODO write tests for all tensor element types
