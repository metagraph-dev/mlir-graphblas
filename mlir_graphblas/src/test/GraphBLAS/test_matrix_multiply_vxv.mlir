// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s
// RUN: graphblas-opt %s | graphblas-linalg-exec main | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @main() -> () {
    %a_dense = arith.constant dense<[4, 0, 2, 3]> : tensor<4xi64>
    %a = sparse_tensor.convert %a_dense : tensor<4xi64> to tensor<?xi64, #CV64>
    
    %b_dense = arith.constant dense<[0, 7, 0, 8]> : tensor<4xi64>
    %b = sparse_tensor.convert %b_dense : tensor<4xi64> to tensor<?xi64, #CV64>
    
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to i64
    // CHECK: answer 24
    graphblas.print %answer { strings = ["answer "] } : i64

    return
}
