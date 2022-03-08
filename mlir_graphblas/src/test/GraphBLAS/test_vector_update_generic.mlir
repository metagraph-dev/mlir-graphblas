
// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @main() {

    %input_dense = arith.constant dense<[9, 0, 0, 2]> : tensor<4xi64>
    %input= sparse_tensor.convert %input_dense : tensor<4xi64> to tensor<?xi64, #CV64>

    %output_dense = arith.constant dense<[0, 0, 5, 6]> : tensor<4xi64>
    %output= sparse_tensor.convert %output_dense : tensor<4xi64> to tensor<?xi64, #CV64>

    graphblas.print %input { strings = ["input "] } : tensor<?xi64, #CV64>
    // CHECK: input [9, _, _, 2]
    graphblas.print %output { strings = ["output "] } : tensor<?xi64, #CV64>
    // CHECK: output [_, _, 5, 6]

    graphblas.update_generic %input -> %output {mask_complement = false, replace = false} : tensor<?xi64, #CV64> -> tensor<?xi64, #CV64> {
    ^bb0(%arg0: i64, %arg1: i64):
      %2 = arith.addi %arg0, %arg1 : i64
      graphblas.yield accumulate %2 : i64
    }

    graphblas.print %input { strings = ["input "] } : tensor<?xi64, #CV64>
    // CHECK: input [9, _, _, 2]
    graphblas.print %output { strings = ["output "] } : tensor<?xi64, #CV64>
    // CHECK: output [9, _, 5, 8]

    return
}
