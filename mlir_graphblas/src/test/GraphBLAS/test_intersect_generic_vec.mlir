// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>


func @main() -> () {
    %dense_v1 = arith.constant dense<[0.0, 7.0, 4.0, 0.0, 5.0, 0.0, 6.0, 8.0]> : tensor<8xf64>
    %v1 = sparse_tensor.convert %dense_v1 : tensor<8xf64> to tensor<?xf64, #CV64>
    
    %dense_v2 = arith.constant dense<[0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0]> : tensor<8xf64>
    %v2 = sparse_tensor.convert %dense_v2 : tensor<8xf64> to tensor<?xf64, #CV64>
    
    %result = graphblas.intersect_generic %v1, %v2 : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64> {
    ^bb0(%arg0: f64, %arg1: f64):
      %3 = arith.mulf %arg0, %arg1 : f64
      graphblas.yield mult %3 : f64
    }
    // CHECK: %result [_, 7, _, _, 10, _, _, 24]
    graphblas.print %result { strings = ["%result "] } : tensor<?xf64, #CV64>
    
    return 
}
