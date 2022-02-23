// RUN: graphblas-opt %s | graphblas-linalg-exec main | FileCheck %s

#accesses = [
  affine_map<(m, n) -> (m, n)>,
  affine_map<(m, n) -> (m, n)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
  func @main() -> () {

      %dense_mat = arith.constant dense<[
          [0.0, -1.0,  2.0,  0.0],
          [0.0,  0.0, -3.0, 4.0]
        ]> : tensor<2x4xf64>
      %mat = sparse_tensor.convert %dense_mat : tensor<2x4xf64> to tensor<?x?xf64, #CSR64>


      %cf2 = arith.constant 2.0 : f64
      %apply_answer = graphblas.apply_generic %mat : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64> {
        ^bb0(%val: f64):
          %negative_val = arith.negf %val: f64
          %result = arith.mulf %negative_val, %cf2 : f64
          graphblas.yield transform_out %result : f64
      }

      // CHECK: %apply_answer [
      // CHECK-NEXT:   [_, 2, -4, _],
      // CHECK-NEXT:   [_, _, 6, -8]
      // CHECK-NEXT: ]
      graphblas.print %apply_answer { strings = ["%apply_answer "] } : tensor<?x?xf64, #CSR64>
      
      return
  }
}
