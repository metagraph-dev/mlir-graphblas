// RUN: graphblas-opt %s | graphblas-exec print_arbitrary_content | FileCheck %s

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
  func @print_arbitrary_content() -> () {
      %c99 = arith.constant 99 : index
      %0 = arith.constant 1.3 : f32
      %1 = arith.constant 34 : i8

      // CHECK: first line : 1.3 1.3 1.3 1.3
      graphblas.print %0, %0, %0, %0 { strings = ["first line : "] } : f32, f32, f32, f32
      // CHECK: second line : 99 string_a  string_b  string_c
      graphblas.print %c99 { strings = ["second line : ", " string_a", " string_b", " string_c"] } : index
      // CHECK: third line : 1.3 |"| 34
      graphblas.print %0, %1 { strings = ["third line : ", " |\"| "] } : f32, i8

      %dense_vec = arith.constant dense<[0.0, 10.0, 20.0, 0.0]> : tensor<4xf64>
      %vec = sparse_tensor.convert %dense_vec : tensor<4xf64> to tensor<?xf64, #CV64>

      // CHECK: vec [0, 10, 20, 0]
      graphblas.print %dense_vec { strings = ["vec "] } : tensor<4xf64>

      // CHECK: vec [_, 10, 20, _]
      graphblas.print %vec { strings = ["vec "] } : tensor<?xf64, #CV64>

      %dense_mat = arith.constant dense<[
          [0.0, 1.0, 2.0, 0.0],
          [0.0, 0.0, 0.0, 3.0]
        ]> : tensor<2x4xf64>
      %mat = sparse_tensor.convert %dense_mat : tensor<2x4xf64> to tensor<?x?xf64, #CSR64>
      %mat_csc = graphblas.convert_layout %mat : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>

      // CHECK: mat [
      // CHECK:   [0, 1, 2, 0],
      // CHECK:   [0, 0, 0, 3],
      // CHECK: ]
      graphblas.print %dense_mat { strings = ["mat "] } : tensor<2x4xf64>

      // CHECK: mat [
      // CHECK:   [_, 1, 2, _],
      // CHECK:   [_, _, _, 3],
      // CHECK: ]
      graphblas.print %mat { strings = ["mat "] } : tensor<?x?xf64, #CSR64>

      // CHECK: mat_csc [
      // CHECK:   [_, 1, 2, _],
      // CHECK:   [_, _, _, 3],
      // CHECK: ]
      graphblas.print %mat_csc { strings = ["mat_csc "] } : tensor<?x?xf64, #CSC64>

      return
  }
}
