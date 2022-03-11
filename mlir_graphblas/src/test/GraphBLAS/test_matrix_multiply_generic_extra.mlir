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

// COM: linalg-lowering does not yet support:
// COM: - including indices inside intersect block
// COM: - mask_complement=true

func @main() -> () {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c_big_num_i64 = arith.constant 9223372036854775807 : i64

    %a_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3]
      ]> : tensor<2x4xi64>
    %a_csr = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %a_csc = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>

    %b_dense = arith.constant dense<[
        [0, 7],
        [4, 0],
        [5, 0],
        [6, 8]
      ]> : tensor<4x2xi64>
    %b_csr = sparse_tensor.convert %b_dense : tensor<4x2xi64> to tensor<?x?xi64, #CSR64>
    %b_csc = sparse_tensor.convert %b_dense : tensor<4x2xi64> to tensor<?x?xi64, #CSC64>

    %mask_dense = arith.constant dense<[
        [9, 0],
        [0, 8]
      ]> : tensor<2x2xi64>
    %mask_csr = sparse_tensor.convert %mask_dense : tensor<2x2xi64> to tensor<?x?xi64, #CSR64>
    %mask_csc = sparse_tensor.convert %mask_dense : tensor<2x2xi64> to tensor<?x?xi64, #CSC64>

    %answer_3 = graphblas.matrix_multiply_generic %a_csc, %b_csr { mask_complement = false } : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>) to tensor<?x?xi64, #CSR64>  {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %28 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %28 : i64
    }
    // CHECK: answer_3 [
    // CHECK-NEXT:   [2, _],
    // CHECK-NEXT:   [3, 3]
    // CHECK-NEXT: ]
    graphblas.print %answer_3 { strings = ["answer_3 "] } : tensor<?x?xi64, #CSR64>

    %answer_7 = graphblas.matrix_multiply_generic %a_csc, %b_csr, %mask_csr { mask_complement = false } : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSR64>) to tensor<?x?xi64, #CSR64>  {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %34 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %34 : i64
    }
    // CHECK-NEXT: answer_7 [
    // CHECK-NEXT:   [2, _],
    // CHECK-NEXT:   [_, 3]
    // CHECK-NEXT: ]
    graphblas.print %answer_7 { strings = ["answer_7 "] } : tensor<?x?xi64, #CSR64>

    %answer_11 = graphblas.matrix_multiply_generic %a_csc, %b_csr, %mask_csc { mask_complement = false } : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>  {
      graphblas.yield add_identity %c0_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: index, %arg3: index, %arg4: index):
      %34 = arith.index_cast %arg4 : index to i64
      graphblas.yield mult %34 : i64
    }
    // CHECK-NEXT: answer_11 [
    // CHECK-NEXT:   [2, _],
    // CHECK-NEXT:   [_, 3]
    // CHECK-NEXT: ]
    graphblas.print %answer_11 { strings = ["answer_11 "] } : tensor<?x?xi64, #CSR64>

    %answer_12 = graphblas.matrix_multiply_generic %a_csc, %b_csc, %mask_csc { mask_complement = true } : (tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>  {
      graphblas.yield add_identity %c1_i64 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield add %arg1 : i64
    }, {
    ^bb0(%arg0: i64, %arg1: i64):
      graphblas.yield mult %c1_i64 : i64
    }
    // CHECK: answer_12 [
    // CHECK-NEXT:   [_, _],
    // CHECK-NEXT:   [1, _]
    // CHECK-NEXT: ]
    graphblas.print %answer_12 { strings = ["answer_12 "] } : tensor<?x?xi64, #CSR64>

    return
}
