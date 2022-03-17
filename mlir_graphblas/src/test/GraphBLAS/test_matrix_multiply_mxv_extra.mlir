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

// COM: linalg-lowering does not yet support:
// COM: - including indices inside intersect block
// COM: - mask_complement=true

func @main() -> () {
    %a_dense = arith.constant dense<[
        [0, 1, 2, 0],
        [0, 0, 0, 3]
      ]> : tensor<2x4xi64>
    %a_csr = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSR64>
    %a_csc = sparse_tensor.convert %a_dense : tensor<2x4xi64> to tensor<?x?xi64, #CSC64>

    %b_dense = arith.constant dense<[0, 7, 0, 8]> : tensor<4xi64>
    %b = sparse_tensor.convert %b_dense : tensor<4xi64> to tensor<?xi64, #CV64>

    %mask_dense = arith.constant dense<[9, 0]> : tensor<2xi64>
    %mask = sparse_tensor.convert %mask_dense : tensor<2xi64> to tensor<?xi64, #CV64>

    %answer_3 = graphblas.matrix_multiply %a_csc, %b { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK: answer_3 [1, 3]
    graphblas.print %answer_3 { strings = ["answer_3 "] } : tensor<?xi64, #CV64>

    %answer_7 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_7 [1, _]
    graphblas.print %answer_7 { strings = ["answer_7 "] } : tensor<?xi64, #CV64>

    %answer_11 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_11 [1, _]
    graphblas.print %answer_11 { strings = ["answer_11 "] } : tensor<?xi64, #CV64>

    %answer_12 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_pair", mask_complement = true } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_12 [_, 1]
    graphblas.print %answer_12 { strings = ["answer_12 "] } : tensor<?xi64, #CV64>

    return
}
