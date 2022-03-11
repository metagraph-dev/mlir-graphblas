// RUN: graphblas-opt %s | graphblas-exec main | FileCheck %s
// RUN: graphblas-opt %s | graphblas-linalg-exec main | FileCheck %s

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
    
    %vec_dense = arith.constant dense<[6, 8]> : tensor<2xi64>
    %vec = sparse_tensor.convert %vec_dense : tensor<2xi64> to tensor<?xi64, #CV64>
    
    %mask_dense = arith.constant dense<[9, 0, 0, 2]> : tensor<4xi64>
    %mask = sparse_tensor.convert %mask_dense : tensor<4xi64> to tensor<?xi64, #CV64>

    %answer_1 = graphblas.matrix_multiply %vec, %mat_csr { semiring = "plus_times" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>) to tensor<?xi64, #CV64>
    // CHECK: answer_1 [_, 6, 12, 24]
    graphblas.print %answer_1 { strings = ["answer_1 "] } : tensor<?xi64, #CV64>

    %answer_2 = graphblas.matrix_multiply %vec, %mat_csr { semiring = "min_plus" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_2 [_, 7, 8, 11]
    graphblas.print %answer_2 { strings = ["answer_2 "] } : tensor<?xi64, #CV64>

    %answer_4 = graphblas.matrix_multiply %vec, %mat_csc { semiring = "any_pair" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSC64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_4 [_, 1, 1, 1]
    graphblas.print %answer_4 { strings = ["answer_4 "] } : tensor<?xi64, #CV64>

    %answer_5 = graphblas.matrix_multiply %vec, %mat_csr, %mask { semiring = "plus_times" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_5 [_, _, _, 24]
    graphblas.print %answer_5 { strings = ["answer_5 "] } : tensor<?xi64, #CV64>

    %answer_6 = graphblas.matrix_multiply %vec, %mat_csr, %mask { semiring = "min_plus" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_6 [_, _, _, 11]
    graphblas.print %answer_6 { strings = ["answer_6 "] } : tensor<?xi64, #CV64>

    %answer_8 = graphblas.matrix_multiply %vec, %mat_csc, %mask { semiring = "any_pair" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_8 [_, _, _, 1]
    graphblas.print %answer_8 { strings = ["answer_8 "] } : tensor<?xi64, #CV64>

    %answer_9 = graphblas.matrix_multiply %vec, %mat_csr, %mask { semiring = "plus_times" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_9 [_, _, _, 24]
    graphblas.print %answer_9 { strings = ["answer_9 "] } : tensor<?xi64, #CV64>

    %answer_10 = graphblas.matrix_multiply %vec, %mat_csr, %mask { semiring = "min_plus" } : (tensor<?xi64, #CV64>, tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_10 [_, _, _, 11]
    graphblas.print %answer_10 { strings = ["answer_10 "] } : tensor<?xi64, #CV64>

    return
}

