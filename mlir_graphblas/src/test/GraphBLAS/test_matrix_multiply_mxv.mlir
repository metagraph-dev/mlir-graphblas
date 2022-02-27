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

    %answer_1 = graphblas.matrix_multiply %a_csr, %b { semiring = "plus_times" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK: answer_1 [7, 24]
    graphblas.print %answer_1 { strings = ["answer_1 "] } : tensor<?xi64, #CV64>

    %answer_2 = graphblas.matrix_multiply %a_csr, %b { semiring = "min_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_2 [8, 11]
    graphblas.print %answer_2 { strings = ["answer_2 "] } : tensor<?xi64, #CV64>

    %answer_3 = graphblas.matrix_multiply %a_csc, %b { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_3 [1, 3]
    graphblas.print %answer_3 { strings = ["answer_3 "] } : tensor<?xi64, #CV64>

    %answer_4 = graphblas.matrix_multiply %a_csc, %b { semiring = "any_pair" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_4 [1, 1]
    graphblas.print %answer_4 { strings = ["answer_4 "] } : tensor<?xi64, #CV64>
    
    %answer_5 = graphblas.matrix_multiply %a_csr, %b, %mask { semiring = "plus_times" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_5 [7, _]
    graphblas.print %answer_5 { strings = ["answer_5 "] } : tensor<?xi64, #CV64>

    %answer_6 = graphblas.matrix_multiply %a_csr, %b, %mask { semiring = "min_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_6 [8, _]
    graphblas.print %answer_6 { strings = ["answer_6 "] } : tensor<?xi64, #CV64>

    %answer_7 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_7 [1, _]
    graphblas.print %answer_7 { strings = ["answer_7 "] } : tensor<?xi64, #CV64>

    %answer_8 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_pair" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_8 [1, _]
    graphblas.print %answer_8 { strings = ["answer_8 "] } : tensor<?xi64, #CV64>

    %answer_9 = graphblas.matrix_multiply %a_csr, %b, %mask { semiring = "plus_times" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_9 [7, _]
    graphblas.print %answer_9 { strings = ["answer_9 "] } : tensor<?xi64, #CV64>

    %answer_10 = graphblas.matrix_multiply %a_csr, %b, %mask { semiring = "min_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_10 [8, _]
    graphblas.print %answer_10 { strings = ["answer_10 "] } : tensor<?xi64, #CV64>

    %answer_11 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_overlapi" } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_11 [1, _]
    graphblas.print %answer_11 { strings = ["answer_11 "] } : tensor<?xi64, #CV64>

    %answer_12 = graphblas.matrix_multiply %a_csc, %b, %mask { semiring = "any_pair", mask_complement = true } : (tensor<?x?xi64, #CSC64>, tensor<?xi64, #CV64>, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
    // CHECK-NEXT: answer_12 [_, 1]
    graphblas.print %answer_12 { strings = ["answer_12 "] } : tensor<?xi64, #CV64>

    return
}

