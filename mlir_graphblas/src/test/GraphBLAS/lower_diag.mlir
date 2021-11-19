// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

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

// CHECK:           func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// COM: make a functional test

   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// COM: make a functional test

   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #CV64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #CV64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }
   
// COM: make a functional test
    func @mat_to_vec_fixed_csr(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #CV64>
        return %vec : tensor<7xi64, #CV64>
    }

// COM: make a functional test
    func @mat_to_vec_fixed_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #CV64>
        return %vec : tensor<7xi64, #CV64>
    }

}

module {

// COM: make a functional test
   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// COM: make a functional test
   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #CV64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #CV64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// COM: make a functional test
    func @mat_to_vec_arbitrary_csr(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
        return %vec : tensor<?xi64, #CV64>
    }

// COM: make a functional test
    func @mat_to_vec_arbitrary_csc(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #CV64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #CV64>
        return %vec : tensor<?xi64, #CV64>
    }

}
