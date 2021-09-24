// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64>, %vector: tensor<3xi64, #CV64>) -> tensor<2xi64, #CV64> {
       %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xi64>, tensor<3xi64, #CV64>) to tensor<2xi64, #CV64> // expected-error {{1st operand must be a sparse tensor.}}
       return %answer : tensor<2xi64, #CV64>
   }

}

// -----

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

   func @vector_matrix_multiply_plus_times(%vector: tensor<99xi64, #CV64>, %matrix: tensor<2x3xi64, #CSC64>) -> tensor<2xi64, #CV64> {
       %answer = graphblas.matrix_multiply %vector, %matrix { semiring = "plus_times" } : (tensor<99xi64, #CV64>, tensor<2x3xi64, #CSC64>) to tensor<2xi64, #CV64> // expected-error {{Operand shapes are incompatible.}}
       return %answer : tensor<2xi64, #CV64>
   }

}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64, #CSR64>, %vector: tensor<3xi64, #CV64>) -> tensor<99xi64, #CV64> {
       %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3xi64, #CV64>) to tensor<99xi64, #CV64> // expected-error {{Operand shapes incompatible with output shape.}}
       return %answer : tensor<99xi64, #CV64>
   }

}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xf64, #CSR64>, %vector: tensor<3xi64, #CV64>) -> tensor<2xf64, #CV64> {
       %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xf64, #CSR64>, tensor<3xi64, #CV64>) to tensor<2xf64, #CV64> // expected-error {{Operand element types must be identical.}}
       return %answer : tensor<2xf64, #CV64>
   }

}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64, #CSR64>, %vector: tensor<3xi64, #CV64>) -> tensor<2xf64, #CV64> {
       %answer = graphblas.matrix_multiply %matrix, %vector { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3xi64, #CV64>) to tensor<2xf64, #CV64> // expected-error {{Result element type differs from the input element types.}}
       return %answer : tensor<2xf64, #CV64>
   }

}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

   func @matrix_vector_multiply_plus_times(%matrix: tensor<2x3xi64, #CSR64>, %vector: tensor<3xi64, #CV64>, %mask: tensor<99xi64, #CV64>) -> tensor<2xi64, #CV64> {
       %answer = graphblas.matrix_multiply %matrix, %vector, %mask { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3xi64, #CV64>, tensor<99xi64, #CV64>) to tensor<2xi64, #CV64> // expected-error {{Mask shape must match shape of matrix multiply result.}}
       return %answer : tensor<2xi64, #CV64>
   }

}
