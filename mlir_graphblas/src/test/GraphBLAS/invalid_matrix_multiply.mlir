// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_pair" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand #1 must be a sparse tensor.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64> // expected-error {{Return value must be a sparse tensor.}}
        return %answer : tensor<2x2xi64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "BAD_SEMIRING" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{"BAD_SEMIRING" is not a supported semiring.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x9xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x9xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand shapes are incompatible.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<9x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<9x2xi64, #CSR64> // expected-error {{Operand shapes incompatible with output shape.}}
        return %answer : tensor<9x2xi64, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSR64>) -> tensor<2x9xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x9xi64, #CSR64> // expected-error {{Operand shapes incompatible with output shape.}}
        return %answer : tensor<2x9xi64, #CSR64>
    }
}
