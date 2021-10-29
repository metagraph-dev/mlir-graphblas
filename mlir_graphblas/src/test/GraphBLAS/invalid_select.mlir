// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16> {
        %answer = graphblas.select %sparse_tensor { selector = "triu" } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16> // expected-error {{result type must match input type.}}
        return %answer : tensor<2x3xbf16>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>) -> tensor<2x3xi8, #CSR64> {
        %answer = graphblas.select %sparse_tensor { selector = "BADSELECTOR" } : tensor<2x3xi8, #CSR64> to tensor<2x3xi8, #CSR64> // expected-error {{"BADSELECTOR" is not a supported selector.}}
        return %answer : tensor<2x3xi8, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16, #CSR64> {
        %thunk = arith.constant 0.0 : f64
        %answer = graphblas.select %sparse_tensor, %thunk { selector = "gt" } : tensor<2x3xbf16, #CSR64>, f64 to tensor<2x3xbf16, #CSR64> // expected-error {{Thunk type must match operand type.}}
        return %answer : tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.select %sparse_tensor { selector = "gt" } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16, #CSR64> // expected-error {{Selector 'gt' requires a thunk.}}
        return %answer : tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16, #CSR64> {
        %thunk = arith.constant 0.0 : bf16
        %answer = graphblas.select %sparse_tensor, %thunk { selector = "triu" } : tensor<2x3xbf16, #CSR64>, bf16 to tensor<2x3xbf16, #CSR64> // expected-error {{Selector 'triu' cannot take a thunk.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @vector_select_wrapper(%sparse_tensor: tensor<2xf64, #CV64>) -> tensor<2xf64, #CV64> {
        %answer_0 = graphblas.select %sparse_tensor { selector = "triu" } : tensor<2xf64, #CV64> to tensor<2xf64, #CV64> // expected-error {{Selector 'triu' not allowed for vectors}}
        return %answer_0 : tensor<2xf64, #CV64>
    }
}