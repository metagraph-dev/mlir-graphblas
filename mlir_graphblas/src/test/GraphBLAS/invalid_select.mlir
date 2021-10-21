// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16> {
        %answer = graphblas.select %sparse_tensor { selectors = ["triu"] } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16> // expected-error {{At least 1 result type does not match that of the input matrix.}}
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
        %answer = graphblas.select %sparse_tensor { selectors = ["BADSELECTOR"] } : tensor<2x3xi8, #CSR64> to tensor<2x3xi8, #CSR64> // expected-error {{"BADSELECTOR" is not a supported selector.}}
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xi16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %answer_0, %answer_1 = graphblas.select %sparse_tensor { selectors = ["triu", "tril"] } : tensor<2x3xbf16, #CSR64> to tensor<2x3xi16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{At least 1 result type does not match that of the input matrix.}}
        return %answer_0, %answer_1 : tensor<2x3xi16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
        %answer = graphblas.select %sparse_tensor, %thunk { selectors = ["gt"] } : tensor<2x3xbf16, #CSR64>, f64 to tensor<2x3xbf16, #CSR64> // expected-error {{Operand #1 is associated with the selector "gt", but has a different type than the input tensor's element type.}}
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %answer_0, %answer_1 = graphblas.select %sparse_tensor { selectors = ["gt", "triu"] } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{Some selectors ("gt") need thunks, but 0 thunks were given.}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %thunk = arith.constant 0.0 : bf16
        %answer_0, %answer_1 = graphblas.select %sparse_tensor, %thunk { selectors = ["gt", "triu"] } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{custom op 'graphblas.select' 1 operands present, but expected 0}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %thunk = arith.constant 0.0 : bf16
        %answer_0, %answer_1 = graphblas.select %sparse_tensor, %thunk { selectors = ["gt", "triu"] } : tensor<2x3xbf16, #CSR64>, bf16, bf16 to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{custom op 'graphblas.select' 1 operands present, but expected 2}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %thunk = arith.constant 0.0 : bf16
        %answer_0, %answer_1 = graphblas.select %sparse_tensor, %thunk { selectors = ["triu"] } : tensor<2x3xbf16, #CSR64>, bf16 to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{No selectors need thunks, but 1 thunks were given.}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %thunk = arith.constant 0.0 : bf16
        %answer_0, %answer_1 = graphblas.select %sparse_tensor, %thunk { selectors = ["tril", "triu"] } : tensor<2x3xbf16, #CSR64>, bf16 to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{No selectors need thunks, but 1 thunks were given.}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) {
        %thunk = arith.constant 0.0 : bf16
        %answer_0, %answer_1 = graphblas.select %sparse_tensor, %thunk { selectors = ["gt", "triu"] } : tensor<2x3xbf16, #CSR64> to tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64> // expected-error {{custom op 'graphblas.select' 1 operands present, but expected 0}}
        return %answer_0, %answer_1 : tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>
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
        %answer_0 = graphblas.select %sparse_tensor { selectors = ["triu"] } : tensor<2xf64, #CV64> to tensor<2xf64, #CV64> // expected-error {{Selector 'triu' not allowed for vectors}}
        return %answer_0 : tensor<2xf64, #CV64>
    }
}