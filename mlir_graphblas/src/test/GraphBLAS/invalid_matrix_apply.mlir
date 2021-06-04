// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xbf16>, %thunk: bf16) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xbf16>, bf16) to tensor<2x3xbf16, #CSR64> // expected-error {{Operand #0 must be a sparse tensor.}}
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>, %thunk: bf16) -> tensor<2x3xbf16> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xbf16, #CSR64>, bf16) to tensor<2x3xbf16> // expected-error {{Return value must be a sparse tensor.}}
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: f16) -> tensor<2x3xf16, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, f16) to tensor<2x3xf16, #CSR64> // expected-error {{Element type of input tensor does not match type of thunk.}}
        return %answer : tensor<2x3xf16, #CSR64>
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<2x3xf16, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<2x3xf16, #CSR64> // expected-error {{Element type of result tensor does not match type of thunk.}}
        return %answer : tensor<2x3xf16, #CSR64>
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<99x99xi8, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<99x99xi8, #CSR64> // expected-error {{Input shape does not match output shape.}}
        return %answer : tensor<99x99xi8, #CSR64>
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<2x3xi8, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "BADOPERATOR" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<2x3xi8, #CSR64> // expected-error {{"BADOPERATOR" is not a supported semiring.}}
        return %answer : tensor<2x3xi8, #CSR64>
    }
}
