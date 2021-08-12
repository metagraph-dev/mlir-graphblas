// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xi16, #CSR64>) -> tensor<2x3xi16, #CSR64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<2x3xi16, #CSR64> to tensor<2x3xi16, #CSR64> // expected-error {{Input and output shapes are expected to be swapped.}}
        return %answer : tensor<2x3xi16, #CSR64>
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
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xi16, #CSR64>) -> tensor<99x2xi16, #CSR64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<2x3xi16, #CSR64> to tensor<99x2xi16, #CSR64> // expected-error {{Input and output shapes are expected to be swapped.}}
        return %answer : tensor<99x2xi16, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSR_BOGUS = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 32,
  indexBitWidth = 64
}>

module {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xi16, #CSR64>) -> tensor<3x2xi16, #CSR_BOGUS> {
        %answer = graphblas.transpose %sparse_tensor : tensor<2x3xi16, #CSR64> to tensor<3x2xi16, #CSR_BOGUS> // expected-error {{Sparse encoding pointer bit widths of input and result tensors do not match.}}
        return %answer : tensor<3x2xi16, #CSR_BOGUS>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSR_BOGUS = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 32
}>

module {
    func @transpose_wrapper(%sparse_tensor: tensor<2x3xi16, #CSR64>) -> tensor<3x2xi16, #CSR_BOGUS> {
        %answer = graphblas.transpose %sparse_tensor : tensor<2x3xi16, #CSR64> to tensor<3x2xi16, #CSR_BOGUS> // expected-error {{Sparse encoding index bit widths of input and result tensors do not match.}}
        return %answer : tensor<3x2xi16, #CSR_BOGUS>
    }
}
