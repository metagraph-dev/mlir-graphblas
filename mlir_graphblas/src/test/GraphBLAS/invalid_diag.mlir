// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @diag_wrapper(%sparse_tensor: tensor<7x7xbf16>) -> tensor<7xbf16, #CV64> {
        %answer = graphblas.diag %sparse_tensor : tensor<7x7xbf16> to tensor<7xbf16, #CV64> // expected-error {{operand must be a sparse tensor.}}
        return %answer : tensor<7xbf16, #CV64>
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
    func @diag_wrapper(%sparse_tensor: tensor<7xbf16>) -> tensor<7x7xbf16, #CSR64> {
        %answer = graphblas.diag %sparse_tensor : tensor<7xbf16> to tensor<7x7xbf16, #CSR64> // expected-error {{operand must be a sparse tensor.}}
        return %answer : tensor<7x7xbf16, #CSR64>
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @diag_wrapper(%sparse_tensor: tensor<7xbf16, #CV64>) -> tensor<7x7xbf16> {
        %answer = graphblas.diag %sparse_tensor : tensor<7xbf16, #CV64> to tensor<7x7xbf16> // expected-error {{result must be a sparse tensor.}}
        return %answer : tensor<7x7xbf16>
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
    func @diag_wrapper(%sparse_tensor: tensor<7x7xbf16, #CSR64>) -> tensor<7xbf16> {
        %answer = graphblas.diag %sparse_tensor : tensor<7x7xbf16, #CSR64> to tensor<7xbf16> // expected-error {{result must be a sparse tensor.}}
        return %answer : tensor<7xbf16>
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
    func @diag_wrapper(%sparse_tensor: tensor<7x8xi8, #CSR64>) -> tensor<7xi8, #CV64> {
        %answer = graphblas.diag %sparse_tensor : tensor<7x8xi8, #CSR64> to tensor<7xi8, #CV64> // expected-error {{Input shape must be square.}}
        return %answer : tensor<7xi8, #CV64>
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
    func @diag_wrapper(%sparse_tensor: tensor<7x7xi8, #CSR64>) -> tensor<99xi8, #CV64> {
        %answer = graphblas.diag %sparse_tensor : tensor<7x7xi8, #CSR64> to tensor<99xi8, #CV64> // expected-error {{Input shape is not compatible with output shape.}}
        return %answer : tensor<99xi8, #CV64>
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
    func @diag_wrapper(%sparse_tensor: tensor<6xi8, #CV64>) -> tensor<6x7xi8, #CSR64> {
        %answer = graphblas.diag %sparse_tensor : tensor<6xi8, #CV64> to tensor<6x7xi8, #CSR64> // expected-error {{Output shape must be square.}}
        return %answer : tensor<6x7xi8, #CSR64>
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
    func @diag_wrapper(%sparse_tensor: tensor<6xi8, #CV64>) -> tensor<99x99xi8, #CSR64> {
        %answer = graphblas.diag %sparse_tensor : tensor<6xi8, #CV64> to tensor<99x99xi8, #CSR64> // expected-error {{Input shape is not compatible with output shape.}}
        return %answer : tensor<99x99xi8, #CSR64>
    }
}
