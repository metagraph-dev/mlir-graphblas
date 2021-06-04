// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_select_wrapper(%sparse_tensor: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
        %answer = graphblas.matrix_select %sparse_tensor { selector = "min" } : tensor<2x3xbf16> // expected-error {{Return value must be a sparse tensor.}}
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
        %answer = graphblas.matrix_select %sparse_tensor { selector = "BADSELECTOR" } : tensor<2x3xi8, #CSR64> // expected-error {{"BADSELECTOR" is not a supported selector.}}
        return %answer : tensor<2x3xi8, #CSR64>
    }
}
