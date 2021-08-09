// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<2x3xi8, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "BADOPERATOR" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<2x3xi8, #CSR64> // expected-error {{"BADOPERATOR" is not a supported operator.}}
        return %answer : tensor<2x3xi8, #CSR64>
    }
}
