// RUN: graphblas-opt %s -split-input-file -verify-diagnostics 

func @matrix_multiply_wrapper(%argA: tensor<1x2x3xi64>, %argB: tensor<3x2xi64>) -> tensor<2x2xi64> {
  %answer = graphblas.matrix_multiply_generic %argA, %argB {mask_complement=false} : (tensor<1x2x3xi64>, tensor<3x2xi64>) to tensor<2x2xi64> // expected-error {{op operand #0 must be 1D/2D tensor of any type values, but got 'tensor<1x2x3xi64>'}}
  return %answer : tensor<2x2xi64>
}

// -----

func @size_wrapper(%argA: tensor<1x1xi64>) -> index {
  %answer = graphblas.size %argA : tensor<1x1xi64> // expected-error {{op operand #0 must be 1D tensor of any type values, but got 'tensor<1x1xi64>'}}
  return %answer : index
}

// -----

func @num_rows_wrapper(%argA: tensor<1xi64>) -> index {
  %answer = graphblas.num_rows %argA : tensor<1xi64> // expected-error {{op operand #0 must be 2D tensor of any type values, but got 'tensor<1xi64>'}}
  return %answer : index
}

// -----

func @num_cols_wrapper(%argA: tensor<1xi64>) -> index {
  %answer = graphblas.num_cols %argA : tensor<1xi64> // expected-error {{op operand #0 must be 2D tensor of any type values, but got 'tensor<1xi64>'}}
  return %answer : index
}

// -----

func @dup_warpper(%argA: tensor<1x1xi64>) -> tensor<1x1xi64> {
  %answer = graphblas.dup %argA : tensor<1x1xi64> // expected-error {{operand #0 must have sparse tensor attribute}}
  return %answer : tensor<1x1xi64>
}

// -----

#E = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @dup_warpper(%argA: tensor<1x1xi64, #E>) -> tensor<1x1xi64, #E> {
  %answer = graphblas.dup %argA : tensor<1x1xi64, #E> // expected-error {{operand #0 must be in CSR or in CSC compression}}
  return %answer : tensor<1x1xi64, #E>
}
