{
  "parameters": {"element_type": "STANDARD_ELEMENT_TYPES", "apply_operator": "MATRIX_APPLY_OPERATORS", "sparsity0": "SPARSITY_TYPES", "sparsity1": "SPARSITY_TYPES", "sparsity2": "SPARSITY_TYPES", "dim0": ["?", "2"], "dim1": ["?", "3"], "dim2": ["?", "4"]},
  "run": "{{graphblas_opt}} {{input_file}} -split-input-file -verify-diagnostics"
}
### START TEST ###

#CSRBAD = #sparse_tensor.encoding<{
  dimLevelType = [ "{{sparsity0}}", "{{sparsity1}}", "{{sparsity2}}" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_apply_wrapper(%sparse_tensor: tensor<{{dim0}}x{{dim1}}x{{dim2}}x{{element_type}}, #CSRBAD>, %thunk: {{element_type}}) -> tensor<2x3x{{element_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<{{dim0}}x{{dim1}}x{{dim2}}x{{element_type}}, #CSRBAD>, {{element_type}}) to tensor<2x3x{{element_type}}, #CSR64> // expected-error {% raw %}{{Operand #0 must have rank 2.}}{% endraw %}
        return %answer : tensor<2x3x{{element_type}}, #CSR64>
    }
}

// -----

#CSRBAD = #sparse_tensor.encoding<{
  dimLevelType = [ "{{sparsity0}}", "{{sparsity1}}", "{{sparsity2}}" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3x{{element_type}}, #CSR64>, %thunk: {{element_type}}) -> tensor<2x3x{{element_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<2x3x{{element_type}}, #CSR64>, {{element_type}}) to tensor<{{dim0}}x{{dim1}}x{{dim2}}x{{element_type}}, #CSRBAD> // expected-error {% raw %}{{Return value must have rank 2.}}{% endraw %}
        return %answer : tensor<{{dim0}}x{{dim1}}x{{dim2}}x{{element_type}}, #CSRBAD>
    }
}
