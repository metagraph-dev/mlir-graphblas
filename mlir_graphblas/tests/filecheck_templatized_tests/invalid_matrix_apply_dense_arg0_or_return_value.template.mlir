{
  "parameters": {"element_type": "STANDARD_ELEMENT_TYPES", "apply_operator": "MATRIX_APPLY_OPERATORS", "dim0": ["?", "2"], "dim1": ["?", "3"]},
  "run": "{{graphblas_opt}} {{input_file}} -split-input-file -verify-diagnostics"
}

### START TEST ###

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @matrix_apply_wrapper(%sparse_tensor: tensor<{{dim0}}x{{dim1}}x{{element_type}}>, %thunk: {{element_type}}) -> tensor<2x3x{{element_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<{{dim0}}x{{dim1}}x{{element_type}}>, {{element_type}}) to tensor<2x3x{{element_type}}, #CSR64> // expected-error {% raw %}{{Operand #0 must be a sparse tensor.}}{% endraw %}
        return %answer : tensor<2x3x{{element_type}}, #CSR64>
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3x{{element_type}}, #CSR64>, %thunk: {{element_type}}) -> tensor<{{dim0}}x{{dim1}}x{{element_type}}> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<2x3x{{element_type}}, #CSR64>, {{element_type}}) to tensor<{{dim0}}x{{dim1}}x{{element_type}}> // expected-error {% raw %}{{Return value must be a sparse tensor.}}{% endraw %}
        return %answer : tensor<{{dim0}}x{{dim1}}x{{element_type}}>
    }
}
