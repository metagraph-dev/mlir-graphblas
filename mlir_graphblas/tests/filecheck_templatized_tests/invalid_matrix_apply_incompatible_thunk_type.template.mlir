{
  "parameters": {"element_type": "STANDARD_ELEMENT_TYPES", "thunk_type": "STANDARD_ELEMENT_TYPES", "apply_operator": "MATRIX_APPLY_OPERATORS"},
  "prefilter": "different_thunk_and_element_type",
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3x{{element_type}}, #CSR64>, %thunk: {{thunk_type}}) -> tensor<2x3x{{thunk_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<2x3x{{element_type}}, #CSR64>, {{thunk_type}}) to tensor<2x3x{{thunk_type}}, #CSR64> // expected-error {% raw %}{{Element type of input tensor does not match type of thunk.}}{% endraw %}
        return %answer : tensor<2x3x{{thunk_type}}, #CSR64>
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<2x3x{{thunk_type}}, #CSR64>, %thunk: {{thunk_type}}) -> tensor<2x3x{{element_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<2x3x{{thunk_type}}, #CSR64>, {{thunk_type}}) to tensor<2x3x{{element_type}}, #CSR64> // expected-error {% raw %}{{Element type of result tensor does not match type of thunk.}}{% endraw %}
        return %answer : tensor<2x3x{{element_type}}, #CSR64>
    }
}
