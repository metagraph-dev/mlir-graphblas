{
  "parameters": {"element_type": "STANDARD_ELEMENT_TYPES", "apply_operator": "MATRIX_APPLY_OPERATORS"},
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
    func @matrix_apply_wrapper(%sparse_tensor: tensor<8x9x{{element_type}}, #CSR64>, %thunk: {{element_type}}) -> tensor<2x3x{{element_type}}, #CSR64> {
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "{{apply_operator}}" } : (tensor<8x9x{{element_type}}, #CSR64>, {{element_type}}) to tensor<2x3x{{element_type}}, #CSR64> // expected-error {% raw %}{{Input shape does not match output shape.}}{% endraw %}
        return %answer : tensor<2x3x{{element_type}}, #CSR64>
    }
}
