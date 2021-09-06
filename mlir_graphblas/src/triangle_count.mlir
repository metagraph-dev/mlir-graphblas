#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @triangle_count(%A: tensor<?x?xf64, #CSR64>) -> f64 {
        %U = graphblas.matrix_select %A { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
        %L = graphblas.matrix_select %A { selectors = ["tril"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
        %U_csc = graphblas.convert_layout %U : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
        %C = graphblas.matrix_multiply %L, %U_csc, %L { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64> 
        %reduce_result = graphblas.matrix_reduce_to_scalar %C { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
        return %reduce_result : f64
    }
}
