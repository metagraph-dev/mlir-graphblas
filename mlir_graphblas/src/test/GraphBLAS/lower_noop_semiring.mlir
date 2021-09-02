// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

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

// verify that directly yielding one of the block arguments lowers without crashing
// CHECK-LABEL:   builtin.func @noop_semiring(
func @noop_semiring(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply_generic %a, %b {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
        ^bb0:
            %identity = constant 0.0 : f64
            graphblas.yield add_identity %identity : f64
    },{
        ^bb0(%add_a: f64, %add_b: f64):
            graphblas.yield add %add_a : f64
    },{
        ^bb0(%mult_a: f64, %mult_b: f64):
            graphblas.yield mult %mult_b : f64
    }
    return %answer : tensor<?x?xf64, #CSR64>
}

