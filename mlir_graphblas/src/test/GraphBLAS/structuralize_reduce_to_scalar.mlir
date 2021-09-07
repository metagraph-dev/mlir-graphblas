// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

    // CHECK-LABEL:   func @matrix_reduce_to_scalar_f64(
    // CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
    // CHECK:           %[[VAL_1:.*]] = constant 0.000000e+00 : f64
    // CHECK:           %[[VAL_2:.*]] = graphblas.reduce_to_scalar_generic %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to f64  {
    // CHECK:             graphblas.yield agg_identity %[[VAL_1]] : f64
    // CHECK:           },  {
    // CHECK:           ^bb0(%[[VAL_3:.*]]: f64, %[[VAL_4:.*]]: f64):
    // CHECK:             %[[VAL_5:.*]] = addf %[[VAL_3]], %[[VAL_4]] : f64
    // CHECK:             graphblas.yield agg %[[VAL_5]] : f64
    // CHECK:           }
    // CHECK:           return %[[VAL_2]] : f64
    // CHECK:         }
    func @matrix_reduce_to_scalar_f64(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #CSR64> to f64
        return %answer : f64
    }
}
