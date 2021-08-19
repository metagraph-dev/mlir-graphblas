// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    // CHECK-LABEL:   func @matrix_reduce_to_scalar_f64(
    // CHECK-SAME:                                      %[[VAL_0:.*]]: tensor<?x?xf64, [[CSR:.*->.*]]>) -> f64 {
    // CHECK-DAG:       %[[VAL_1:.*]] = constant 0.000000e+00 : f64
    // CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : index
    // CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
    // CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, [[CSR]]>
    // CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, [[CSR]]> to memref<?xi64>
    // CHECK:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, [[CSR]]> to memref<?xf64>
    // CHECK:           %[[VAL_7:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xi64>
    // CHECK:           %[[VAL_8:.*]] = index_cast %[[VAL_7]] : i64 to index
    // CHECK:           %[[VAL_9:.*]] = scf.parallel (%[[VAL_10:.*]]) = (%[[VAL_2]]) to (%[[VAL_8]]) step (%[[VAL_3]]) init (%[[VAL_1]]) -> f64 {
    // CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_10]]] : memref<?xf64>
    // CHECK:             scf.reduce(%[[VAL_11]])  : f64 {
    // CHECK:             ^bb0(%[[VAL_12:.*]]: f64, %[[VAL_13:.*]]: f64):
    // CHECK:               %[[VAL_14:.*]] = addf %[[VAL_12]], %[[VAL_13]] : f64
    // CHECK:               scf.reduce.return %[[VAL_14]] : f64
    // CHECK:             }
    // CHECK:             scf.yield
    // CHECK:           }
    // CHECK:           return %[[VAL_15:.*]] : f64
    // CHECK:         }
    func @matrix_reduce_to_scalar_f64(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> f64 {
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<?x?xf64, #CSR64> to f64
        return %answer : f64
    }
}

// COM: TODO write tests for all tensor element types
