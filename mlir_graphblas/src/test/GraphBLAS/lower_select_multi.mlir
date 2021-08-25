// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>


// CHECK:               %[[VAL_35:.*]] = cmpf ogt, %[[VAL_34:.*]], %[[VAL_5:.*]] : f64
// CHECK:               %[[VAL_39:.*]] = cmpi ugt, %[[VAL_33:.*]], %[[VAL_22:.*]] : index
// CHECK:               %[[VAL_43:.*]] = cmpi ult, %[[VAL_33:.*]], %[[VAL_22:.*]] : index
// CHECK:           return %[[VAL_10:.*]], %[[VAL_14:.*]], %[[VAL_18:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
func @select_multi(%sparse_tensor: tensor<?x?xf64, #CSR64>, %thunk: f64) -> (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) {
    %answer1, %answer2, %answer3 = graphblas.matrix_select %sparse_tensor, %thunk { selectors = ["gt", "triu", "tril"] } : tensor<?x?xf64, #CSR64>, f64 to tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
    return %answer1, %answer2, %answer3 : tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
}
