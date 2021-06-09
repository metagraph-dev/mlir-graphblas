// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @select_gt0(
// CHECK-SAME:                     %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 0 : index
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = call @dup_tensor(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_11]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_11]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_11]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_12]]{{\[}}%[[VAL_1]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_1]] to %[[VAL_6]] step %[[VAL_2]] {
// CHECK:             %[[VAL_16:.*]] = addi %[[VAL_15]], %[[VAL_2]] : index
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_17]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = index_cast %[[VAL_18]] : i64 to index
// CHECK:             %[[VAL_21:.*]] = index_cast %[[VAL_19]] : i64 to index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_2]] {
// CHECK:               %[[VAL_27:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:               %[[VAL_24:.*]] = cmpf ogt, %[[VAL_23]], %[[VAL_5]] : f64
// CHECK:               scf.if %[[VAL_24]] {
// CHECK:                 %[[VAL_25:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:                 %[[VAL_26:.*]] = index_cast %[[VAL_25]] : i64 to index
// CHECK:                 memref.store %[[VAL_27]], %[[VAL_13]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_23]], %[[VAL_14]]{{\[}}%[[VAL_26]]] : memref<?xf64>
// CHECK:                 %[[VAL_28:.*]] = addi %[[VAL_25]], %[[VAL_4]] : i64
// CHECK:                 memref.store %[[VAL_28]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:           %[[VAL_30:.*]] = index_cast %[[VAL_29]] : i64 to index
// CHECK:           call @resize_index(%[[VAL_11]], %[[VAL_2]], %[[VAL_30]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_values(%[[VAL_11]], %[[VAL_30]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           return %[[VAL_11]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @select_gt0(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_select %sparse_tensor { selectors = ["gt0"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
