// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>
 

// CHECK-LABEL:   func @transpose_swap(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 0 : index
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_13:.*]] = call @empty_like(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_dim(%[[VAL_13]], %[[VAL_1]], %[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_dim(%[[VAL_13]], %[[VAL_2]], %[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_pointers(%[[VAL_13]], %[[VAL_2]], %[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_index(%[[VAL_13]], %[[VAL_2]], %[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_values(%[[VAL_13]], %[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_13]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_13]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_13]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_17:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             memref.store %[[VAL_3]], %[[VAL_14]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_18:.*]] = %[[VAL_1]] to %[[VAL_12]] step %[[VAL_2]] {
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = index_cast %[[VAL_19]] : i64 to index
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = addi %[[VAL_21]], %[[VAL_4]] : i64
// CHECK:             memref.store %[[VAL_22]], %[[VAL_14]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_3]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_23:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_25]], %[[VAL_14]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_26:.*]] = addi %[[VAL_25]], %[[VAL_24]] : i64
// CHECK:             memref.store %[[VAL_26]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_27:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_28:.*]] = index_cast %[[VAL_27]] : index to i64
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_27]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = index_cast %[[VAL_29]] : i64 to index
// CHECK:             %[[VAL_31:.*]] = addi %[[VAL_27]], %[[VAL_2]] : index
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = index_cast %[[VAL_32]] : i64 to index
// CHECK:             scf.for %[[VAL_34:.*]] = %[[VAL_30]] to %[[VAL_33]] step %[[VAL_2]] {
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:               %[[VAL_36:.*]] = index_cast %[[VAL_35]] : i64 to index
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:               %[[VAL_38:.*]] = index_cast %[[VAL_37]] : i64 to index
// CHECK:               memref.store %[[VAL_28]], %[[VAL_15]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_34]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_39]], %[[VAL_16]]{{\[}}%[[VAL_38]]] : memref<?xf64>
// CHECK:               %[[VAL_40:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:               %[[VAL_41:.*]] = addi %[[VAL_40]], %[[VAL_4]] : i64
// CHECK:               memref.store %[[VAL_41]], %[[VAL_14]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_42:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_43:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_44:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             %[[VAL_45:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_45]], %[[VAL_14]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_44]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_42]], %[[VAL_14]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           return %[[VAL_13]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @transpose_swap(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.transpose %sparse_tensor { swap_sizes = true } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
