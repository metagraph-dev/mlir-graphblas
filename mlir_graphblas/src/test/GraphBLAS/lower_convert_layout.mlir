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

// CHECK-DAG:     func private @cast_csx_to_csc(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @cast_csx_to_csr(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @matrix_resize_values(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @matrix_resize_index(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @matrix_resize_pointers(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @matrix_resize_dim(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @matrix_empty_like(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-DAG:     func private @cast_csc_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @convert_layout(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_100:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_101:.*]] = memref.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_100]]{{\[}}%[[VAL_101]]] : memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_13:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_14:.*]] = call @matrix_empty_like(%[[VAL_13]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_dim(%[[VAL_14]], %[[VAL_1]], %[[VAL_8]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_dim(%[[VAL_14]], %[[VAL_2]], %[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_pointers(%[[VAL_14]], %[[VAL_2]], %[[VAL_10]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_index(%[[VAL_14]], %[[VAL_2]], %[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_values(%[[VAL_14]], %[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_15:.*]] = call @cast_csx_to_csc(%[[VAL_14]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.pointers %[[VAL_15]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_15]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_15]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_19:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             memref.store %[[VAL_3]], %[[VAL_16]]{{\[}}%[[VAL_19]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_20:.*]] = %[[VAL_1]] to %[[VAL_12]] step %[[VAL_2]] {
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_20]]] : memref<?xi64>
// CHECK:             %[[VAL_22:.*]] = index_cast %[[VAL_21]] : i64 to index
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = addi %[[VAL_23]], %[[VAL_4]] : i64
// CHECK:             memref.store %[[VAL_24]], %[[VAL_16]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_3]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_25:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_27]], %[[VAL_16]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = addi %[[VAL_27]], %[[VAL_26]] : i64
// CHECK:             memref.store %[[VAL_28]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_29:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_30:.*]] = index_cast %[[VAL_29]] : index to i64
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             %[[VAL_32:.*]] = index_cast %[[VAL_31]] : i64 to index
// CHECK:             %[[VAL_33:.*]] = addi %[[VAL_29]], %[[VAL_2]] : index
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:             %[[VAL_35:.*]] = index_cast %[[VAL_34]] : i64 to index
// CHECK:             scf.for %[[VAL_36:.*]] = %[[VAL_32]] to %[[VAL_35]] step %[[VAL_2]] {
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:               %[[VAL_38:.*]] = index_cast %[[VAL_37]] : i64 to index
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:               %[[VAL_40:.*]] = index_cast %[[VAL_39]] : i64 to index
// CHECK:               memref.store %[[VAL_30]], %[[VAL_17]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_36]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_41]], %[[VAL_18]]{{\[}}%[[VAL_40]]] : memref<?xf64>
// CHECK:               %[[VAL_42:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:               %[[VAL_43:.*]] = addi %[[VAL_42]], %[[VAL_4]] : i64
// CHECK:               memref.store %[[VAL_43]], %[[VAL_16]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_44:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_45:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_46:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_47]], %[[VAL_16]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_46]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_44]], %[[VAL_16]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           return %[[VAL_15]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @convert_layout(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSC64> {
    %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
    return %answer : tensor<?x?xf64, #CSC64>
}


// CHECK-LABEL:   func @convert_layout_csc(
// CHECK:           return %[[VAL_1:.*]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
func @convert_layout_csc(%sparse_tensor: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSC64> to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}


// CHECK-LABEL:   func @convert_layout_noop(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           return %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
func @convert_layout_noop(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
