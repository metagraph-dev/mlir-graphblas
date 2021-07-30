// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-DAG:     func private @delSparseVector(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
// CHECK-DAG:     func private @vector_resize_values(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK-DAG:     func private @vector_resize_index(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_pointers(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_resize_dim(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK-DAG:     func private @vector_empty_like(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>

// CHECK-LABEL:   func @vector_vector_multiply_plus_times(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = constant true
// CHECK-DAG:       %[[VAL_6:.*]] = constant false
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = call @vector_empty_like(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @vector_resize_dim(%[[VAL_9]], %[[VAL_3]], %[[VAL_4]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_pointers(%[[VAL_9]], %[[VAL_3]], %[[VAL_2]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_index(%[[VAL_9]], %[[VAL_3]], %[[VAL_4]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @vector_resize_values(%[[VAL_9]], %[[VAL_4]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = index_cast %[[VAL_18]] : i64 to index
// CHECK:           %[[VAL_20:.*]] = memref.alloc(%[[VAL_8]]) : memref<?xf64>
// CHECK:           %[[VAL_21:.*]] = memref.alloc(%[[VAL_8]]) : memref<?xi1>
// CHECK:           linalg.fill(%[[VAL_21]], %[[VAL_6]]) : memref<?xi1>, i1
// CHECK:           scf.parallel (%[[VAL_22:.*]]) = (%[[VAL_3]]) to (%[[VAL_19]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:             memref.store %[[VAL_5]], %[[VAL_21]]{{\[}}%[[VAL_24]]] : memref<?xi1>
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:             memref.store %[[VAL_25]], %[[VAL_20]]{{\[}}%[[VAL_24]]] : memref<?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_4]] iter_args(%[[VAL_28:.*]] = %[[VAL_3]]) -> (index) {
// CHECK:             %[[VAL_29:.*]] = index_cast %[[VAL_27]] : index to i64
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_27]], %[[VAL_4]] : index
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_27]]] : memref<?xi64>
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = index_cast %[[VAL_31]] : i64 to index
// CHECK:             %[[VAL_34:.*]] = index_cast %[[VAL_32]] : i64 to index
// CHECK:             %[[VAL_35:.*]]:2 = scf.for %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_34]] step %[[VAL_4]] iter_args(%[[VAL_37:.*]] = %[[VAL_7]], %[[VAL_38:.*]] = %[[VAL_6]]) -> (f64, i1) {
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:               %[[VAL_40:.*]] = index_cast %[[VAL_39]] : i64 to index
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_40]]] : memref<?xi1>
// CHECK:               %[[VAL_42:.*]]:2 = scf.if %[[VAL_41]] -> (f64, i1) {
// CHECK:                 %[[VAL_43:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_40]]] : memref<?xf64>
// CHECK:                 %[[VAL_44:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_36]]] : memref<?xf64>
// CHECK:                 %[[VAL_45:.*]] = mulf %[[VAL_43]], %[[VAL_44]] : f64
// CHECK:                 %[[VAL_46:.*]] = addf %[[VAL_37]], %[[VAL_45]] : f64
// CHECK:                 scf.yield %[[VAL_46]], %[[VAL_5]] : f64, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_37]], %[[VAL_38]] : f64, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_47:.*]]#0, %[[VAL_47]]#1 : f64, i1
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = scf.if %[[VAL_35]]#1 -> (index) {
// CHECK:               memref.store %[[VAL_29]], %[[VAL_16]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_35]]#0, %[[VAL_17]]{{\[}}%[[VAL_28]]] : memref<?xf64>
// CHECK:               %[[VAL_50:.*]] = addi %[[VAL_28]], %[[VAL_4]] : index
// CHECK:               scf.yield %[[VAL_50]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_28]] : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_48]] : index
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_20]] : memref<?xf64>
// CHECK:           memref.dealloc %[[VAL_21]] : memref<?xi1>
// CHECK:           %[[VAL_52:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_3]]] : memref<?xf64>
// CHECK:           call @delSparseVector(%[[VAL_9]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> ()
// CHECK:           return %[[VAL_52]] : f64
// CHECK:         }

func @vector_vector_multiply_plus_times(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> f64 {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to f64
    return %answer : f64
}

// TODO: Check all type combinations
