// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_vector_multiply_plus_times(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> f64 {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = constant true
// CHECK-DAG:       %[[VAL_6:.*]] = constant false
// CHECK-DAG:       %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @empty_like(%[[VAL_9]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_11:.*]] = call @ptr8_to_vector_f64_p64i64(%[[VAL_10]]) : (!llvm.ptr<i8>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_12]], %[[VAL_3]], %[[VAL_4]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_13:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_13]], %[[VAL_3]], %[[VAL_2]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_14:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_14]], %[[VAL_3]], %[[VAL_4]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_15:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_15]], %[[VAL_4]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.indices %[[VAL_11]], %[[VAL_3]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.values %[[VAL_11]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_24:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:           %[[VAL_25:.*]] = index_cast %[[VAL_24]] : i64 to index
// CHECK:           %[[VAL_26:.*]] = memref.alloc(%[[VAL_8]]) : memref<?xf64>
// CHECK:           %[[VAL_27:.*]] = memref.alloc(%[[VAL_8]]) : memref<?xi1>
// CHECK:           linalg.fill(%[[VAL_6]], %[[VAL_27]]) : i1, memref<?xi1>
// CHECK:           scf.parallel (%[[VAL_28:.*]]) = (%[[VAL_3]]) to (%[[VAL_25]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = index_cast %[[VAL_29]] : i64 to index
// CHECK:             memref.store %[[VAL_5]], %[[VAL_27]]{{\[}}%[[VAL_30]]] : memref<?xi1>
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_28]]] : memref<?xf64>
// CHECK:             memref.store %[[VAL_31]], %[[VAL_26]]{{\[}}%[[VAL_30]]] : memref<?xf64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = scf.for %[[VAL_33:.*]] = %[[VAL_3]] to %[[VAL_4]] step %[[VAL_4]] iter_args(%[[VAL_34:.*]] = %[[VAL_3]]) -> (index) {
// CHECK:             %[[VAL_35:.*]] = index_cast %[[VAL_33]] : index to i64
// CHECK:             %[[VAL_36:.*]] = addi %[[VAL_33]], %[[VAL_4]] : index
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:             %[[VAL_38:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             %[[VAL_39:.*]] = index_cast %[[VAL_37]] : i64 to index
// CHECK:             %[[VAL_40:.*]] = index_cast %[[VAL_38]] : i64 to index
// CHECK:             %[[VAL_41:.*]]:2 = scf.for %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_40]] step %[[VAL_4]] iter_args(%[[VAL_43:.*]] = %[[VAL_7]], %[[VAL_44:.*]] = %[[VAL_6]]) -> (f64, i1) {
// CHECK:               %[[VAL_45:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:               %[[VAL_46:.*]] = index_cast %[[VAL_45]] : i64 to index
// CHECK:               %[[VAL_47:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_46]]] : memref<?xi1>
// CHECK:               %[[VAL_48:.*]]:2 = scf.if %[[VAL_47]] -> (f64, i1) {
// CHECK:                 %[[VAL_49:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_46]]] : memref<?xf64>
// CHECK:                 %[[VAL_50:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_42]]] : memref<?xf64>
// CHECK:                 %[[VAL_51:.*]] = mulf %[[VAL_49]], %[[VAL_50]] : f64
// CHECK:                 %[[VAL_52:.*]] = addf %[[VAL_43]], %[[VAL_51]] : f64
// CHECK:                 scf.yield %[[VAL_52]], %[[VAL_5]] : f64, i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_43]], %[[VAL_44]] : f64, i1
// CHECK:               }
// CHECK:               scf.yield %[[VAL_53:.*]]#0, %[[VAL_53]]#1 : f64, i1
// CHECK:             }
// CHECK:             %[[VAL_54:.*]] = scf.if %[[VAL_55:.*]]#1 -> (index) {
// CHECK:               memref.store %[[VAL_35]], %[[VAL_22]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_55]]#0, %[[VAL_23]]{{\[}}%[[VAL_34]]] : memref<?xf64>
// CHECK:               %[[VAL_56:.*]] = addi %[[VAL_34]], %[[VAL_4]] : index
// CHECK:               scf.yield %[[VAL_56]] : index
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_34]] : index
// CHECK:             }
// CHECK:             scf.yield %[[VAL_57:.*]] : index
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_26]] : memref<?xf64>
// CHECK:           memref.dealloc %[[VAL_27]] : memref<?xi1>
// CHECK:           %[[VAL_58:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_3]]] : memref<?xf64>
// CHECK:           sparse_tensor.release %[[VAL_11]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_58]] : f64
// CHECK:         }

func @vector_vector_multiply_plus_times(%a: tensor<?xf64, #CV64>, %b: tensor<?xf64, #CV64>) -> f64 {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to f64
    return %answer : f64
}

// TODO: Check all type combinations
