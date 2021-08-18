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

module {
// CHECK-LABEL:   func @transpose_different_compression(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 0 : index
// CHECK:           %[[VAL_2:.*]] = constant 1 : index
// CHECK:           %[[VAL_3:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_4:.*]] = call @dup_matrix(%[[VAL_3]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_5:.*]] = call @cast_csx_to_csc(%[[VAL_4]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_8:.*]] = cmpi ne, %[[VAL_6]], %[[VAL_7]] : index
// CHECK:           scf.if %[[VAL_8]] {
// CHECK:             %[[VAL_9:.*]] = call @cast_csc_to_csx(%[[VAL_5]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_9]], %[[VAL_1]], %[[VAL_7]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_10:.*]] = call @cast_csc_to_csx(%[[VAL_5]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_10]], %[[VAL_2]], %[[VAL_6]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_5]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
    func @transpose_different_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSC64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
        return %answer : tensor<?x?xi64, #CSC64>
    }
 
// CHECK-LABEL:   func @transpose_same_compression(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = constant 0 : index
// CHECK:           %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = addi %[[VAL_9]], %[[VAL_4]] : index
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = index_cast %[[VAL_13]] : i64 to index
// CHECK:           %[[VAL_15:.*]] = call @cast_csr_to_csx(%[[VAL_0]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = call @matrix_empty_like(%[[VAL_15]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @matrix_resize_dim(%[[VAL_16]], %[[VAL_3]], %[[VAL_8]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_dim(%[[VAL_16]], %[[VAL_4]], %[[VAL_9]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_pointers(%[[VAL_16]], %[[VAL_4]], %[[VAL_10]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_index(%[[VAL_16]], %[[VAL_4]], %[[VAL_14]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @matrix_resize_values(%[[VAL_16]], %[[VAL_14]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_17:.*]] = call @cast_csx_to_csc(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.pointers %[[VAL_17]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_17]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_17]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.for %[[VAL_21:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             memref.store %[[VAL_1]], %[[VAL_18]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_22:.*]] = %[[VAL_3]] to %[[VAL_14]] step %[[VAL_4]] {
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_24]]] : memref<?xi64>
// CHECK:             %[[VAL_26:.*]] = addi %[[VAL_25]], %[[VAL_2]] : i64
// CHECK:             memref.store %[[VAL_26]], %[[VAL_18]]{{\[}}%[[VAL_24]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_1]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_27:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<?xi64>
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_29]], %[[VAL_18]]{{\[}}%[[VAL_27]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_28]] : i64
// CHECK:             memref.store %[[VAL_30]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_31:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_4]] {
// CHECK:             %[[VAL_32:.*]] = index_cast %[[VAL_31]] : index to i64
// CHECK:             %[[VAL_33:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_34:.*]] = index_cast %[[VAL_33]] : i64 to index
// CHECK:             %[[VAL_35:.*]] = addi %[[VAL_31]], %[[VAL_4]] : index
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:             %[[VAL_37:.*]] = index_cast %[[VAL_36]] : i64 to index
// CHECK:             scf.for %[[VAL_38:.*]] = %[[VAL_34]] to %[[VAL_37]] step %[[VAL_4]] {
// CHECK:               %[[VAL_39:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:               %[[VAL_40:.*]] = index_cast %[[VAL_39]] : i64 to index
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:               %[[VAL_42:.*]] = index_cast %[[VAL_41]] : i64 to index
// CHECK:               memref.store %[[VAL_32]], %[[VAL_19]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_43]], %[[VAL_20]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:               %[[VAL_45:.*]] = addi %[[VAL_44]], %[[VAL_2]] : i64
// CHECK:               memref.store %[[VAL_45]], %[[VAL_18]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_1]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_47:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             %[[VAL_48:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:             %[[VAL_49:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_49]], %[[VAL_18]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_48]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_46]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           %[[VAL_50:.*]] = call @cast_csc_to_csx(%[[VAL_17]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_51:.*]] = call @dup_matrix(%[[VAL_50]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_52:.*]] = call @cast_csx_to_csr(%[[VAL_51]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_53:.*]] = tensor.dim %[[VAL_17]], %[[VAL_3]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_54:.*]] = tensor.dim %[[VAL_17]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_55:.*]] = cmpi ne, %[[VAL_53]], %[[VAL_54]] : index
// CHECK:           scf.if %[[VAL_55]] {
// CHECK:             %[[VAL_56:.*]] = call @cast_csr_to_csx(%[[VAL_52]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_56]], %[[VAL_3]], %[[VAL_54]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_57:.*]] = call @cast_csr_to_csx(%[[VAL_52]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_57]], %[[VAL_4]], %[[VAL_53]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = call @cast_csc_to_csx(%[[VAL_17]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @delSparseTensor(%[[VAL_58]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> ()
// CHECK:           return %[[VAL_52]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
    func @transpose_same_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSR64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>
        return %answer : tensor<?x?xi64, #CSR64>
    }
   
}
