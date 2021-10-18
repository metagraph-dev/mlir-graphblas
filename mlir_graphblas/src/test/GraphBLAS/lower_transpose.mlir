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
// CHECK:           %[[VAL_3:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_4:.*]] = call @dup_tensor(%[[VAL_3]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = call @ptr8_to_matrix_csr_i64_p64i64(%[[VAL_4]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_6:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_5]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = call @ptr8_to_matrix_csc_i64_p64i64(%[[VAL_6]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_7]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
    func @transpose_different_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSC64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
        return %answer : tensor<?x?xi64, #CSC64>
    }
 
// CHECK-LABEL:   func @transpose_same_compression(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = index_cast %[[VAL_12]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = call @empty_like(%[[VAL_14]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = call @ptr8_to_matrix_csr_i64_p64i64(%[[VAL_15]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_17]], %[[VAL_3]], %[[VAL_9]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_18:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_18]], %[[VAL_4]], %[[VAL_8]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_19:.*]] = addi %[[VAL_9]], %[[VAL_4]] : index
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_20]], %[[VAL_4]], %[[VAL_19]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_21]], %[[VAL_4]], %[[VAL_13]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_22:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_22]], %[[VAL_13]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_23:.*]] = call @matrix_csr_i64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_24:.*]] = call @ptr8_to_matrix_csc_i64_p64i64(%[[VAL_23]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.pointers %[[VAL_24]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.indices %[[VAL_24]], %[[VAL_4]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.values %[[VAL_24]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.for %[[VAL_28:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             memref.store %[[VAL_1]], %[[VAL_25]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_29:.*]] = %[[VAL_3]] to %[[VAL_13]] step %[[VAL_4]] {
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             %[[VAL_31:.*]] = index_cast %[[VAL_30]] : i64 to index
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = addi %[[VAL_32]], %[[VAL_2]] : i64
// CHECK:             memref.store %[[VAL_33]], %[[VAL_25]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_1]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_34:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_36]], %[[VAL_25]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:             %[[VAL_37:.*]] = addi %[[VAL_36]], %[[VAL_35]] : i64
// CHECK:             memref.store %[[VAL_37]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_38:.*]] = %[[VAL_3]] to %[[VAL_8]] step %[[VAL_4]] {
// CHECK:             %[[VAL_39:.*]] = index_cast %[[VAL_38]] : index to i64
// CHECK:             %[[VAL_40:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:             %[[VAL_41:.*]] = index_cast %[[VAL_40]] : i64 to index
// CHECK:             %[[VAL_42:.*]] = addi %[[VAL_38]], %[[VAL_4]] : index
// CHECK:             %[[VAL_43:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             %[[VAL_44:.*]] = index_cast %[[VAL_43]] : i64 to index
// CHECK:             scf.for %[[VAL_45:.*]] = %[[VAL_41]] to %[[VAL_44]] step %[[VAL_4]] {
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:               %[[VAL_47:.*]] = index_cast %[[VAL_46]] : i64 to index
// CHECK:               %[[VAL_48:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:               %[[VAL_49:.*]] = index_cast %[[VAL_48]] : i64 to index
// CHECK:               memref.store %[[VAL_39]], %[[VAL_26]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_50]], %[[VAL_27]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:               %[[VAL_51:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:               %[[VAL_52:.*]] = addi %[[VAL_51]], %[[VAL_2]] : i64
// CHECK:               memref.store %[[VAL_52]], %[[VAL_25]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_1]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_54:.*]] = %[[VAL_3]] to %[[VAL_9]] step %[[VAL_4]] {
// CHECK:             %[[VAL_55:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_54]]] : memref<?xi64>
// CHECK:             %[[VAL_56:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_56]], %[[VAL_25]]{{\[}}%[[VAL_54]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_55]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_53]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           %[[VAL_57:.*]] = call @matrix_csc_i64_p64i64_to_ptr8(%[[VAL_24]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_58:.*]] = call @dup_tensor(%[[VAL_57]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_59:.*]] = call @ptr8_to_matrix_csc_i64_p64i64(%[[VAL_58]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_60:.*]] = call @matrix_csc_i64_p64i64_to_ptr8(%[[VAL_59]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_61:.*]] = call @ptr8_to_matrix_csr_i64_p64i64(%[[VAL_60]]) : (!llvm.ptr<i8>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           sparse_tensor.release %[[VAL_24]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           return %[[VAL_61]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
    func @transpose_same_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSR64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>
        return %answer : tensor<?x?xi64, #CSR64>
    }
   
}
