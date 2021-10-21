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

// CHECK-LABEL:   func private @matrix_csr_f64_p64i64_to_ptr8(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:         func private @ptr8_to_matrix_csr_f64_p64i64(!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @resize_values(!llvm.ptr<i8>, index)
// CHECK:         func private @resize_index(!llvm.ptr<i8>, index, index)
// CHECK:         func private @resize_pointers(!llvm.ptr<i8>, index, index)
// CHECK:         func private @resize_dim(!llvm.ptr<i8>, index, index)
// CHECK:         func private @assign_rev(!llvm.ptr<i8>, index, index)
// CHECK:         func private @ptr8_to_matrix_csc_f64_p64i64(!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         func private @empty_like(!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         func private @matrix_csc_f64_p64i64_to_ptr8(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>

// CHECK-LABEL:   func @convert_layout(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = call @empty_like(%[[VAL_14]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_15]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @assign_rev(%[[VAL_17]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_18:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @assign_rev(%[[VAL_18]], %[[VAL_2]], %[[VAL_1]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_19:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_19]], %[[VAL_1]], %[[VAL_9]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_20]], %[[VAL_2]], %[[VAL_8]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_22:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_22]], %[[VAL_2]], %[[VAL_21]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_23:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_23]], %[[VAL_2]], %[[VAL_13]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_24:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_24]], %[[VAL_13]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_25:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_26:.*]] = call @ptr8_to_matrix_csc_f64_p64i64(%[[VAL_25]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.pointers %[[VAL_26]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_28:.*]] = sparse_tensor.indices %[[VAL_26]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_29:.*]] = sparse_tensor.values %[[VAL_26]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_30:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             memref.store %[[VAL_3]], %[[VAL_27]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_31:.*]] = %[[VAL_1]] to %[[VAL_13]] step %[[VAL_2]] {
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_4]] : i64
// CHECK:             memref.store %[[VAL_35]], %[[VAL_27]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_3]], %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_36:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             %[[VAL_38:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_38]], %[[VAL_27]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_37]] : i64
// CHECK:             memref.store %[[VAL_39]], %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_40:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_41:.*]] = arith.index_cast %[[VAL_40]] : index to i64
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:             %[[VAL_43:.*]] = arith.index_cast %[[VAL_42]] : i64 to index
// CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_40]], %[[VAL_2]] : index
// CHECK:             %[[VAL_45:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:             %[[VAL_46:.*]] = arith.index_cast %[[VAL_45]] : i64 to index
// CHECK:             scf.for %[[VAL_47:.*]] = %[[VAL_43]] to %[[VAL_46]] step %[[VAL_2]] {
// CHECK:               %[[VAL_48:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:               %[[VAL_49:.*]] = arith.index_cast %[[VAL_48]] : i64 to index
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:               %[[VAL_51:.*]] = arith.index_cast %[[VAL_50]] : i64 to index
// CHECK:               memref.store %[[VAL_41]], %[[VAL_28]]{{\[}}%[[VAL_51]]] : memref<?xi64>
// CHECK:               %[[VAL_52:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_47]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_52]], %[[VAL_29]]{{\[}}%[[VAL_51]]] : memref<?xf64>
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:               %[[VAL_54:.*]] = arith.addi %[[VAL_53]], %[[VAL_4]] : i64
// CHECK:               memref.store %[[VAL_54]], %[[VAL_27]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_56:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_57:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_56]]] : memref<?xi64>
// CHECK:             %[[VAL_58:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_58]], %[[VAL_27]]{{\[}}%[[VAL_56]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_57]], %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_55]], %[[VAL_27]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           return %[[VAL_26]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
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
