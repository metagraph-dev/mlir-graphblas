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
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = sparse_tensor.init{{\[}}%[[VAL_14]], %[[VAL_15]]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_16]], %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : index
// CHECK:           %[[VAL_19:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_19]], %[[VAL_2]], %[[VAL_18]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @assign_rev(%[[VAL_20]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @assign_rev(%[[VAL_21]], %[[VAL_2]], %[[VAL_1]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_22:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_22]], %[[VAL_1]], %[[VAL_9]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_23:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_23]], %[[VAL_2]], %[[VAL_8]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
// CHECK:           %[[VAL_25:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_25]], %[[VAL_2]], %[[VAL_24]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_26:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_26]], %[[VAL_2]], %[[VAL_13]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_27:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_27]], %[[VAL_13]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_28:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_29:.*]] = call @ptr8_to_matrix_csc_f64_p64i64(%[[VAL_28]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_30:.*]] = sparse_tensor.pointers %[[VAL_29]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_31:.*]] = sparse_tensor.indices %[[VAL_29]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_32:.*]] = sparse_tensor.values %[[VAL_29]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_33:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             memref.store %[[VAL_3]], %[[VAL_30]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_34:.*]] = %[[VAL_1]] to %[[VAL_13]] step %[[VAL_2]] {
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : i64 to index
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_4]] : i64
// CHECK:             memref.store %[[VAL_38]], %[[VAL_30]]{{\[}}%[[VAL_36]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_3]], %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_39:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_40:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:             %[[VAL_41:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_41]], %[[VAL_30]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_40]] : i64
// CHECK:             memref.store %[[VAL_42]], %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.for %[[VAL_43:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : index to i64
// CHECK:             %[[VAL_45:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             %[[VAL_46:.*]] = arith.index_cast %[[VAL_45]] : i64 to index
// CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_43]], %[[VAL_2]] : index
// CHECK:             %[[VAL_48:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_47]]] : memref<?xi64>
// CHECK:             %[[VAL_49:.*]] = arith.index_cast %[[VAL_48]] : i64 to index
// CHECK:             scf.for %[[VAL_50:.*]] = %[[VAL_46]] to %[[VAL_49]] step %[[VAL_2]] {
// CHECK:               %[[VAL_51:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_50]]] : memref<?xi64>
// CHECK:               %[[VAL_52:.*]] = arith.index_cast %[[VAL_51]] : i64 to index
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:               %[[VAL_54:.*]] = arith.index_cast %[[VAL_53]] : i64 to index
// CHECK:               memref.store %[[VAL_44]], %[[VAL_31]]{{\[}}%[[VAL_54]]] : memref<?xi64>
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_50]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_55]], %[[VAL_32]]{{\[}}%[[VAL_54]]] : memref<?xf64>
// CHECK:               %[[VAL_56:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:               %[[VAL_57:.*]] = arith.addi %[[VAL_56]], %[[VAL_4]] : i64
// CHECK:               memref.store %[[VAL_57]], %[[VAL_30]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           memref.store %[[VAL_3]], %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_59:.*]] = %[[VAL_1]] to %[[VAL_9]] step %[[VAL_2]] {
// CHECK:             %[[VAL_60:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_59]]] : memref<?xi64>
// CHECK:             %[[VAL_61:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_61]], %[[VAL_30]]{{\[}}%[[VAL_59]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_60]], %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           memref.store %[[VAL_58]], %[[VAL_30]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           return %[[VAL_29]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
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
