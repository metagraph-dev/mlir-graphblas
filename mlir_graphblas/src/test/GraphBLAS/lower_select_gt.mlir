// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @select_gt(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_10:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_11:.*]] = call @dup_tensor(%[[VAL_10]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_12:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_11]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_12]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_12]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_12]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_4]] : index
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_18]], %[[VAL_13]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_19]] : i64 to index
// CHECK:             %[[VAL_22:.*]] = arith.index_cast %[[VAL_20]] : i64 to index
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_4]] {
// CHECK:               %[[VAL_24:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_23]]] : memref<?xf64>
// CHECK:               %[[VAL_26:.*]] = arith.cmpf ogt, %[[VAL_25]], %[[VAL_1]] : f64
// CHECK:               scf.if %[[VAL_26]] {
// CHECK:                 %[[VAL_27:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:                 %[[VAL_28:.*]] = arith.index_cast %[[VAL_27]] : i64 to index
// CHECK:                 memref.store %[[VAL_24]], %[[VAL_14]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_25]], %[[VAL_15]]{{\[}}%[[VAL_28]]] : memref<?xf64>
// CHECK:                 %[[VAL_29:.*]] = arith.addi %[[VAL_27]], %[[VAL_3]] : i64
// CHECK:                 memref.store %[[VAL_29]], %[[VAL_13]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = sparse_tensor.pointers %[[VAL_12]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_31:.*]] = tensor.dim %[[VAL_12]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_32:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:           %[[VAL_34:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_34]], %[[VAL_4]], %[[VAL_33]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_35:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_35]], %[[VAL_33]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           return %[[VAL_12]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @select_gt(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %c0_f64 = arith.constant 0.0 : f64
    %answer = graphblas.select %sparse_tensor, %c0_f64 { selectors = ["gt"] } : tensor<?x?xf64, #CSR64>, f64 to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
