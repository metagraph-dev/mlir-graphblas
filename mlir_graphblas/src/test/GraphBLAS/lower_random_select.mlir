// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

func private @choose_uniform(!llvm.ptr<i8>, i64, i64, memref<?xi64, #map>, memref<?xf64, #map>)

// CHECK-LABEL:   func @select_random_uniform(
// CHECK-SAME:                                %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                %[[VAL_2:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
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
// CHECK:           memref.store %[[VAL_3]], %[[VAL_13]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:           scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_4]] : index
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.cmpi ule, %[[VAL_20]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_20]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_22]] : i64
// CHECK:             memref.store %[[VAL_24]], %[[VAL_13]]{{\[}}%[[VAL_17]]] : memref<?xi64>
// CHECK:           }
// CHECK:           scf.parallel (%[[VAL_25:.*]]) = (%[[VAL_5]]) to (%[[VAL_6]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_4]] : index
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = arith.index_cast %[[VAL_27]] : i64 to index
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : i64 to index
// CHECK:             %[[VAL_31:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_25]]] : memref<?xi64>
// CHECK:             %[[VAL_32:.*]] = arith.index_cast %[[VAL_31]] : i64 to index
// CHECK:             %[[VAL_33:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             %[[VAL_34:.*]] = arith.index_cast %[[VAL_33]] : i64 to index
// CHECK:             %[[VAL_35:.*]] = arith.subi %[[VAL_30]], %[[VAL_28]] : index
// CHECK:             %[[VAL_36:.*]] = arith.index_cast %[[VAL_35]] : index to i64
// CHECK:             %[[VAL_37:.*]] = arith.subi %[[VAL_34]], %[[VAL_32]] : index
// CHECK:             %[[VAL_38:.*]] = arith.index_cast %[[VAL_37]] : index to i64
// CHECK:             %[[VAL_39:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_37]] : index
// CHECK:             %[[VAL_40:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_32]]] {{\[}}%[[VAL_37]]] {{\[}}%[[VAL_4]]] : memref<?xi64> to memref<?xi64, #map>
// CHECK:             %[[VAL_41:.*]] = memref.subview %[[VAL_15]]{{\[}}%[[VAL_32]]] {{\[}}%[[VAL_37]]] {{\[}}%[[VAL_4]]] : memref<?xf64> to memref<?xf64, #map>
// CHECK:             %[[VAL_42:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_28]]] {{\[}}%[[VAL_35]]] {{\[}}%[[VAL_4]]] : memref<?xi64> to memref<?xi64, #map>
// CHECK:             %[[VAL_43:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_28]]] {{\[}}%[[VAL_35]]] {{\[}}%[[VAL_4]]] : memref<?xf64> to memref<?xf64, #map>
// CHECK:             scf.if %[[VAL_39]] {
// CHECK:               memref.copy %[[VAL_42]], %[[VAL_40]] : memref<?xi64, #map> to memref<?xi64, #map>
// CHECK:               memref.copy %[[VAL_43]], %[[VAL_41]] : memref<?xf64, #map> to memref<?xf64, #map>
// CHECK:             } else {
// CHECK:               call @choose_uniform(%[[VAL_2]], %[[VAL_38]], %[[VAL_36]], %[[VAL_40]], %[[VAL_43]]) : (!llvm.ptr<i8>, i64, i64, memref<?xi64, #map>, memref<?xf64, #map>) -> ()
// CHECK:               scf.parallel (%[[VAL_44:.*]]) = (%[[VAL_5]]) to (%[[VAL_37]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_40]]{{\[}}%[[VAL_44]]] : memref<?xi64, #map>
// CHECK:                 %[[VAL_46:.*]] = arith.index_cast %[[VAL_45]] : i64 to index
// CHECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_42]]{{\[}}%[[VAL_46]]] : memref<?xi64, #map>
// CHECK:                 %[[VAL_48:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_46]]] : memref<?xf64, #map>
// CHECK:                 memref.store %[[VAL_47]], %[[VAL_40]]{{\[}}%[[VAL_44]]] : memref<?xi64, #map>
// CHECK:                 memref.store %[[VAL_48]], %[[VAL_41]]{{\[}}%[[VAL_44]]] : memref<?xf64, #map>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_12]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }
func @select_random_uniform(%sparse_tensor: tensor<?x?xf64, #CSR64>, %n: i64, %ctx: !llvm.ptr<i8>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_select_random %sparse_tensor, %n, %ctx { choose_n = @choose_uniform } : (tensor<?x?xf64, #CSR64>, i64, !llvm.ptr<i8>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}
