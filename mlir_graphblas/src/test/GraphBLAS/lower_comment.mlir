// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @select_triu(
// CHECK-SAME:                      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_9:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = call @dup_tensor(%[[VAL_9]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_11:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_10]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_11]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_11]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_11]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:             %[[VAL_16:.*]] = addi %[[VAL_15]], %[[VAL_3]] : index
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_17]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = index_cast %[[VAL_18]] : i64 to index
// CHECK:             %[[VAL_21:.*]] = index_cast %[[VAL_19]] : i64 to index
// CHECK:             scf.for %[[VAL_22:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_3]] {
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_22]]] : memref<?xi64>
// CHECK:               %[[VAL_24:.*]] = index_cast %[[VAL_23]] : i64 to index
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:               %[[VAL_26:.*]] = cmpi ugt, %[[VAL_24]], %[[VAL_15]] : index
// CHECK:               scf.if %[[VAL_26]] {
// CHECK:                 %[[VAL_27:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:                 %[[VAL_28:.*]] = index_cast %[[VAL_27]] : i64 to index
// CHECK:                 memref.store %[[VAL_23]], %[[VAL_13]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_25]], %[[VAL_14]]{{\[}}%[[VAL_28]]] : memref<?xf64>
// CHECK:                 %[[VAL_29:.*]] = addi %[[VAL_27]], %[[VAL_2]] : i64
// CHECK:                 memref.store %[[VAL_29]], %[[VAL_12]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = sparse_tensor.pointers %[[VAL_11]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_31:.*]] = tensor.dim %[[VAL_11]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_32:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:           %[[VAL_33:.*]] = index_cast %[[VAL_32]] : i64 to index
// CHECK:           %[[VAL_34:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_34]], %[[VAL_3]], %[[VAL_33]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_35:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_11]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_35]], %[[VAL_33]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           return %[[VAL_11]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @select_triu(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    graphblas.comment { comment = "comment number 1" } 
    graphblas.comment { comment = "comment number 2" } 
    %answer = graphblas.select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    graphblas.comment { comment = "comment number 3" } 
    return %answer : tensor<?x?xf64, #CSR64>
}

// CHECK:   func @do_nothing_func() {
// CHECK-NEXT:           return
func @do_nothing_func() -> () {
    graphblas.comment { comment = "comment number 1" } 
    graphblas.comment { comment = "comment number 2" } 
    graphblas.comment { comment = "comment number 3" } 
    return 
}
