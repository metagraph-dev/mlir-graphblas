// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_matrix_multiply_plus_times(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = call @empty_like(%[[VAL_12]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = call @ptr8_to_vector_f64_p64i64(%[[VAL_13]]) : (!llvm.ptr<i8>) -> tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_15:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_14]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_15]], %[[VAL_5]], %[[VAL_10]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_16:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_14]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_16]], %[[VAL_5]], %[[VAL_2]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_18:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.pointers %[[VAL_14]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_24:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : i64 to index
// CHECK:           %[[VAL_26:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_25]] : index
// CHECK:           %[[VAL_27:.*]] = scf.if %[[VAL_26]] -> (i64) {
// CHECK:             scf.yield %[[VAL_3]] : i64
// CHECK:           } else {
// CHECK:             %[[VAL_28:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:             linalg.fill(%[[VAL_8]], %[[VAL_28]]) : i1, memref<?xi1>
// CHECK:             scf.parallel (%[[VAL_29:.*]]) = (%[[VAL_5]]) to (%[[VAL_25]]) step (%[[VAL_6]]) {
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:               %[[VAL_31:.*]] = arith.index_cast %[[VAL_30]] : i64 to index
// CHECK:               memref.store %[[VAL_7]], %[[VAL_28]]{{\[}}%[[VAL_31]]] : memref<?xi1>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             %[[VAL_32:.*]] = scf.parallel (%[[VAL_33:.*]]) = (%[[VAL_5]]) to (%[[VAL_10]]) step (%[[VAL_6]]) init (%[[VAL_3]]) -> i64 {
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_6]] : index
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:               %[[VAL_36:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:               %[[VAL_37:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_36]] : i64
// CHECK:               %[[VAL_38:.*]] = scf.if %[[VAL_37]] -> (i64) {
// CHECK:                 scf.yield %[[VAL_3]] : i64
// CHECK:               } else {
// CHECK:                 %[[VAL_39:.*]] = scf.while (%[[VAL_40:.*]] = %[[VAL_35]]) : (i64) -> i64 {
// CHECK:                   %[[VAL_41:.*]] = arith.cmpi uge, %[[VAL_40]], %[[VAL_36]] : i64
// CHECK:                   %[[VAL_42:.*]]:2 = scf.if %[[VAL_41]] -> (i1, i64) {
// CHECK:                     scf.yield %[[VAL_8]], %[[VAL_3]] : i1, i64
// CHECK:                   } else {
// CHECK:                     %[[VAL_43:.*]] = arith.index_cast %[[VAL_40]] : i64 to index
// CHECK:                     %[[VAL_44:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:                     %[[VAL_45:.*]] = arith.index_cast %[[VAL_44]] : i64 to index
// CHECK:                     %[[VAL_46:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_45]]] : memref<?xi1>
// CHECK:                     %[[VAL_47:.*]] = select %[[VAL_46]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                     %[[VAL_48:.*]] = select %[[VAL_46]], %[[VAL_4]], %[[VAL_40]] : i64
// CHECK:                     scf.yield %[[VAL_47]], %[[VAL_48]] : i1, i64
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_49:.*]]#0) %[[VAL_49]]#1 : i64
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_50:.*]]: i64):
// CHECK:                   %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_4]] : i64
// CHECK:                   scf.yield %[[VAL_51]] : i64
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_52:.*]] : i64
// CHECK:               }
// CHECK:               scf.reduce(%[[VAL_53:.*]])  : i64 {
// CHECK:               ^bb0(%[[VAL_54:.*]]: i64, %[[VAL_55:.*]]: i64):
// CHECK:                 %[[VAL_56:.*]] = arith.addi %[[VAL_54]], %[[VAL_55]] : i64
// CHECK:                 scf.reduce.return %[[VAL_56]] : i64
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_28]] : memref<?xi1>
// CHECK:             scf.yield %[[VAL_57:.*]] : i64
// CHECK:           }
// CHECK:           %[[VAL_58:.*]] = arith.index_cast %[[VAL_59:.*]] : i64 to index
// CHECK:           memref.store %[[VAL_59]], %[[VAL_23]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:           %[[VAL_60:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_14]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_60]], %[[VAL_5]], %[[VAL_58]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_61:.*]] = call @vector_f64_p64i64_to_ptr8(%[[VAL_14]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_61]], %[[VAL_58]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_62:.*]] = sparse_tensor.indices %[[VAL_14]], %[[VAL_5]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_63:.*]] = sparse_tensor.values %[[VAL_14]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_64:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_58]] : index
// CHECK:           scf.if %[[VAL_64]] {
// CHECK:             %[[VAL_65:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xf64>
// CHECK:             %[[VAL_66:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:             linalg.fill(%[[VAL_8]], %[[VAL_66]]) : i1, memref<?xi1>
// CHECK:             scf.parallel (%[[VAL_67:.*]]) = (%[[VAL_5]]) to (%[[VAL_25]]) step (%[[VAL_6]]) {
// CHECK:               %[[VAL_68:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_67]]] : memref<?xi64>
// CHECK:               %[[VAL_69:.*]] = arith.index_cast %[[VAL_68]] : i64 to index
// CHECK:               memref.store %[[VAL_7]], %[[VAL_66]]{{\[}}%[[VAL_69]]] : memref<?xi1>
// CHECK:               %[[VAL_70:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_67]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_70]], %[[VAL_65]]{{\[}}%[[VAL_69]]] : memref<?xf64>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             %[[VAL_71:.*]] = scf.for %[[VAL_72:.*]] = %[[VAL_5]] to %[[VAL_10]] step %[[VAL_6]] iter_args(%[[VAL_73:.*]] = %[[VAL_5]]) -> (index) {
// CHECK:               %[[VAL_74:.*]] = arith.index_cast %[[VAL_72]] : index to i64
// CHECK:               %[[VAL_75:.*]] = arith.addi %[[VAL_72]], %[[VAL_6]] : index
// CHECK:               %[[VAL_76:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_72]]] : memref<?xi64>
// CHECK:               %[[VAL_77:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_75]]] : memref<?xi64>
// CHECK:               %[[VAL_78:.*]] = arith.index_cast %[[VAL_76]] : i64 to index
// CHECK:               %[[VAL_79:.*]] = arith.index_cast %[[VAL_77]] : i64 to index
// CHECK:               %[[VAL_80:.*]]:2 = scf.for %[[VAL_81:.*]] = %[[VAL_78]] to %[[VAL_79]] step %[[VAL_6]] iter_args(%[[VAL_82:.*]] = %[[VAL_9]], %[[VAL_83:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                 %[[VAL_84:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:                 %[[VAL_85:.*]] = arith.index_cast %[[VAL_84]] : i64 to index
// CHECK:                 %[[VAL_86:.*]] = memref.load %[[VAL_66]]{{\[}}%[[VAL_85]]] : memref<?xi1>
// CHECK:                 %[[VAL_87:.*]]:2 = scf.if %[[VAL_86]] -> (f64, i1) {
// CHECK:                   %[[VAL_88:.*]] = memref.load %[[VAL_65]]{{\[}}%[[VAL_85]]] : memref<?xf64>
// CHECK:                   %[[VAL_89:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_81]]] : memref<?xf64>
// CHECK:                   %[[VAL_90:.*]] = arith.mulf %[[VAL_88]], %[[VAL_89]] : f64
// CHECK:                   %[[VAL_91:.*]] = arith.addf %[[VAL_82]], %[[VAL_90]] : f64
// CHECK:                   scf.yield %[[VAL_91]], %[[VAL_7]] : f64, i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_82]], %[[VAL_83]] : f64, i1
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_92:.*]]#0, %[[VAL_92]]#1 : f64, i1
// CHECK:               }
// CHECK:               %[[VAL_93:.*]] = scf.if %[[VAL_94:.*]]#1 -> (index) {
// CHECK:                 memref.store %[[VAL_74]], %[[VAL_62]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_94]]#0, %[[VAL_63]]{{\[}}%[[VAL_73]]] : memref<?xf64>
// CHECK:                 %[[VAL_95:.*]] = arith.addi %[[VAL_73]], %[[VAL_6]] : index
// CHECK:                 scf.yield %[[VAL_95]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_73]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_96:.*]] : index
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_65]] : memref<?xf64>
// CHECK:             memref.dealloc %[[VAL_66]] : memref<?xi1>
// CHECK:           }
// CHECK:           return %[[VAL_14]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }


func @vector_matrix_multiply_plus_times(%a: tensor<?xf64, #CV64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?xf64, #CV64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?xf64, #CV64>, tensor<?x?xf64, #CSC64>) to tensor<?xf64, #CV64>
    return %answer : tensor<?xf64, #CV64>
}

// TODO: Check all type combinations
