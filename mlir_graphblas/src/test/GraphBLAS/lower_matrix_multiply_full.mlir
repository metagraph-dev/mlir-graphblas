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

// CHECK-LABEL:   func @matrix_multiply_plus_times(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = constant true
// CHECK-DAG:       %[[VAL_7:.*]] = constant false
// CHECK-DAG:       %[[VAL_8:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = addi %[[VAL_9]], %[[VAL_5]] : index
// CHECK:           %[[VAL_13:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = call @empty_like(%[[VAL_13]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_14]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_16]], %[[VAL_4]], %[[VAL_9]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_17:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_17]], %[[VAL_5]], %[[VAL_10]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_18:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_18]], %[[VAL_5]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_19:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_24:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.pointers %[[VAL_15]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_26:.*]]) = (%[[VAL_4]]) to (%[[VAL_9]]) step (%[[VAL_5]]) {
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = addi %[[VAL_26]], %[[VAL_5]] : index
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_28]]] : memref<?xi64>
// CHECK:             %[[VAL_30:.*]] = cmpi eq, %[[VAL_27]], %[[VAL_29]] : i64
// CHECK:             %[[VAL_31:.*]] = scf.if %[[VAL_30]] -> (i64) {
// CHECK:               scf.yield %[[VAL_2]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_32:.*]] = index_cast %[[VAL_27]] : i64 to index
// CHECK:               %[[VAL_33:.*]] = index_cast %[[VAL_29]] : i64 to index
// CHECK:               %[[VAL_34:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_7]], %[[VAL_34]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_35:.*]]) = (%[[VAL_32]]) to (%[[VAL_33]]) step (%[[VAL_5]]) {
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:                 %[[VAL_37:.*]] = index_cast %[[VAL_36]] : i64 to index
// CHECK:                 memref.store %[[VAL_6]], %[[VAL_34]]{{\[}}%[[VAL_37]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_38:.*]] = scf.parallel (%[[VAL_39:.*]]) = (%[[VAL_4]]) to (%[[VAL_10]]) step (%[[VAL_5]]) init (%[[VAL_2]]) -> i64 {
// CHECK:                 %[[VAL_40:.*]] = addi %[[VAL_39]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_41:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_39]]] : memref<?xi64>
// CHECK:                 %[[VAL_42:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_40]]] : memref<?xi64>
// CHECK:                 %[[VAL_43:.*]] = cmpi eq, %[[VAL_41]], %[[VAL_42]] : i64
// CHECK:                 %[[VAL_44:.*]] = scf.if %[[VAL_43]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_2]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_45:.*]] = scf.while (%[[VAL_46:.*]] = %[[VAL_41]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_47:.*]] = cmpi uge, %[[VAL_46]], %[[VAL_42]] : i64
// CHECK:                     %[[VAL_48:.*]]:2 = scf.if %[[VAL_47]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_7]], %[[VAL_2]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_49:.*]] = index_cast %[[VAL_46]] : i64 to index
// CHECK:                       %[[VAL_50:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:                       %[[VAL_51:.*]] = index_cast %[[VAL_50]] : i64 to index
// CHECK:                       %[[VAL_52:.*]] = memref.load %[[VAL_34]]{{\[}}%[[VAL_51]]] : memref<?xi1>
// CHECK:                       %[[VAL_53:.*]] = select %[[VAL_52]], %[[VAL_7]], %[[VAL_6]] : i1
// CHECK:                       %[[VAL_54:.*]] = select %[[VAL_52]], %[[VAL_3]], %[[VAL_46]] : i64
// CHECK:                       scf.yield %[[VAL_53]], %[[VAL_54]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_55:.*]]#0) %[[VAL_55]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_56:.*]]: i64):
// CHECK:                     %[[VAL_57:.*]] = addi %[[VAL_56]], %[[VAL_3]] : i64
// CHECK:                     scf.yield %[[VAL_57]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_58:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_59:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_60:.*]]: i64, %[[VAL_61:.*]]: i64):
// CHECK:                   %[[VAL_62:.*]] = addi %[[VAL_60]], %[[VAL_61]] : i64
// CHECK:                   scf.reduce.return %[[VAL_62]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_34]] : memref<?xi1>
// CHECK:               scf.yield %[[VAL_63:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_64:.*]], %[[VAL_25]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_65:.*]] = %[[VAL_4]] to %[[VAL_9]] step %[[VAL_5]] {
// CHECK:             %[[VAL_66:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_65]]] : memref<?xi64>
// CHECK:             %[[VAL_67:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_67]], %[[VAL_25]]{{\[}}%[[VAL_65]]] : memref<?xi64>
// CHECK:             %[[VAL_68:.*]] = addi %[[VAL_67]], %[[VAL_66]] : i64
// CHECK:             memref.store %[[VAL_68]], %[[VAL_25]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_69:.*]] = sparse_tensor.pointers %[[VAL_15]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_70:.*]] = tensor.dim %[[VAL_15]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_71:.*]] = memref.load %[[VAL_69]]{{\[}}%[[VAL_70]]] : memref<?xi64>
// CHECK:           %[[VAL_72:.*]] = index_cast %[[VAL_71]] : i64 to index
// CHECK:           %[[VAL_73:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_73]], %[[VAL_5]], %[[VAL_72]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_74:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_74]], %[[VAL_72]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_75:.*]] = sparse_tensor.indices %[[VAL_15]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_76:.*]] = sparse_tensor.values %[[VAL_15]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_77:.*]]) = (%[[VAL_4]]) to (%[[VAL_9]]) step (%[[VAL_5]]) {
// CHECK:             %[[VAL_78:.*]] = addi %[[VAL_77]], %[[VAL_5]] : index
// CHECK:             %[[VAL_79:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:             %[[VAL_80:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:             %[[VAL_81:.*]] = cmpi ne, %[[VAL_79]], %[[VAL_80]] : i64
// CHECK:             scf.if %[[VAL_81]] {
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_83:.*]] = index_cast %[[VAL_82]] : i64 to index
// CHECK:               %[[VAL_84:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:               %[[VAL_86:.*]] = index_cast %[[VAL_84]] : i64 to index
// CHECK:               %[[VAL_87:.*]] = index_cast %[[VAL_85]] : i64 to index
// CHECK:               %[[VAL_88:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xf64>
// CHECK:               %[[VAL_89:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_7]], %[[VAL_89]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_90:.*]]) = (%[[VAL_86]]) to (%[[VAL_87]]) step (%[[VAL_5]]) {
// CHECK:                 %[[VAL_91:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:                 %[[VAL_92:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:                 memref.store %[[VAL_6]], %[[VAL_89]]{{\[}}%[[VAL_92]]] : memref<?xi1>
// CHECK:                 %[[VAL_93:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_90]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_93]], %[[VAL_88]]{{\[}}%[[VAL_92]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_94:.*]] = scf.for %[[VAL_95:.*]] = %[[VAL_4]] to %[[VAL_10]] step %[[VAL_5]] iter_args(%[[VAL_96:.*]] = %[[VAL_4]]) -> (index) {
// CHECK:                 %[[VAL_97:.*]] = index_cast %[[VAL_95]] : index to i64
// CHECK:                 %[[VAL_98:.*]] = addi %[[VAL_95]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_99:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_95]]] : memref<?xi64>
// CHECK:                 %[[VAL_100:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_98]]] : memref<?xi64>
// CHECK:                 %[[VAL_101:.*]] = index_cast %[[VAL_99]] : i64 to index
// CHECK:                 %[[VAL_102:.*]] = index_cast %[[VAL_100]] : i64 to index
// CHECK:                 %[[VAL_103:.*]]:2 = scf.for %[[VAL_104:.*]] = %[[VAL_101]] to %[[VAL_102]] step %[[VAL_5]] iter_args(%[[VAL_105:.*]] = %[[VAL_8]], %[[VAL_106:.*]] = %[[VAL_7]]) -> (f64, i1) {
// CHECK:                   %[[VAL_107:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_104]]] : memref<?xi64>
// CHECK:                   %[[VAL_108:.*]] = index_cast %[[VAL_107]] : i64 to index
// CHECK:                   %[[VAL_109:.*]] = memref.load %[[VAL_89]]{{\[}}%[[VAL_108]]] : memref<?xi1>
// CHECK:                   %[[VAL_110:.*]]:2 = scf.if %[[VAL_109]] -> (f64, i1) {
// CHECK:                     %[[VAL_111:.*]] = memref.load %[[VAL_88]]{{\[}}%[[VAL_108]]] : memref<?xf64>
// CHECK:                     %[[VAL_112:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_104]]] : memref<?xf64>
// CHECK:                     %[[VAL_113:.*]] = mulf %[[VAL_111]], %[[VAL_112]] : f64
// CHECK:                     %[[VAL_114:.*]] = addf %[[VAL_105]], %[[VAL_113]] : f64
// CHECK:                     scf.yield %[[VAL_114]], %[[VAL_6]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_105]], %[[VAL_106]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_115:.*]]#0, %[[VAL_115]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_116:.*]] = scf.if %[[VAL_117:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_118:.*]] = addi %[[VAL_83]], %[[VAL_96]] : index
// CHECK:                   memref.store %[[VAL_97]], %[[VAL_75]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_117]]#0, %[[VAL_76]]{{\[}}%[[VAL_118]]] : memref<?xf64>
// CHECK:                   %[[VAL_119:.*]] = addi %[[VAL_96]], %[[VAL_5]] : index
// CHECK:                   scf.yield %[[VAL_119]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_96]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_120:.*]] : index
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_88]] : memref<?xf64>
// CHECK:               memref.dealloc %[[VAL_89]] : memref<?xi1>
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_15]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_multiply_plus_times(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
