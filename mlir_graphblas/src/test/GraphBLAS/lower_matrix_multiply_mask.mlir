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


// CHECK-LABEL:   func @matrix_multiply_mask_plus_pair(
// CHECK-SAME:        %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:        %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:        %[[VAL_2:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-SAME:    ) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_3:.*]] = constant 0 : index
// CHECK:           %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_8:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = constant true
// CHECK:           %[[VAL_10:.*]] = constant false
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_16:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = memref.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_19:.*]] = addi %[[VAL_16]], %[[VAL_4]] : index
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.indices %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_22:.*]] = call @empty_like(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_dim(%[[VAL_22]], %[[VAL_3]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_dim(%[[VAL_22]], %[[VAL_4]], %[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_pointers(%[[VAL_22]], %[[VAL_4]], %[[VAL_19]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.pointers %[[VAL_22]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_24:.*]]) = (%[[VAL_3]]) to (%[[VAL_16]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_24]]] : memref<?xi64>
// CHECK:             %[[VAL_26:.*]] = addi %[[VAL_24]], %[[VAL_4]] : index
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:             %[[VAL_28:.*]] = cmpi eq, %[[VAL_25]], %[[VAL_27]] : i64
// CHECK:             %[[VAL_29:.*]] = scf.if %[[VAL_28]] -> (i64) {
// CHECK:               scf.yield %[[VAL_5]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_25]] : i64 to index
// CHECK:               %[[VAL_31:.*]] = index_cast %[[VAL_27]] : i64 to index
// CHECK:               %[[VAL_32:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_32]], %[[VAL_10]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_33:.*]]) = (%[[VAL_30]]) to (%[[VAL_31]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_33]]] : memref<?xi64>
// CHECK:                 %[[VAL_35:.*]] = index_cast %[[VAL_34]] : i64 to index
// CHECK:                 memref.store %[[VAL_9]], %[[VAL_32]]{{\[}}%[[VAL_35]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_36:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_24]]] : memref<?xi64>
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:               %[[VAL_38:.*]] = index_cast %[[VAL_36]] : i64 to index
// CHECK:               %[[VAL_39:.*]] = index_cast %[[VAL_37]] : i64 to index
// CHECK:               %[[VAL_40:.*]] = scf.parallel (%[[VAL_41:.*]]) = (%[[VAL_38]]) to (%[[VAL_39]]) step (%[[VAL_4]]) init (%[[VAL_5]]) -> i64 {
// CHECK:                 %[[VAL_42:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:                 %[[VAL_43:.*]] = index_cast %[[VAL_42]] : i64 to index
// CHECK:                 %[[VAL_44:.*]] = addi %[[VAL_43]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:                 %[[VAL_46:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:                 %[[VAL_47:.*]] = cmpi eq, %[[VAL_45]], %[[VAL_46]] : i64
// CHECK:                 %[[VAL_48:.*]] = scf.if %[[VAL_47]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_5]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_49:.*]] = scf.while (%[[VAL_50:.*]] = %[[VAL_45]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_51:.*]] = cmpi uge, %[[VAL_50]], %[[VAL_46]] : i64
// CHECK:                     %[[VAL_52:.*]]:2 = scf.if %[[VAL_51]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_10]], %[[VAL_5]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_53:.*]] = index_cast %[[VAL_50]] : i64 to index
// CHECK:                       %[[VAL_54:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_53]]] : memref<?xi64>
// CHECK:                       %[[VAL_55:.*]] = index_cast %[[VAL_54]] : i64 to index
// CHECK:                       %[[VAL_56:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_55]]] : memref<?xi1>
// CHECK:                       %[[VAL_57:.*]] = select %[[VAL_56]], %[[VAL_10]], %[[VAL_9]] : i1
// CHECK:                       %[[VAL_58:.*]] = select %[[VAL_56]], %[[VAL_6]], %[[VAL_50]] : i64
// CHECK:                       scf.yield %[[VAL_57]], %[[VAL_58]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_59:.*]]#0) %[[VAL_59]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_60:.*]]: i64):
// CHECK:                     %[[VAL_61:.*]] = addi %[[VAL_60]], %[[VAL_6]] : i64
// CHECK:                     scf.yield %[[VAL_61]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_62:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_63:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_64:.*]]: i64, %[[VAL_65:.*]]: i64):
// CHECK:                   %[[VAL_66:.*]] = addi %[[VAL_64]], %[[VAL_65]] : i64
// CHECK:                   scf.reduce.return %[[VAL_66]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_67:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_68:.*]], %[[VAL_23]]{{\[}}%[[VAL_24]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_69:.*]] = %[[VAL_3]] to %[[VAL_16]] step %[[VAL_4]] {
// CHECK:             %[[VAL_70:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:             %[[VAL_71:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_71]], %[[VAL_23]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:             %[[VAL_72:.*]] = addi %[[VAL_71]], %[[VAL_70]] : i64
// CHECK:             memref.store %[[VAL_72]], %[[VAL_23]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_73:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:           %[[VAL_74:.*]] = index_cast %[[VAL_73]] : i64 to index
// CHECK:           call @resize_index(%[[VAL_22]], %[[VAL_4]], %[[VAL_74]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_values(%[[VAL_22]], %[[VAL_74]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_75:.*]] = sparse_tensor.indices %[[VAL_22]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_76:.*]] = sparse_tensor.values %[[VAL_22]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_77:.*]]) = (%[[VAL_3]]) to (%[[VAL_16]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_78:.*]] = addi %[[VAL_77]], %[[VAL_4]] : index
// CHECK:             %[[VAL_79:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:             %[[VAL_80:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:             %[[VAL_81:.*]] = cmpi ne, %[[VAL_79]], %[[VAL_80]] : i64
// CHECK:             scf.if %[[VAL_81]] {
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_83:.*]] = index_cast %[[VAL_82]] : i64 to index
// CHECK:               %[[VAL_84:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:               %[[VAL_86:.*]] = index_cast %[[VAL_84]] : i64 to index
// CHECK:               %[[VAL_87:.*]] = index_cast %[[VAL_85]] : i64 to index
// CHECK:               %[[VAL_88:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xf64>
// CHECK:               %[[VAL_89:.*]] = memref.alloc(%[[VAL_18]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_89]], %[[VAL_10]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_90:.*]]) = (%[[VAL_86]]) to (%[[VAL_87]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_91:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:                 %[[VAL_92:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:                 memref.store %[[VAL_9]], %[[VAL_89]]{{\[}}%[[VAL_92]]] : memref<?xi1>
// CHECK:                 %[[VAL_93:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_90]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_93]], %[[VAL_88]]{{\[}}%[[VAL_92]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_78]]] : memref<?xi64>
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = index_cast %[[VAL_95]] : i64 to index
// CHECK:               %[[VAL_98:.*]] = scf.for %[[VAL_99:.*]] = %[[VAL_96]] to %[[VAL_97]] step %[[VAL_4]] iter_args(%[[VAL_100:.*]] = %[[VAL_3]]) -> (index) {
// CHECK:                 %[[VAL_101:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_99]]] : memref<?xi64>
// CHECK:                 %[[VAL_102:.*]] = index_cast %[[VAL_101]] : i64 to index
// CHECK:                 %[[VAL_103:.*]] = addi %[[VAL_102]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_104:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_102]]] : memref<?xi64>
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = index_cast %[[VAL_104]] : i64 to index
// CHECK:                 %[[VAL_107:.*]] = index_cast %[[VAL_105]] : i64 to index
// CHECK:                 %[[VAL_108:.*]]:2 = scf.for %[[VAL_109:.*]] = %[[VAL_106]] to %[[VAL_107]] step %[[VAL_4]] iter_args(%[[VAL_110:.*]] = %[[VAL_7]], %[[VAL_111:.*]] = %[[VAL_10]]) -> (f64, i1) {
// CHECK:                   %[[VAL_112:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                   %[[VAL_113:.*]] = index_cast %[[VAL_112]] : i64 to index
// CHECK:                   %[[VAL_114:.*]] = memref.load %[[VAL_89]]{{\[}}%[[VAL_113]]] : memref<?xi1>
// CHECK:                   %[[VAL_115:.*]]:2 = scf.if %[[VAL_114]] -> (f64, i1) {
// CHECK:                     %[[VAL_116:.*]] = addf %[[VAL_110]], %[[VAL_8]] : f64
// CHECK:                     scf.yield %[[VAL_116]], %[[VAL_9]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_110]], %[[VAL_111]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_117:.*]]#0, %[[VAL_117]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_118:.*]] = scf.if %[[VAL_119:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_120:.*]] = addi %[[VAL_83]], %[[VAL_100]] : index
// CHECK:                   memref.store %[[VAL_101]], %[[VAL_75]]{{\[}}%[[VAL_120]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_119]]#0, %[[VAL_76]]{{\[}}%[[VAL_120]]] : memref<?xf64>
// CHECK:                   %[[VAL_121:.*]] = addi %[[VAL_100]], %[[VAL_4]] : index
// CHECK:                   scf.yield %[[VAL_121]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_100]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_122:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_22]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }


func @matrix_multiply_mask_plus_pair(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>, %m: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b, %m { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations