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
// CHECK-SAME:        %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:        %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK-SAME:    ) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = constant 1 : index
// CHECK:           %[[VAL_4:.*]] = constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = constant 0.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = constant true
// CHECK:           %[[VAL_8:.*]] = constant false
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = memref.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = memref.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_3]] : index
// CHECK:           %[[VAL_19:.*]] = call @empty_like(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           call @resize_dim(%[[VAL_19]], %[[VAL_2]], %[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_dim(%[[VAL_19]], %[[VAL_3]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_pointers(%[[VAL_19]], %[[VAL_3]], %[[VAL_18]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.pointers %[[VAL_19]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_21:.*]]) = (%[[VAL_2]]) to (%[[VAL_15]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:             %[[VAL_23:.*]] = addi %[[VAL_21]], %[[VAL_3]] : index
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_23]]] : memref<?xi64>
// CHECK:             %[[VAL_25:.*]] = cmpi eq, %[[VAL_22]], %[[VAL_24]] : i64
// CHECK:             %[[VAL_26:.*]] = scf.if %[[VAL_25]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_27:.*]] = index_cast %[[VAL_22]] : i64 to index
// CHECK:               %[[VAL_28:.*]] = index_cast %[[VAL_24]] : i64 to index
// CHECK:               %[[VAL_29:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_29]], %[[VAL_8]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_30:.*]]) = (%[[VAL_27]]) to (%[[VAL_28]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_31:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_30]]] : memref<?xi64>
// CHECK:                 %[[VAL_32:.*]] = index_cast %[[VAL_31]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_29]]{{\[}}%[[VAL_32]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_33:.*]] = scf.parallel (%[[VAL_34:.*]]) = (%[[VAL_2]]) to (%[[VAL_16]]) step (%[[VAL_3]]) init (%[[VAL_4]]) -> i64 {
// CHECK:                 %[[VAL_35:.*]] = addi %[[VAL_34]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:                 %[[VAL_37:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_35]]] : memref<?xi64>
// CHECK:                 %[[VAL_38:.*]] = cmpi eq, %[[VAL_36]], %[[VAL_37]] : i64
// CHECK:                 %[[VAL_39:.*]] = scf.if %[[VAL_38]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_4]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_40:.*]] = scf.while (%[[VAL_41:.*]] = %[[VAL_36]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_42:.*]] = cmpi uge, %[[VAL_41]], %[[VAL_37]] : i64
// CHECK:                     %[[VAL_43:.*]]:2 = scf.if %[[VAL_42]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_8]], %[[VAL_4]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_44:.*]] = index_cast %[[VAL_41]] : i64 to index
// CHECK:                       %[[VAL_45:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:                       %[[VAL_46:.*]] = index_cast %[[VAL_45]] : i64 to index
// CHECK:                       %[[VAL_47:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_46]]] : memref<?xi1>
// CHECK:                       %[[VAL_48:.*]] = select %[[VAL_47]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                       %[[VAL_49:.*]] = select %[[VAL_47]], %[[VAL_5]], %[[VAL_41]] : i64
// CHECK:                       scf.yield %[[VAL_48]], %[[VAL_49]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_50:.*]]#0) %[[VAL_50]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_51:.*]]: i64):
// CHECK:                     %[[VAL_52:.*]] = addi %[[VAL_51]], %[[VAL_5]] : i64
// CHECK:                     scf.yield %[[VAL_52]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_53:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_54:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_55:.*]]: i64, %[[VAL_56:.*]]: i64):
// CHECK:                   %[[VAL_57:.*]] = addi %[[VAL_55]], %[[VAL_56]] : i64
// CHECK:                   scf.reduce.return %[[VAL_57]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               scf.yield %[[VAL_58:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_59:.*]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_60:.*]] = %[[VAL_2]] to %[[VAL_15]] step %[[VAL_3]] {
// CHECK:             %[[VAL_61:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_60]]] : memref<?xi64>
// CHECK:             %[[VAL_62:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_62]], %[[VAL_20]]{{\[}}%[[VAL_60]]] : memref<?xi64>
// CHECK:             %[[VAL_63:.*]] = addi %[[VAL_62]], %[[VAL_61]] : i64
// CHECK:             memref.store %[[VAL_63]], %[[VAL_20]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_64:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:           %[[VAL_65:.*]] = index_cast %[[VAL_64]] : i64 to index
// CHECK:           call @resize_index(%[[VAL_19]], %[[VAL_3]], %[[VAL_65]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:           call @resize_values(%[[VAL_19]], %[[VAL_65]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:           %[[VAL_66:.*]] = sparse_tensor.indices %[[VAL_19]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_67:.*]] = sparse_tensor.values %[[VAL_19]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_68:.*]]) = (%[[VAL_2]]) to (%[[VAL_15]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_69:.*]] = addi %[[VAL_68]], %[[VAL_3]] : index
// CHECK:             %[[VAL_70:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:             %[[VAL_71:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:             %[[VAL_72:.*]] = cmpi ne, %[[VAL_70]], %[[VAL_71]] : i64
// CHECK:             scf.if %[[VAL_72]] {
// CHECK:               %[[VAL_73:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:               %[[VAL_74:.*]] = index_cast %[[VAL_73]] : i64 to index
// CHECK:               %[[VAL_75:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:               %[[VAL_76:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_69]]] : memref<?xi64>
// CHECK:               %[[VAL_77:.*]] = index_cast %[[VAL_75]] : i64 to index
// CHECK:               %[[VAL_78:.*]] = index_cast %[[VAL_76]] : i64 to index
// CHECK:               %[[VAL_79:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xf64>
// CHECK:               %[[VAL_80:.*]] = memref.alloc(%[[VAL_17]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_80]], %[[VAL_8]]) : memref<?xi1>, i1
// CHECK:               scf.parallel (%[[VAL_81:.*]]) = (%[[VAL_77]]) to (%[[VAL_78]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_82:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:                 %[[VAL_83:.*]] = index_cast %[[VAL_82]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_80]]{{\[}}%[[VAL_83]]] : memref<?xi1>
// CHECK:                 %[[VAL_84:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_81]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_84]], %[[VAL_79]]{{\[}}%[[VAL_83]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_85:.*]] = scf.for %[[VAL_86:.*]] = %[[VAL_2]] to %[[VAL_16]] step %[[VAL_3]] iter_args(%[[VAL_87:.*]] = %[[VAL_2]]) -> (index) {
// CHECK:                 %[[VAL_88:.*]] = index_cast %[[VAL_86]] : index to i64
// CHECK:                 %[[VAL_89:.*]] = addi %[[VAL_86]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_90:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:                 %[[VAL_91:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:                 %[[VAL_92:.*]] = index_cast %[[VAL_90]] : i64 to index
// CHECK:                 %[[VAL_93:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:                 %[[VAL_94:.*]]:2 = scf.for %[[VAL_95:.*]] = %[[VAL_92]] to %[[VAL_93]] step %[[VAL_3]] iter_args(%[[VAL_96:.*]] = %[[VAL_6]], %[[VAL_97:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                   %[[VAL_98:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_95]]] : memref<?xi64>
// CHECK:                   %[[VAL_99:.*]] = index_cast %[[VAL_98]] : i64 to index
// CHECK:                   %[[VAL_100:.*]] = memref.load %[[VAL_80]]{{\[}}%[[VAL_99]]] : memref<?xi1>
// CHECK:                   %[[VAL_101:.*]]:2 = scf.if %[[VAL_100]] -> (f64, i1) {
// CHECK:                     %[[VAL_102:.*]] = memref.load %[[VAL_79]]{{\[}}%[[VAL_99]]] : memref<?xf64>
// CHECK:                     %[[VAL_103:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_95]]] : memref<?xf64>
// CHECK:                     %[[VAL_104:.*]] = mulf %[[VAL_102]], %[[VAL_103]] : f64
// CHECK:                     %[[VAL_105:.*]] = addf %[[VAL_96]], %[[VAL_104]] : f64
// CHECK:                     scf.yield %[[VAL_105]], %[[VAL_7]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_96]], %[[VAL_97]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_106:.*]]#0, %[[VAL_106]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_107:.*]] = scf.if %[[VAL_108:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_109:.*]] = addi %[[VAL_74]], %[[VAL_87]] : index
// CHECK:                   memref.store %[[VAL_88]], %[[VAL_66]]{{\[}}%[[VAL_109]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_108]]#0, %[[VAL_67]]{{\[}}%[[VAL_109]]] : memref<?xf64>
// CHECK:                   %[[VAL_110:.*]] = addi %[[VAL_87]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_110]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_87]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_111:.*]] : index
// CHECK:               }
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_19]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }


func @matrix_multiply_plus_times(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
