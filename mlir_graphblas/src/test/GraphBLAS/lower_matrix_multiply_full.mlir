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
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = arith.constant false
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_9]], %[[VAL_3]] : index
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.init{{\[}}%[[VAL_13]], %[[VAL_14]]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_15]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_3]] : index
// CHECK:           %[[VAL_18:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_18]], %[[VAL_3]], %[[VAL_17]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_19:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_19]], %[[VAL_2]], %[[VAL_9]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_20]], %[[VAL_3]], %[[VAL_10]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_21]], %[[VAL_3]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_24:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_28:.*]] = sparse_tensor.pointers %[[VAL_15]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_29:.*]]) = (%[[VAL_2]]) to (%[[VAL_9]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_29]], %[[VAL_3]] : index
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = arith.cmpi eq, %[[VAL_30]], %[[VAL_32]] : i64
// CHECK:             %[[VAL_34:.*]] = scf.if %[[VAL_33]] -> (i64) {
// CHECK:               scf.yield %[[VAL_4]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_35:.*]] = arith.index_cast %[[VAL_30]] : i64 to index
// CHECK:               %[[VAL_36:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:               %[[VAL_37:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_7]], %[[VAL_37]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_38:.*]]) = (%[[VAL_35]]) to (%[[VAL_36]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_39:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_38]]] : memref<?xi64>
// CHECK:                 %[[VAL_40:.*]] = arith.index_cast %[[VAL_39]] : i64 to index
// CHECK:                 memref.store %[[VAL_6]], %[[VAL_37]]{{\[}}%[[VAL_40]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_41:.*]] = scf.parallel (%[[VAL_42:.*]]) = (%[[VAL_2]]) to (%[[VAL_10]]) step (%[[VAL_3]]) init (%[[VAL_4]]) -> i64 {
// CHECK:                 %[[VAL_43:.*]] = arith.addi %[[VAL_42]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_44:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:                 %[[VAL_46:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_45]] : i64
// CHECK:                 %[[VAL_47:.*]] = scf.if %[[VAL_46]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_4]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_48:.*]] = scf.while (%[[VAL_49:.*]] = %[[VAL_44]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_50:.*]] = arith.cmpi uge, %[[VAL_49]], %[[VAL_45]] : i64
// CHECK:                     %[[VAL_51:.*]]:2 = scf.if %[[VAL_50]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_7]], %[[VAL_4]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_52:.*]] = arith.index_cast %[[VAL_49]] : i64 to index
// CHECK:                       %[[VAL_53:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:                       %[[VAL_54:.*]] = arith.index_cast %[[VAL_53]] : i64 to index
// CHECK:                       %[[VAL_55:.*]] = memref.load %[[VAL_37]]{{\[}}%[[VAL_54]]] : memref<?xi1>
// CHECK:                       %[[VAL_56:.*]] = arith.select %[[VAL_55]], %[[VAL_7]], %[[VAL_6]] : i1
// CHECK:                       %[[VAL_57:.*]] = arith.select %[[VAL_55]], %[[VAL_5]], %[[VAL_49]] : i64
// CHECK:                       scf.yield %[[VAL_56]], %[[VAL_57]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_58:.*]]#0) %[[VAL_58]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_59:.*]]: i64):
// CHECK:                     %[[VAL_60:.*]] = arith.addi %[[VAL_59]], %[[VAL_5]] : i64
// CHECK:                     scf.yield %[[VAL_60]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_61:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_62:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_63:.*]]: i64, %[[VAL_64:.*]]: i64):
// CHECK:                   %[[VAL_65:.*]] = arith.addi %[[VAL_63]], %[[VAL_64]] : i64
// CHECK:                   scf.reduce.return %[[VAL_65]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_37]] : memref<?xi1>
// CHECK:               scf.yield %[[VAL_66:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_67:.*]], %[[VAL_28]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_68:.*]] = %[[VAL_2]] to %[[VAL_9]] step %[[VAL_3]] {
// CHECK:             %[[VAL_69:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:             %[[VAL_70:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_70]], %[[VAL_28]]{{\[}}%[[VAL_68]]] : memref<?xi64>
// CHECK:             %[[VAL_71:.*]] = arith.addi %[[VAL_70]], %[[VAL_69]] : i64
// CHECK:             memref.store %[[VAL_71]], %[[VAL_28]]{{\[}}%[[VAL_9]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = sparse_tensor.pointers %[[VAL_15]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_73:.*]] = tensor.dim %[[VAL_15]], %[[VAL_2]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_74:.*]] = memref.load %[[VAL_72]]{{\[}}%[[VAL_73]]] : memref<?xi64>
// CHECK:           %[[VAL_75:.*]] = arith.index_cast %[[VAL_74]] : i64 to index
// CHECK:           %[[VAL_76:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_76]], %[[VAL_3]], %[[VAL_75]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_77:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_15]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_77]], %[[VAL_75]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_78:.*]] = sparse_tensor.indices %[[VAL_15]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_79:.*]] = sparse_tensor.values %[[VAL_15]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_80:.*]]) = (%[[VAL_2]]) to (%[[VAL_9]]) step (%[[VAL_3]]) {
// CHECK:             %[[VAL_81:.*]] = arith.addi %[[VAL_80]], %[[VAL_3]] : index
// CHECK:             %[[VAL_82:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:             %[[VAL_83:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = arith.cmpi ne, %[[VAL_82]], %[[VAL_83]] : i64
// CHECK:             scf.if %[[VAL_84]] {
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_86:.*]] = arith.index_cast %[[VAL_85]] : i64 to index
// CHECK:               %[[VAL_87:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_80]]] : memref<?xi64>
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:               %[[VAL_89:.*]] = arith.index_cast %[[VAL_87]] : i64 to index
// CHECK:               %[[VAL_90:.*]] = arith.index_cast %[[VAL_88]] : i64 to index
// CHECK:               %[[VAL_91:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xf64>
// CHECK:               %[[VAL_92:.*]] = memref.alloc(%[[VAL_11]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_7]], %[[VAL_92]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_93:.*]]) = (%[[VAL_89]]) to (%[[VAL_90]]) step (%[[VAL_3]]) {
// CHECK:                 %[[VAL_94:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_93]]] : memref<?xi64>
// CHECK:                 %[[VAL_95:.*]] = arith.index_cast %[[VAL_94]] : i64 to index
// CHECK:                 memref.store %[[VAL_6]], %[[VAL_92]]{{\[}}%[[VAL_95]]] : memref<?xi1>
// CHECK:                 %[[VAL_96:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_93]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_96]], %[[VAL_91]]{{\[}}%[[VAL_95]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_97:.*]] = scf.for %[[VAL_98:.*]] = %[[VAL_2]] to %[[VAL_10]] step %[[VAL_3]] iter_args(%[[VAL_99:.*]] = %[[VAL_2]]) -> (index) {
// CHECK:                 %[[VAL_100:.*]] = arith.index_cast %[[VAL_98]] : index to i64
// CHECK:                 %[[VAL_101:.*]] = arith.addi %[[VAL_98]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_102:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_98]]] : memref<?xi64>
// CHECK:                 %[[VAL_103:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_101]]] : memref<?xi64>
// CHECK:                 %[[VAL_104:.*]] = arith.index_cast %[[VAL_102]] : i64 to index
// CHECK:                 %[[VAL_105:.*]] = arith.index_cast %[[VAL_103]] : i64 to index
// CHECK:                 %[[VAL_106:.*]]:2 = scf.for %[[VAL_107:.*]] = %[[VAL_104]] to %[[VAL_105]] step %[[VAL_3]] iter_args(%[[VAL_108:.*]] = %[[VAL_8]], %[[VAL_109:.*]] = %[[VAL_7]]) -> (f64, i1) {
// CHECK:                   %[[VAL_110:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_107]]] : memref<?xi64>
// CHECK:                   %[[VAL_111:.*]] = arith.index_cast %[[VAL_110]] : i64 to index
// CHECK:                   %[[VAL_112:.*]] = memref.load %[[VAL_92]]{{\[}}%[[VAL_111]]] : memref<?xi1>
// CHECK:                   %[[VAL_113:.*]]:2 = scf.if %[[VAL_112]] -> (f64, i1) {
// CHECK:                     %[[VAL_114:.*]] = memref.load %[[VAL_91]]{{\[}}%[[VAL_111]]] : memref<?xf64>
// CHECK:                     %[[VAL_115:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_107]]] : memref<?xf64>
// CHECK:                     %[[VAL_116:.*]] = arith.mulf %[[VAL_114]], %[[VAL_115]] : f64
// CHECK:                     %[[VAL_117:.*]] = arith.addf %[[VAL_108]], %[[VAL_116]] : f64
// CHECK:                     scf.yield %[[VAL_117]], %[[VAL_6]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_108]], %[[VAL_109]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_118:.*]]#0, %[[VAL_118]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_119:.*]] = scf.if %[[VAL_120:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_121:.*]] = arith.addi %[[VAL_86]], %[[VAL_99]] : index
// CHECK:                   memref.store %[[VAL_100]], %[[VAL_78]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_120]]#0, %[[VAL_79]]{{\[}}%[[VAL_121]]] : memref<?xf64>
// CHECK:                   %[[VAL_122:.*]] = arith.addi %[[VAL_99]], %[[VAL_3]] : index
// CHECK:                   scf.yield %[[VAL_122]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_99]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_123:.*]] : index
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_91]] : memref<?xf64>
// CHECK:               memref.dealloc %[[VAL_92]] : memref<?xi1>
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
