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
// CHECK-SAME:                                         %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK-DAG:       %[[VAL_3:.*]] = constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = constant 1 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = constant true
// CHECK-DAG:       %[[VAL_8:.*]] = constant false
// CHECK-DAG:       %[[VAL_9:.*]] = constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_10:.*]] = constant 1.000000e+00 : f64
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_14:.*]] = addi %[[VAL_11]], %[[VAL_6]] : index
// CHECK:           %[[VAL_15:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_0]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = call @empty_like(%[[VAL_15]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = call @ptr8_to_matrix_csr_f64_p64i64(%[[VAL_16]]) : (!llvm.ptr<i8>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_18]], %[[VAL_5]], %[[VAL_11]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_19:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_19]], %[[VAL_6]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_20]], %[[VAL_6]], %[[VAL_14]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_24:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.pointers %[[VAL_17]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.pointers %[[VAL_2]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_28:.*]] = sparse_tensor.indices %[[VAL_2]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_29:.*]]) = (%[[VAL_5]]) to (%[[VAL_11]]) step (%[[VAL_6]]) {
// CHECK:             %[[VAL_30:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             %[[VAL_31:.*]] = addi %[[VAL_29]], %[[VAL_6]] : index
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:             %[[VAL_33:.*]] = cmpi eq, %[[VAL_30]], %[[VAL_32]] : i64
// CHECK:             %[[VAL_34:.*]] = scf.if %[[VAL_33]] -> (i64) {
// CHECK:               scf.yield %[[VAL_3]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_35:.*]] = index_cast %[[VAL_30]] : i64 to index
// CHECK:               %[[VAL_36:.*]] = index_cast %[[VAL_32]] : i64 to index
// CHECK:               %[[VAL_37:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:               %[[VAL_38:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_31]]] : memref<?xi64>
// CHECK:               %[[VAL_39:.*]] = index_cast %[[VAL_37]] : i64 to index
// CHECK:               %[[VAL_40:.*]] = index_cast %[[VAL_38]] : i64 to index
// CHECK:               %[[VAL_41:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_41]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_42:.*]]) = (%[[VAL_35]]) to (%[[VAL_36]]) step (%[[VAL_6]]) {
// CHECK:                 %[[VAL_43:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:                 %[[VAL_44:.*]] = index_cast %[[VAL_43]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_41]]{{\[}}%[[VAL_44]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_45:.*]] = scf.parallel (%[[VAL_46:.*]]) = (%[[VAL_39]]) to (%[[VAL_40]]) step (%[[VAL_6]]) init (%[[VAL_3]]) -> i64 {
// CHECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_46]]] : memref<?xi64>
// CHECK:                 %[[VAL_48:.*]] = index_cast %[[VAL_47]] : i64 to index
// CHECK:                 %[[VAL_49:.*]] = addi %[[VAL_48]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_50:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_48]]] : memref<?xi64>
// CHECK:                 %[[VAL_51:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:                 %[[VAL_52:.*]] = cmpi eq, %[[VAL_50]], %[[VAL_51]] : i64
// CHECK:                 %[[VAL_53:.*]] = scf.if %[[VAL_52]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_3]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_54:.*]] = scf.while (%[[VAL_55:.*]] = %[[VAL_50]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_56:.*]] = cmpi uge, %[[VAL_55]], %[[VAL_51]] : i64
// CHECK:                     %[[VAL_57:.*]]:2 = scf.if %[[VAL_56]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_8]], %[[VAL_3]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_58:.*]] = index_cast %[[VAL_55]] : i64 to index
// CHECK:                       %[[VAL_59:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_58]]] : memref<?xi64>
// CHECK:                       %[[VAL_60:.*]] = index_cast %[[VAL_59]] : i64 to index
// CHECK:                       %[[VAL_61:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_60]]] : memref<?xi1>
// CHECK:                       %[[VAL_62:.*]] = select %[[VAL_61]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                       %[[VAL_63:.*]] = select %[[VAL_61]], %[[VAL_4]], %[[VAL_55]] : i64
// CHECK:                       scf.yield %[[VAL_62]], %[[VAL_63]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_64:.*]]#0) %[[VAL_64]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_65:.*]]: i64):
// CHECK:                     %[[VAL_66:.*]] = addi %[[VAL_65]], %[[VAL_4]] : i64
// CHECK:                     scf.yield %[[VAL_66]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_67:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_68:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_69:.*]]: i64, %[[VAL_70:.*]]: i64):
// CHECK:                   %[[VAL_71:.*]] = addi %[[VAL_69]], %[[VAL_70]] : i64
// CHECK:                   scf.reduce.return %[[VAL_71]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_41]] : memref<?xi1>
// CHECK:               scf.yield %[[VAL_72:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_73:.*]], %[[VAL_26]]{{\[}}%[[VAL_29]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_74:.*]] = %[[VAL_5]] to %[[VAL_11]] step %[[VAL_6]] {
// CHECK:             %[[VAL_75:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_74]]] : memref<?xi64>
// CHECK:             %[[VAL_76:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_76]], %[[VAL_26]]{{\[}}%[[VAL_74]]] : memref<?xi64>
// CHECK:             %[[VAL_77:.*]] = addi %[[VAL_76]], %[[VAL_75]] : i64
// CHECK:             memref.store %[[VAL_77]], %[[VAL_26]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_78:.*]] = sparse_tensor.pointers %[[VAL_17]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_79:.*]] = tensor.dim %[[VAL_17]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_80:.*]] = memref.load %[[VAL_78]]{{\[}}%[[VAL_79]]] : memref<?xi64>
// CHECK:           %[[VAL_81:.*]] = index_cast %[[VAL_80]] : i64 to index
// CHECK:           %[[VAL_82:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_82]], %[[VAL_6]], %[[VAL_81]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_83:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_83]], %[[VAL_81]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_17]], %[[VAL_6]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_17]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_86:.*]]) = (%[[VAL_5]]) to (%[[VAL_11]]) step (%[[VAL_6]]) {
// CHECK:             %[[VAL_87:.*]] = addi %[[VAL_86]], %[[VAL_6]] : index
// CHECK:             %[[VAL_88:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:             %[[VAL_89:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:             %[[VAL_90:.*]] = cmpi ne, %[[VAL_88]], %[[VAL_89]] : i64
// CHECK:             scf.if %[[VAL_90]] {
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_92:.*]] = index_cast %[[VAL_91]] : i64 to index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_21]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_86]]] : memref<?xi64>
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_87]]] : memref<?xi64>
// CHECK:               %[[VAL_99:.*]] = index_cast %[[VAL_97]] : i64 to index
// CHECK:               %[[VAL_100:.*]] = index_cast %[[VAL_98]] : i64 to index
// CHECK:               %[[VAL_101:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xf64>
// CHECK:               %[[VAL_102:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_102]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_103:.*]]) = (%[[VAL_95]]) to (%[[VAL_96]]) step (%[[VAL_6]]) {
// CHECK:                 %[[VAL_104:.*]] = memref.load %[[VAL_22]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_105:.*]] = index_cast %[[VAL_104]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_102]]{{\[}}%[[VAL_105]]] : memref<?xi1>
// CHECK:                 %[[VAL_106:.*]] = memref.load %[[VAL_23]]{{\[}}%[[VAL_103]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_106]], %[[VAL_101]]{{\[}}%[[VAL_105]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_107:.*]] = scf.for %[[VAL_108:.*]] = %[[VAL_99]] to %[[VAL_100]] step %[[VAL_6]] iter_args(%[[VAL_109:.*]] = %[[VAL_5]]) -> (index) {
// CHECK:                 %[[VAL_110:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_108]]] : memref<?xi64>
// CHECK:                 %[[VAL_111:.*]] = index_cast %[[VAL_110]] : i64 to index
// CHECK:                 %[[VAL_112:.*]] = addi %[[VAL_111]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_113:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_111]]] : memref<?xi64>
// CHECK:                 %[[VAL_114:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_112]]] : memref<?xi64>
// CHECK:                 %[[VAL_115:.*]] = index_cast %[[VAL_113]] : i64 to index
// CHECK:                 %[[VAL_116:.*]] = index_cast %[[VAL_114]] : i64 to index
// CHECK:                 %[[VAL_117:.*]]:2 = scf.for %[[VAL_118:.*]] = %[[VAL_115]] to %[[VAL_116]] step %[[VAL_6]] iter_args(%[[VAL_119:.*]] = %[[VAL_9]], %[[VAL_120:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                   %[[VAL_121:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                   %[[VAL_122:.*]] = index_cast %[[VAL_121]] : i64 to index
// CHECK:                   %[[VAL_123:.*]] = memref.load %[[VAL_102]]{{\[}}%[[VAL_122]]] : memref<?xi1>
// CHECK:                   %[[VAL_124:.*]]:2 = scf.if %[[VAL_123]] -> (f64, i1) {
// CHECK:                     %[[VAL_125:.*]] = addf %[[VAL_119]], %[[VAL_10]] : f64
// CHECK:                     scf.yield %[[VAL_125]], %[[VAL_7]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_119]], %[[VAL_120]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_126:.*]]#0, %[[VAL_126]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_127:.*]] = scf.if %[[VAL_128:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_129:.*]] = addi %[[VAL_92]], %[[VAL_109]] : index
// CHECK:                   memref.store %[[VAL_110]], %[[VAL_84]]{{\[}}%[[VAL_129]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_128]]#0, %[[VAL_85]]{{\[}}%[[VAL_129]]] : memref<?xf64>
// CHECK:                   %[[VAL_130:.*]] = addi %[[VAL_109]], %[[VAL_6]] : index
// CHECK:                   scf.yield %[[VAL_130]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_109]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_131:.*]] : index
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_101]] : memref<?xf64>
// CHECK:               memref.dealloc %[[VAL_102]] : memref<?xi1>
// CHECK:             }
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return %[[VAL_17]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:         }

func @matrix_multiply_mask_plus_pair(%a: tensor<?x?xf64, #CSR64>, %b: tensor<?x?xf64, #CSC64>, %m: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSR64> {
    %answer = graphblas.matrix_multiply %a, %b, %m { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
    return %answer : tensor<?x?xf64, #CSR64>
}

// TODO: Check all type combinations
