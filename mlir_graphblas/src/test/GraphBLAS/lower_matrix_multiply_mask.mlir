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
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant true
// CHECK:           %[[VAL_8:.*]] = arith.constant false
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_10:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : index
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_17:.*]] = sparse_tensor.init{{\[}}%[[VAL_15]], %[[VAL_16]]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_17]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_4]] : index
// CHECK:           %[[VAL_20:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_20]], %[[VAL_4]], %[[VAL_19]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_21:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_21]], %[[VAL_3]], %[[VAL_11]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_22:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_dim(%[[VAL_22]], %[[VAL_4]], %[[VAL_12]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_23:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_pointers(%[[VAL_23]], %[[VAL_4]], %[[VAL_14]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_24:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           %[[VAL_27:.*]] = sparse_tensor.pointers %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_28:.*]] = sparse_tensor.indices %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_29:.*]] = sparse_tensor.pointers %[[VAL_17]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_30:.*]] = sparse_tensor.pointers %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_31:.*]] = sparse_tensor.indices %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           scf.parallel (%[[VAL_32:.*]]) = (%[[VAL_3]]) to (%[[VAL_11]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_33:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_32]], %[[VAL_4]] : index
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:             %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_33]], %[[VAL_35]] : i64
// CHECK:             %[[VAL_37:.*]] = scf.if %[[VAL_36]] -> (i64) {
// CHECK:               scf.yield %[[VAL_5]] : i64
// CHECK:             } else {
// CHECK:               %[[VAL_38:.*]] = arith.index_cast %[[VAL_33]] : i64 to index
// CHECK:               %[[VAL_39:.*]] = arith.index_cast %[[VAL_35]] : i64 to index
// CHECK:               %[[VAL_40:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:               %[[VAL_42:.*]] = arith.index_cast %[[VAL_40]] : i64 to index
// CHECK:               %[[VAL_43:.*]] = arith.index_cast %[[VAL_41]] : i64 to index
// CHECK:               %[[VAL_44:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_44]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_45:.*]]) = (%[[VAL_38]]) to (%[[VAL_39]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_46:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_45]]] : memref<?xi64>
// CHECK:                 %[[VAL_47:.*]] = arith.index_cast %[[VAL_46]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_44]]{{\[}}%[[VAL_47]]] : memref<?xi1>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_48:.*]] = scf.parallel (%[[VAL_49:.*]]) = (%[[VAL_42]]) to (%[[VAL_43]]) step (%[[VAL_4]]) init (%[[VAL_5]]) -> i64 {
// CHECK:                 %[[VAL_50:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_49]]] : memref<?xi64>
// CHECK:                 %[[VAL_51:.*]] = arith.index_cast %[[VAL_50]] : i64 to index
// CHECK:                 %[[VAL_52:.*]] = arith.addi %[[VAL_51]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_53:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_51]]] : memref<?xi64>
// CHECK:                 %[[VAL_54:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_52]]] : memref<?xi64>
// CHECK:                 %[[VAL_55:.*]] = arith.cmpi eq, %[[VAL_53]], %[[VAL_54]] : i64
// CHECK:                 %[[VAL_56:.*]] = scf.if %[[VAL_55]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_5]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_57:.*]] = scf.while (%[[VAL_58:.*]] = %[[VAL_53]]) : (i64) -> i64 {
// CHECK:                     %[[VAL_59:.*]] = arith.cmpi uge, %[[VAL_58]], %[[VAL_54]] : i64
// CHECK:                     %[[VAL_60:.*]]:2 = scf.if %[[VAL_59]] -> (i1, i64) {
// CHECK:                       scf.yield %[[VAL_8]], %[[VAL_5]] : i1, i64
// CHECK:                     } else {
// CHECK:                       %[[VAL_61:.*]] = arith.index_cast %[[VAL_58]] : i64 to index
// CHECK:                       %[[VAL_62:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_61]]] : memref<?xi64>
// CHECK:                       %[[VAL_63:.*]] = arith.index_cast %[[VAL_62]] : i64 to index
// CHECK:                       %[[VAL_64:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_63]]] : memref<?xi1>
// CHECK:                       %[[VAL_65:.*]] = select %[[VAL_64]], %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:                       %[[VAL_66:.*]] = select %[[VAL_64]], %[[VAL_6]], %[[VAL_58]] : i64
// CHECK:                       scf.yield %[[VAL_65]], %[[VAL_66]] : i1, i64
// CHECK:                     }
// CHECK:                     scf.condition(%[[VAL_67:.*]]#0) %[[VAL_67]]#1 : i64
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_68:.*]]: i64):
// CHECK:                     %[[VAL_69:.*]] = arith.addi %[[VAL_68]], %[[VAL_6]] : i64
// CHECK:                     scf.yield %[[VAL_69]] : i64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_70:.*]] : i64
// CHECK:                 }
// CHECK:                 scf.reduce(%[[VAL_71:.*]])  : i64 {
// CHECK:                 ^bb0(%[[VAL_72:.*]]: i64, %[[VAL_73:.*]]: i64):
// CHECK:                   %[[VAL_74:.*]] = arith.addi %[[VAL_72]], %[[VAL_73]] : i64
// CHECK:                   scf.reduce.return %[[VAL_74]] : i64
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_44]] : memref<?xi1>
// CHECK:               scf.yield %[[VAL_75:.*]] : i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_76:.*]], %[[VAL_29]]{{\[}}%[[VAL_32]]] : memref<?xi64>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           scf.for %[[VAL_77:.*]] = %[[VAL_3]] to %[[VAL_11]] step %[[VAL_4]] {
// CHECK:             %[[VAL_78:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:             %[[VAL_79:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:             memref.store %[[VAL_79]], %[[VAL_29]]{{\[}}%[[VAL_77]]] : memref<?xi64>
// CHECK:             %[[VAL_80:.*]] = arith.addi %[[VAL_79]], %[[VAL_78]] : i64
// CHECK:             memref.store %[[VAL_80]], %[[VAL_29]]{{\[}}%[[VAL_11]]] : memref<?xi64>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = sparse_tensor.pointers %[[VAL_17]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_82:.*]] = tensor.dim %[[VAL_17]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           %[[VAL_83:.*]] = memref.load %[[VAL_81]]{{\[}}%[[VAL_82]]] : memref<?xi64>
// CHECK:           %[[VAL_84:.*]] = arith.index_cast %[[VAL_83]] : i64 to index
// CHECK:           %[[VAL_85:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_index(%[[VAL_85]], %[[VAL_4]], %[[VAL_84]]) : (!llvm.ptr<i8>, index, index) -> ()
// CHECK:           %[[VAL_86:.*]] = call @matrix_csr_f64_p64i64_to_ptr8(%[[VAL_17]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> !llvm.ptr<i8>
// CHECK:           call @resize_values(%[[VAL_86]], %[[VAL_84]]) : (!llvm.ptr<i8>, index) -> ()
// CHECK:           %[[VAL_87:.*]] = sparse_tensor.indices %[[VAL_17]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:           %[[VAL_88:.*]] = sparse_tensor.values %[[VAL_17]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:           scf.parallel (%[[VAL_89:.*]]) = (%[[VAL_3]]) to (%[[VAL_11]]) step (%[[VAL_4]]) {
// CHECK:             %[[VAL_90:.*]] = arith.addi %[[VAL_89]], %[[VAL_4]] : index
// CHECK:             %[[VAL_91:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:             %[[VAL_92:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:             %[[VAL_93:.*]] = arith.cmpi ne, %[[VAL_91]], %[[VAL_92]] : i64
// CHECK:             scf.if %[[VAL_93]] {
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_29]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = arith.index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_98:.*]] = arith.index_cast %[[VAL_96]] : i64 to index
// CHECK:               %[[VAL_99:.*]] = arith.index_cast %[[VAL_97]] : i64 to index
// CHECK:               %[[VAL_100:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:               %[[VAL_101:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_102:.*]] = arith.index_cast %[[VAL_100]] : i64 to index
// CHECK:               %[[VAL_103:.*]] = arith.index_cast %[[VAL_101]] : i64 to index
// CHECK:               %[[VAL_104:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xf64>
// CHECK:               %[[VAL_105:.*]] = memref.alloc(%[[VAL_13]]) : memref<?xi1>
// CHECK:               linalg.fill(%[[VAL_8]], %[[VAL_105]]) : i1, memref<?xi1>
// CHECK:               scf.parallel (%[[VAL_106:.*]]) = (%[[VAL_98]]) to (%[[VAL_99]]) step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_107:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_106]]] : memref<?xi64>
// CHECK:                 %[[VAL_108:.*]] = arith.index_cast %[[VAL_107]] : i64 to index
// CHECK:                 memref.store %[[VAL_7]], %[[VAL_105]]{{\[}}%[[VAL_108]]] : memref<?xi1>
// CHECK:                 %[[VAL_109:.*]] = memref.load %[[VAL_26]]{{\[}}%[[VAL_106]]] : memref<?xf64>
// CHECK:                 memref.store %[[VAL_109]], %[[VAL_104]]{{\[}}%[[VAL_108]]] : memref<?xf64>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               %[[VAL_110:.*]] = scf.for %[[VAL_111:.*]] = %[[VAL_102]] to %[[VAL_103]] step %[[VAL_4]] iter_args(%[[VAL_112:.*]] = %[[VAL_3]]) -> (index) {
// CHECK:                 %[[VAL_113:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_111]]] : memref<?xi64>
// CHECK:                 %[[VAL_114:.*]] = arith.index_cast %[[VAL_113]] : i64 to index
// CHECK:                 %[[VAL_115:.*]] = arith.addi %[[VAL_114]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_116:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_114]]] : memref<?xi64>
// CHECK:                 %[[VAL_117:.*]] = memref.load %[[VAL_27]]{{\[}}%[[VAL_115]]] : memref<?xi64>
// CHECK:                 %[[VAL_118:.*]] = arith.index_cast %[[VAL_116]] : i64 to index
// CHECK:                 %[[VAL_119:.*]] = arith.index_cast %[[VAL_117]] : i64 to index
// CHECK:                 %[[VAL_120:.*]]:2 = scf.for %[[VAL_121:.*]] = %[[VAL_118]] to %[[VAL_119]] step %[[VAL_4]] iter_args(%[[VAL_122:.*]] = %[[VAL_9]], %[[VAL_123:.*]] = %[[VAL_8]]) -> (f64, i1) {
// CHECK:                   %[[VAL_124:.*]] = memref.load %[[VAL_28]]{{\[}}%[[VAL_121]]] : memref<?xi64>
// CHECK:                   %[[VAL_125:.*]] = arith.index_cast %[[VAL_124]] : i64 to index
// CHECK:                   %[[VAL_126:.*]] = memref.load %[[VAL_105]]{{\[}}%[[VAL_125]]] : memref<?xi1>
// CHECK:                   %[[VAL_127:.*]]:2 = scf.if %[[VAL_126]] -> (f64, i1) {
// CHECK:                     %[[VAL_128:.*]] = arith.addf %[[VAL_122]], %[[VAL_10]] : f64
// CHECK:                     scf.yield %[[VAL_128]], %[[VAL_7]] : f64, i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_122]], %[[VAL_123]] : f64, i1
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_129:.*]]#0, %[[VAL_129]]#1 : f64, i1
// CHECK:                 }
// CHECK:                 %[[VAL_130:.*]] = scf.if %[[VAL_131:.*]]#1 -> (index) {
// CHECK:                   %[[VAL_132:.*]] = arith.addi %[[VAL_95]], %[[VAL_112]] : index
// CHECK:                   memref.store %[[VAL_113]], %[[VAL_87]]{{\[}}%[[VAL_132]]] : memref<?xi64>
// CHECK:                   memref.store %[[VAL_131]]#0, %[[VAL_88]]{{\[}}%[[VAL_132]]] : memref<?xf64>
// CHECK:                   %[[VAL_133:.*]] = arith.addi %[[VAL_112]], %[[VAL_4]] : index
// CHECK:                   scf.yield %[[VAL_133]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_112]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_134:.*]] : index
// CHECK:               }
// CHECK:               memref.dealloc %[[VAL_104]] : memref<?xf64>
// CHECK:               memref.dealloc %[[VAL_105]] : memref<?xi1>
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
