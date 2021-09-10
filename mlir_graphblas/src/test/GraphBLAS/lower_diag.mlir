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

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {

// CHECK:           builtin.func private @matrix_resize_values(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK:           builtin.func private @matrix_resize_index(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @matrix_resize_pointers(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @matrix_resize_dim(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @cast_csr_to_csx(tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           builtin.func private @matrix_empty(tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           builtin.func @vec_to_mat_fixed_csr(%[[VAL_0:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 8 : index
// CHECK:             %[[VAL_4:.*]] = constant 7 : index
// CHECK:             %[[VAL_5:.*]] = constant 2 : index
// CHECK:             %[[VAL_6:.*]] = constant 0 : index
// CHECK:             %[[VAL_7:.*]] = constant 1 : index
// CHECK:             %[[VAL_8:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_6]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_10:.*]] = call @matrix_empty(%[[VAL_0]], %[[VAL_5]]) : (tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_11:.*]] = call @cast_csr_to_csx(%[[VAL_10]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_11]], %[[VAL_6]], %[[VAL_4]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_12:.*]] = call @cast_csr_to_csx(%[[VAL_10]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_12]], %[[VAL_7]], %[[VAL_4]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_13:.*]] = call @cast_csr_to_csx(%[[VAL_10]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_pointers(%[[VAL_13]], %[[VAL_7]], %[[VAL_3]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_6]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<?xi64>
// CHECK:             %[[VAL_16:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:             %[[VAL_17:.*]] = call @cast_csr_to_csx(%[[VAL_10]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_index(%[[VAL_17]], %[[VAL_7]], %[[VAL_16]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_18:.*]] = call @cast_csr_to_csx(%[[VAL_10]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_values(%[[VAL_18]], %[[VAL_16]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_10]], %[[VAL_7]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_10]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_21:.*]] = %[[VAL_6]] to %[[VAL_16]] step %[[VAL_7]] {
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_22]], %[[VAL_19]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_21]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_23]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = sparse_tensor.pointers %[[VAL_10]], %[[VAL_7]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_25:.*]]:3 = scf.for %[[VAL_26:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_7]] iter_args(%[[VAL_27:.*]] = %[[VAL_1]], %[[VAL_28:.*]] = %[[VAL_6]], %[[VAL_29:.*]] = %[[VAL_1]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_27]], %[[VAL_24]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_26]] : index to i64
// CHECK:               %[[VAL_31:.*]] = cmpi eq, %[[VAL_29]], %[[VAL_30]] : i64
// CHECK:               %[[VAL_32:.*]]:3 = scf.if %[[VAL_31]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_33:.*]] = addi %[[VAL_27]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_34:.*]] = addi %[[VAL_28]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_35:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_27]], %[[VAL_28]], %[[VAL_29]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_36:.*]]#0, %[[VAL_36]]#1, %[[VAL_36]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_15]], %[[VAL_24]]{{\[}}%[[VAL_4]]] : memref<?xi64>
// CHECK:             return %[[VAL_10]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csr(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSR64>
       return %answer : tensor<7x7xf64, #CSR64>
   }

// CHECK:           builtin.func @vec_to_mat_fixed_csc(%[[VAL_37:.*]]: tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_38:.*]] = constant 0 : i64
// CHECK:             %[[VAL_39:.*]] = constant 1 : i64
// CHECK:             %[[VAL_40:.*]] = constant 8 : index
// CHECK:             %[[VAL_41:.*]] = constant 7 : index
// CHECK:             %[[VAL_42:.*]] = constant 2 : index
// CHECK:             %[[VAL_43:.*]] = constant 0 : index
// CHECK:             %[[VAL_44:.*]] = constant 1 : index
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.indices %[[VAL_37]], %[[VAL_43]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_46:.*]] = sparse_tensor.values %[[VAL_37]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_47:.*]] = call @matrix_empty(%[[VAL_37]], %[[VAL_42]]) : (tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_48:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_48]], %[[VAL_43]], %[[VAL_41]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_49:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_49]], %[[VAL_44]], %[[VAL_41]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_50:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_pointers(%[[VAL_50]], %[[VAL_44]], %[[VAL_40]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.pointers %[[VAL_37]], %[[VAL_43]] : tensor<7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]] = memref.load %[[VAL_51]]{{\[}}%[[VAL_44]]] : memref<?xi64>
// CHECK:             %[[VAL_53:.*]] = index_cast %[[VAL_52]] : i64 to index
// CHECK:             %[[VAL_54:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_index(%[[VAL_54]], %[[VAL_44]], %[[VAL_53]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_55:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_values(%[[VAL_55]], %[[VAL_53]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_56:.*]] = sparse_tensor.indices %[[VAL_47]], %[[VAL_44]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_57:.*]] = sparse_tensor.values %[[VAL_47]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_58:.*]] = %[[VAL_43]] to %[[VAL_53]] step %[[VAL_44]] {
// CHECK:               %[[VAL_59:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_58]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_59]], %[[VAL_56]]{{\[}}%[[VAL_58]]] : memref<?xi64>
// CHECK:               %[[VAL_60:.*]] = memref.load %[[VAL_46]]{{\[}}%[[VAL_58]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_60]], %[[VAL_57]]{{\[}}%[[VAL_58]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_61:.*]] = sparse_tensor.pointers %[[VAL_47]], %[[VAL_44]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_62:.*]]:3 = scf.for %[[VAL_63:.*]] = %[[VAL_43]] to %[[VAL_41]] step %[[VAL_44]] iter_args(%[[VAL_64:.*]] = %[[VAL_38]], %[[VAL_65:.*]] = %[[VAL_43]], %[[VAL_66:.*]] = %[[VAL_38]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_64]], %[[VAL_61]]{{\[}}%[[VAL_63]]] : memref<?xi64>
// CHECK:               %[[VAL_67:.*]] = index_cast %[[VAL_63]] : index to i64
// CHECK:               %[[VAL_68:.*]] = cmpi eq, %[[VAL_66]], %[[VAL_67]] : i64
// CHECK:               %[[VAL_69:.*]]:3 = scf.if %[[VAL_68]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_70:.*]] = addi %[[VAL_64]], %[[VAL_39]] : i64
// CHECK:                 %[[VAL_71:.*]] = addi %[[VAL_65]], %[[VAL_44]] : index
// CHECK:                 %[[VAL_72:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_70]], %[[VAL_71]], %[[VAL_72]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_66]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_73:.*]]#0, %[[VAL_73]]#1, %[[VAL_73]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_52]], %[[VAL_61]]{{\[}}%[[VAL_41]]] : memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = call @cast_csr_to_csx(%[[VAL_47]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_75:.*]] = call @cast_csx_to_csc(%[[VAL_74]]) : (tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             return %[[VAL_75]] : tensor<7x7xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_fixed_csc(%sparse_tensor: tensor<7xf64, #SparseVec64>) -> tensor<7x7xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<7xf64, #SparseVec64> to tensor<7x7xf64, #CSC64>
       return %answer : tensor<7x7xf64, #CSC64>
   }

// CHECK:           builtin.func @mat_to_vec_fixed_csr(%[[VAL_76:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = constant true
// CHECK:             %[[VAL_78:.*]] = constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = constant 0 : index
// CHECK:             %[[VAL_80:.*]] = constant 2 : index
// CHECK:             %[[VAL_81:.*]] = constant 7 : i64
// CHECK:             %[[VAL_82:.*]] = constant 7 : index
// CHECK:             %[[VAL_83:.*]] = constant 1 : index
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_83]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_83]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_87:.*]] = call @cast_csc_to_csx(%[[VAL_76]]) : (tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_88:.*]] = call @vector_empty(%[[VAL_87]], %[[VAL_83]]) : (tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @vector_resize_dim(%[[VAL_88]], %[[VAL_79]], %[[VAL_82]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_89:.*]] = scf.for %[[VAL_90:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_83]] iter_args(%[[VAL_91:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_92:.*]] = addi %[[VAL_90]], %[[VAL_83]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_90]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_92]]] : memref<?xi64>
// CHECK:               %[[VAL_95:.*]] = index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_94]] : i64 to index
// CHECK:               %[[VAL_97:.*]] = index_cast %[[VAL_90]] : index to i64
// CHECK:               %[[VAL_98:.*]]:2 = scf.while (%[[VAL_99:.*]] = %[[VAL_95]], %[[VAL_100:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_101:.*]] = cmpi ult, %[[VAL_99]], %[[VAL_96]] : index
// CHECK:                 %[[VAL_102:.*]] = and %[[VAL_100]], %[[VAL_101]] : i1
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_99]], %[[VAL_100]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_103:.*]]: index, %[[VAL_104:.*]]: i1):
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_103]]] : memref<?xi64>
// CHECK:                 %[[VAL_106:.*]] = cmpi ne, %[[VAL_105]], %[[VAL_97]] : i64
// CHECK:                 %[[VAL_107:.*]] = addi %[[VAL_103]], %[[VAL_83]] : index
// CHECK:                 scf.yield %[[VAL_107]], %[[VAL_106]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_108:.*]] = scf.if %[[VAL_109:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_91]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_110:.*]] = addi %[[VAL_91]], %[[VAL_83]] : index
// CHECK:                 scf.yield %[[VAL_110]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_111:.*]] : index
// CHECK:             }
// CHECK:             call @vector_resize_pointers(%[[VAL_88]], %[[VAL_79]], %[[VAL_80]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_index(%[[VAL_88]], %[[VAL_79]], %[[VAL_112:.*]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_values(%[[VAL_88]], %[[VAL_112]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_113:.*]] = sparse_tensor.pointers %[[VAL_88]], %[[VAL_79]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             memref.store %[[VAL_81]], %[[VAL_113]]{{\[}}%[[VAL_83]]] : memref<?xi64>
// CHECK:             %[[VAL_114:.*]] = sparse_tensor.indices %[[VAL_88]], %[[VAL_79]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_115:.*]] = sparse_tensor.values %[[VAL_88]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_116:.*]] = scf.for %[[VAL_117:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_83]] iter_args(%[[VAL_118:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_119:.*]] = addi %[[VAL_117]], %[[VAL_83]] : index
// CHECK:               %[[VAL_120:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:               %[[VAL_121:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_119]]] : memref<?xi64>
// CHECK:               %[[VAL_122:.*]] = index_cast %[[VAL_120]] : i64 to index
// CHECK:               %[[VAL_123:.*]] = index_cast %[[VAL_121]] : i64 to index
// CHECK:               %[[VAL_124:.*]] = index_cast %[[VAL_117]] : index to i64
// CHECK:               %[[VAL_125:.*]]:3 = scf.while (%[[VAL_126:.*]] = %[[VAL_122]], %[[VAL_127:.*]] = %[[VAL_77]], %[[VAL_128:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_129:.*]] = cmpi ult, %[[VAL_126]], %[[VAL_123]] : index
// CHECK:                 %[[VAL_130:.*]] = and %[[VAL_127]], %[[VAL_129]] : i1
// CHECK:                 scf.condition(%[[VAL_130]]) %[[VAL_126]], %[[VAL_127]], %[[VAL_128]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_131:.*]]: index, %[[VAL_132:.*]]: i1, %[[VAL_133:.*]]: i64):
// CHECK:                 %[[VAL_134:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_131]]] : memref<?xi64>
// CHECK:                 %[[VAL_135:.*]] = cmpi ne, %[[VAL_134]], %[[VAL_124]] : i64
// CHECK:                 %[[VAL_136:.*]] = scf.if %[[VAL_135]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_133]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_137:.*]] = memref.load %[[VAL_86]]{{\[}}%[[VAL_131]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_138:.*]] = addi %[[VAL_131]], %[[VAL_83]] : index
// CHECK:                 scf.yield %[[VAL_138]], %[[VAL_135]], %[[VAL_139:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_140:.*]] = scf.if %[[VAL_141:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_118]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_142:.*]]#2, %[[VAL_115]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_124]], %[[VAL_114]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                 %[[VAL_143:.*]] = addi %[[VAL_118]], %[[VAL_83]] : index
// CHECK:                 scf.yield %[[VAL_143]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_144:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_88]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_fixed_csr(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }
    
// CHECK:           builtin.func @mat_to_vec_fixed_csc(%[[VAL_145:.*]]: tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_146:.*]] = constant true
// CHECK:             %[[VAL_147:.*]] = constant 1 : i64
// CHECK:             %[[VAL_148:.*]] = constant 0 : index
// CHECK:             %[[VAL_149:.*]] = constant 2 : index
// CHECK:             %[[VAL_150:.*]] = constant 7 : i64
// CHECK:             %[[VAL_151:.*]] = constant 7 : index
// CHECK:             %[[VAL_152:.*]] = constant 1 : index
// CHECK:             %[[VAL_153:.*]] = sparse_tensor.pointers %[[VAL_145]], %[[VAL_152]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_154:.*]] = sparse_tensor.indices %[[VAL_145]], %[[VAL_152]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_155:.*]] = sparse_tensor.values %[[VAL_145]] : tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_156:.*]] = call @cast_csc_to_csx(%[[VAL_145]]) : (tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_157:.*]] = call @vector_empty(%[[VAL_156]], %[[VAL_152]]) : (tensor<7x7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @vector_resize_dim(%[[VAL_157]], %[[VAL_148]], %[[VAL_151]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_158:.*]] = scf.for %[[VAL_159:.*]] = %[[VAL_148]] to %[[VAL_151]] step %[[VAL_152]] iter_args(%[[VAL_160:.*]] = %[[VAL_148]]) -> (index) {
// CHECK:               %[[VAL_161:.*]] = addi %[[VAL_159]], %[[VAL_152]] : index
// CHECK:               %[[VAL_162:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_159]]] : memref<?xi64>
// CHECK:               %[[VAL_163:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_161]]] : memref<?xi64>
// CHECK:               %[[VAL_164:.*]] = index_cast %[[VAL_162]] : i64 to index
// CHECK:               %[[VAL_165:.*]] = index_cast %[[VAL_163]] : i64 to index
// CHECK:               %[[VAL_166:.*]] = index_cast %[[VAL_159]] : index to i64
// CHECK:               %[[VAL_167:.*]]:2 = scf.while (%[[VAL_168:.*]] = %[[VAL_164]], %[[VAL_169:.*]] = %[[VAL_146]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_170:.*]] = cmpi ult, %[[VAL_168]], %[[VAL_165]] : index
// CHECK:                 %[[VAL_171:.*]] = and %[[VAL_169]], %[[VAL_170]] : i1
// CHECK:                 scf.condition(%[[VAL_171]]) %[[VAL_168]], %[[VAL_169]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_172:.*]]: index, %[[VAL_173:.*]]: i1):
// CHECK:                 %[[VAL_174:.*]] = memref.load %[[VAL_154]]{{\[}}%[[VAL_172]]] : memref<?xi64>
// CHECK:                 %[[VAL_175:.*]] = cmpi ne, %[[VAL_174]], %[[VAL_166]] : i64
// CHECK:                 %[[VAL_176:.*]] = addi %[[VAL_172]], %[[VAL_152]] : index
// CHECK:                 scf.yield %[[VAL_176]], %[[VAL_175]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_177:.*]] = scf.if %[[VAL_178:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_160]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_179:.*]] = addi %[[VAL_160]], %[[VAL_152]] : index
// CHECK:                 scf.yield %[[VAL_179]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_180:.*]] : index
// CHECK:             }
// CHECK:             call @vector_resize_pointers(%[[VAL_157]], %[[VAL_148]], %[[VAL_149]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_index(%[[VAL_157]], %[[VAL_148]], %[[VAL_181:.*]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_values(%[[VAL_157]], %[[VAL_181]]) : (tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_182:.*]] = sparse_tensor.pointers %[[VAL_157]], %[[VAL_148]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             memref.store %[[VAL_150]], %[[VAL_182]]{{\[}}%[[VAL_152]]] : memref<?xi64>
// CHECK:             %[[VAL_183:.*]] = sparse_tensor.indices %[[VAL_157]], %[[VAL_148]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_184:.*]] = sparse_tensor.values %[[VAL_157]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_185:.*]] = scf.for %[[VAL_186:.*]] = %[[VAL_148]] to %[[VAL_151]] step %[[VAL_152]] iter_args(%[[VAL_187:.*]] = %[[VAL_148]]) -> (index) {
// CHECK:               %[[VAL_188:.*]] = addi %[[VAL_186]], %[[VAL_152]] : index
// CHECK:               %[[VAL_189:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_186]]] : memref<?xi64>
// CHECK:               %[[VAL_190:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_188]]] : memref<?xi64>
// CHECK:               %[[VAL_191:.*]] = index_cast %[[VAL_189]] : i64 to index
// CHECK:               %[[VAL_192:.*]] = index_cast %[[VAL_190]] : i64 to index
// CHECK:               %[[VAL_193:.*]] = index_cast %[[VAL_186]] : index to i64
// CHECK:               %[[VAL_194:.*]]:3 = scf.while (%[[VAL_195:.*]] = %[[VAL_191]], %[[VAL_196:.*]] = %[[VAL_146]], %[[VAL_197:.*]] = %[[VAL_147]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_198:.*]] = cmpi ult, %[[VAL_195]], %[[VAL_192]] : index
// CHECK:                 %[[VAL_199:.*]] = and %[[VAL_196]], %[[VAL_198]] : i1
// CHECK:                 scf.condition(%[[VAL_199]]) %[[VAL_195]], %[[VAL_196]], %[[VAL_197]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_200:.*]]: index, %[[VAL_201:.*]]: i1, %[[VAL_202:.*]]: i64):
// CHECK:                 %[[VAL_203:.*]] = memref.load %[[VAL_154]]{{\[}}%[[VAL_200]]] : memref<?xi64>
// CHECK:                 %[[VAL_204:.*]] = cmpi ne, %[[VAL_203]], %[[VAL_193]] : i64
// CHECK:                 %[[VAL_205:.*]] = scf.if %[[VAL_204]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_202]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_206:.*]] = memref.load %[[VAL_155]]{{\[}}%[[VAL_200]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_206]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_207:.*]] = addi %[[VAL_200]], %[[VAL_152]] : index
// CHECK:                 scf.yield %[[VAL_207]], %[[VAL_204]], %[[VAL_208:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_209:.*]] = scf.if %[[VAL_210:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_187]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_211:.*]]#2, %[[VAL_184]]{{\[}}%[[VAL_187]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_193]], %[[VAL_183]]{{\[}}%[[VAL_187]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = addi %[[VAL_187]], %[[VAL_152]] : index
// CHECK:                 scf.yield %[[VAL_212]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_213:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_157]] : tensor<7xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
// CHECK:         }

    func @mat_to_vec_fixed_csc(%mat: tensor<7x7xi64, #CSC64>) -> tensor<7xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<7x7xi64, #CSC64> to tensor<7xi64, #SparseVec64>
        return %vec : tensor<7xi64, #SparseVec64>
    }

}

module {

// CHECK:           builtin.func private @matrix_resize_values(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index)
// CHECK:           builtin.func private @matrix_resize_index(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @matrix_resize_pointers(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @matrix_resize_dim(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index)
// CHECK:           builtin.func private @cast_csr_to_csx(tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           builtin.func private @matrix_empty(tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           builtin.func @vec_to_mat_arbitrary_csr(%[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_1:.*]] = constant 0 : i64
// CHECK:             %[[VAL_2:.*]] = constant 1 : i64
// CHECK:             %[[VAL_3:.*]] = constant 2 : index
// CHECK:             %[[VAL_4:.*]] = constant 0 : index
// CHECK:             %[[VAL_5:.*]] = constant 1 : index
// CHECK:             %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_4]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_9:.*]] = call @matrix_empty(%[[VAL_0]], %[[VAL_3]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_10:.*]] = call @cast_csr_to_csx(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_10]], %[[VAL_4]], %[[VAL_6]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_11:.*]] = call @cast_csr_to_csx(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_11]], %[[VAL_5]], %[[VAL_6]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_12:.*]] = addi %[[VAL_6]], %[[VAL_5]] : index
// CHECK:             %[[VAL_13:.*]] = call @cast_csr_to_csx(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_pointers(%[[VAL_13]], %[[VAL_5]], %[[VAL_12]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_14:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_4]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_5]]] : memref<?xi64>
// CHECK:             %[[VAL_16:.*]] = index_cast %[[VAL_15]] : i64 to index
// CHECK:             %[[VAL_17:.*]] = call @cast_csr_to_csx(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_index(%[[VAL_17]], %[[VAL_5]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_18:.*]] = call @cast_csr_to_csx(%[[VAL_9]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_values(%[[VAL_18]], %[[VAL_16]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_19:.*]] = sparse_tensor.indices %[[VAL_9]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_20:.*]] = sparse_tensor.values %[[VAL_9]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_21:.*]] = %[[VAL_4]] to %[[VAL_16]] step %[[VAL_5]] {
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_22]], %[[VAL_19]]{{\[}}%[[VAL_21]]] : memref<?xi64>
// CHECK:               %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_21]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_23]], %[[VAL_20]]{{\[}}%[[VAL_21]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = sparse_tensor.pointers %[[VAL_9]], %[[VAL_5]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_25:.*]]:3 = scf.for %[[VAL_26:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_5]] iter_args(%[[VAL_27:.*]] = %[[VAL_1]], %[[VAL_28:.*]] = %[[VAL_4]], %[[VAL_29:.*]] = %[[VAL_1]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_27]], %[[VAL_24]]{{\[}}%[[VAL_26]]] : memref<?xi64>
// CHECK:               %[[VAL_30:.*]] = index_cast %[[VAL_26]] : index to i64
// CHECK:               %[[VAL_31:.*]] = cmpi eq, %[[VAL_29]], %[[VAL_30]] : i64
// CHECK:               %[[VAL_32:.*]]:3 = scf.if %[[VAL_31]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_33:.*]] = addi %[[VAL_27]], %[[VAL_2]] : i64
// CHECK:                 %[[VAL_34:.*]] = addi %[[VAL_28]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_35:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_34]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_27]], %[[VAL_28]], %[[VAL_29]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_36:.*]]#0, %[[VAL_36]]#1, %[[VAL_36]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_15]], %[[VAL_24]]{{\[}}%[[VAL_6]]] : memref<?xi64>
// CHECK:             return %[[VAL_9]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csr(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSR64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSR64>
       return %answer : tensor<?x?xf64, #CSR64>
   }

// CHECK:           builtin.func @vec_to_mat_arbitrary_csc(%[[VAL_37:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_38:.*]] = constant 0 : i64
// CHECK:             %[[VAL_39:.*]] = constant 1 : i64
// CHECK:             %[[VAL_40:.*]] = constant 2 : index
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 1 : index
// CHECK:             %[[VAL_43:.*]] = tensor.dim %[[VAL_37]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_44:.*]] = sparse_tensor.indices %[[VAL_37]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_45:.*]] = sparse_tensor.values %[[VAL_37]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             %[[VAL_46:.*]] = call @matrix_empty(%[[VAL_37]], %[[VAL_40]]) : (tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_47:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_47]], %[[VAL_41]], %[[VAL_43]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_48:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_dim(%[[VAL_48]], %[[VAL_42]], %[[VAL_43]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_49:.*]] = addi %[[VAL_43]], %[[VAL_42]] : index
// CHECK:             %[[VAL_50:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_pointers(%[[VAL_50]], %[[VAL_42]], %[[VAL_49]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.pointers %[[VAL_37]], %[[VAL_41]] : tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_52:.*]] = memref.load %[[VAL_51]]{{\[}}%[[VAL_42]]] : memref<?xi64>
// CHECK:             %[[VAL_53:.*]] = index_cast %[[VAL_52]] : i64 to index
// CHECK:             %[[VAL_54:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_index(%[[VAL_54]], %[[VAL_42]], %[[VAL_53]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_55:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @matrix_resize_values(%[[VAL_55]], %[[VAL_53]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_56:.*]] = sparse_tensor.indices %[[VAL_46]], %[[VAL_42]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_57:.*]] = sparse_tensor.values %[[VAL_46]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xf64>
// CHECK:             scf.for %[[VAL_58:.*]] = %[[VAL_41]] to %[[VAL_53]] step %[[VAL_42]] {
// CHECK:               %[[VAL_59:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_58]]] : memref<?xi64>
// CHECK:               memref.store %[[VAL_59]], %[[VAL_56]]{{\[}}%[[VAL_58]]] : memref<?xi64>
// CHECK:               %[[VAL_60:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_58]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_60]], %[[VAL_57]]{{\[}}%[[VAL_58]]] : memref<?xf64>
// CHECK:             }
// CHECK:             %[[VAL_61:.*]] = sparse_tensor.pointers %[[VAL_46]], %[[VAL_42]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_62:.*]]:3 = scf.for %[[VAL_63:.*]] = %[[VAL_41]] to %[[VAL_43]] step %[[VAL_42]] iter_args(%[[VAL_64:.*]] = %[[VAL_38]], %[[VAL_65:.*]] = %[[VAL_41]], %[[VAL_66:.*]] = %[[VAL_38]]) -> (i64, index, i64) {
// CHECK:               memref.store %[[VAL_64]], %[[VAL_61]]{{\[}}%[[VAL_63]]] : memref<?xi64>
// CHECK:               %[[VAL_67:.*]] = index_cast %[[VAL_63]] : index to i64
// CHECK:               %[[VAL_68:.*]] = cmpi eq, %[[VAL_66]], %[[VAL_67]] : i64
// CHECK:               %[[VAL_69:.*]]:3 = scf.if %[[VAL_68]] -> (i64, index, i64) {
// CHECK:                 %[[VAL_70:.*]] = addi %[[VAL_64]], %[[VAL_39]] : i64
// CHECK:                 %[[VAL_71:.*]] = addi %[[VAL_65]], %[[VAL_42]] : index
// CHECK:                 %[[VAL_72:.*]] = memref.load %[[VAL_44]]{{\[}}%[[VAL_71]]] : memref<?xi64>
// CHECK:                 scf.yield %[[VAL_70]], %[[VAL_71]], %[[VAL_72]] : i64, index, i64
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_64]], %[[VAL_65]], %[[VAL_66]] : i64, index, i64
// CHECK:               }
// CHECK:               scf.yield %[[VAL_73:.*]]#0, %[[VAL_73]]#1, %[[VAL_73]]#2 : i64, index, i64
// CHECK:             }
// CHECK:             memref.store %[[VAL_52]], %[[VAL_61]]{{\[}}%[[VAL_43]]] : memref<?xi64>
// CHECK:             %[[VAL_74:.*]] = call @cast_csr_to_csx(%[[VAL_46]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_75:.*]] = call @cast_csx_to_csc(%[[VAL_74]]) : (tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             return %[[VAL_75]] : tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

   func @vec_to_mat_arbitrary_csc(%sparse_tensor: tensor<?xf64, #SparseVec64>) -> tensor<?x?xf64, #CSC64> {
       %answer = graphblas.diag %sparse_tensor : tensor<?xf64, #SparseVec64> to tensor<?x?xf64, #CSC64>
       return %answer : tensor<?x?xf64, #CSC64>
   }

// CHECK:           builtin.func @mat_to_vec_arbitrary_csr(%[[VAL_76:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_77:.*]] = constant true
// CHECK:             %[[VAL_78:.*]] = constant 1 : i64
// CHECK:             %[[VAL_79:.*]] = constant 0 : index
// CHECK:             %[[VAL_80:.*]] = constant 2 : index
// CHECK:             %[[VAL_81:.*]] = constant 1 : index
// CHECK:             %[[VAL_82:.*]] = tensor.dim %[[VAL_76]], %[[VAL_79]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_83:.*]] = sparse_tensor.pointers %[[VAL_76]], %[[VAL_81]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_84:.*]] = sparse_tensor.indices %[[VAL_76]], %[[VAL_81]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_85:.*]] = sparse_tensor.values %[[VAL_76]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_86:.*]] = call @cast_csc_to_csx(%[[VAL_76]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_87:.*]] = call @vector_empty(%[[VAL_86]], %[[VAL_81]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @vector_resize_dim(%[[VAL_87]], %[[VAL_79]], %[[VAL_82]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_88:.*]] = scf.for %[[VAL_89:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_81]] iter_args(%[[VAL_90:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_91:.*]] = addi %[[VAL_89]], %[[VAL_81]] : index
// CHECK:               %[[VAL_92:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_89]]] : memref<?xi64>
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_91]]] : memref<?xi64>
// CHECK:               %[[VAL_94:.*]] = index_cast %[[VAL_92]] : i64 to index
// CHECK:               %[[VAL_95:.*]] = index_cast %[[VAL_93]] : i64 to index
// CHECK:               %[[VAL_96:.*]] = index_cast %[[VAL_89]] : index to i64
// CHECK:               %[[VAL_97:.*]]:2 = scf.while (%[[VAL_98:.*]] = %[[VAL_94]], %[[VAL_99:.*]] = %[[VAL_77]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_100:.*]] = cmpi ult, %[[VAL_98]], %[[VAL_95]] : index
// CHECK:                 %[[VAL_101:.*]] = and %[[VAL_99]], %[[VAL_100]] : i1
// CHECK:                 scf.condition(%[[VAL_101]]) %[[VAL_98]], %[[VAL_99]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_102:.*]]: index, %[[VAL_103:.*]]: i1):
// CHECK:                 %[[VAL_104:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_102]]] : memref<?xi64>
// CHECK:                 %[[VAL_105:.*]] = cmpi ne, %[[VAL_104]], %[[VAL_96]] : i64
// CHECK:                 %[[VAL_106:.*]] = addi %[[VAL_102]], %[[VAL_81]] : index
// CHECK:                 scf.yield %[[VAL_106]], %[[VAL_105]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_107:.*]] = scf.if %[[VAL_108:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_90]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_109:.*]] = addi %[[VAL_90]], %[[VAL_81]] : index
// CHECK:                 scf.yield %[[VAL_109]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_110:.*]] : index
// CHECK:             }
// CHECK:             call @vector_resize_pointers(%[[VAL_87]], %[[VAL_79]], %[[VAL_80]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_index(%[[VAL_87]], %[[VAL_79]], %[[VAL_111:.*]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_values(%[[VAL_87]], %[[VAL_111]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_112:.*]] = sparse_tensor.pointers %[[VAL_87]], %[[VAL_79]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_113:.*]] = index_cast %[[VAL_82]] : index to i64
// CHECK:             memref.store %[[VAL_113]], %[[VAL_112]]{{\[}}%[[VAL_81]]] : memref<?xi64>
// CHECK:             %[[VAL_114:.*]] = sparse_tensor.indices %[[VAL_87]], %[[VAL_79]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_115:.*]] = sparse_tensor.values %[[VAL_87]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_116:.*]] = scf.for %[[VAL_117:.*]] = %[[VAL_79]] to %[[VAL_82]] step %[[VAL_81]] iter_args(%[[VAL_118:.*]] = %[[VAL_79]]) -> (index) {
// CHECK:               %[[VAL_119:.*]] = addi %[[VAL_117]], %[[VAL_81]] : index
// CHECK:               %[[VAL_120:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_117]]] : memref<?xi64>
// CHECK:               %[[VAL_121:.*]] = memref.load %[[VAL_83]]{{\[}}%[[VAL_119]]] : memref<?xi64>
// CHECK:               %[[VAL_122:.*]] = index_cast %[[VAL_120]] : i64 to index
// CHECK:               %[[VAL_123:.*]] = index_cast %[[VAL_121]] : i64 to index
// CHECK:               %[[VAL_124:.*]] = index_cast %[[VAL_117]] : index to i64
// CHECK:               %[[VAL_125:.*]]:3 = scf.while (%[[VAL_126:.*]] = %[[VAL_122]], %[[VAL_127:.*]] = %[[VAL_77]], %[[VAL_128:.*]] = %[[VAL_78]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_129:.*]] = cmpi ult, %[[VAL_126]], %[[VAL_123]] : index
// CHECK:                 %[[VAL_130:.*]] = and %[[VAL_127]], %[[VAL_129]] : i1
// CHECK:                 scf.condition(%[[VAL_130]]) %[[VAL_126]], %[[VAL_127]], %[[VAL_128]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_131:.*]]: index, %[[VAL_132:.*]]: i1, %[[VAL_133:.*]]: i64):
// CHECK:                 %[[VAL_134:.*]] = memref.load %[[VAL_84]]{{\[}}%[[VAL_131]]] : memref<?xi64>
// CHECK:                 %[[VAL_135:.*]] = cmpi ne, %[[VAL_134]], %[[VAL_124]] : i64
// CHECK:                 %[[VAL_136:.*]] = scf.if %[[VAL_135]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_133]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_137:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_131]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_137]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_138:.*]] = addi %[[VAL_131]], %[[VAL_81]] : index
// CHECK:                 scf.yield %[[VAL_138]], %[[VAL_135]], %[[VAL_139:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_140:.*]] = scf.if %[[VAL_141:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_118]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_142:.*]]#2, %[[VAL_115]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_124]], %[[VAL_114]]{{\[}}%[[VAL_118]]] : memref<?xi64>
// CHECK:                 %[[VAL_143:.*]] = addi %[[VAL_118]], %[[VAL_81]] : index
// CHECK:                 scf.yield %[[VAL_143]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_144:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_87]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }

    func @mat_to_vec_arbitrary_csr(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

// CHECK:           builtin.func @mat_to_vec_arbitrary_csc(%[[VAL_145:.*]]: tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> {
// CHECK:             %[[VAL_146:.*]] = constant true
// CHECK:             %[[VAL_147:.*]] = constant 1 : i64
// CHECK:             %[[VAL_148:.*]] = constant 0 : index
// CHECK:             %[[VAL_149:.*]] = constant 2 : index
// CHECK:             %[[VAL_150:.*]] = constant 1 : index
// CHECK:             %[[VAL_151:.*]] = tensor.dim %[[VAL_145]], %[[VAL_148]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_152:.*]] = sparse_tensor.pointers %[[VAL_145]], %[[VAL_150]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_153:.*]] = sparse_tensor.indices %[[VAL_145]], %[[VAL_150]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_154:.*]] = sparse_tensor.values %[[VAL_145]] : tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_155:.*]] = call @cast_csc_to_csx(%[[VAL_145]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 64, indexBitWidth = 64 }>>) -> tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             %[[VAL_156:.*]] = call @vector_empty(%[[VAL_155]], %[[VAL_150]]) : (tensor<?x?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:             call @vector_resize_dim(%[[VAL_156]], %[[VAL_148]], %[[VAL_151]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             %[[VAL_157:.*]] = scf.for %[[VAL_158:.*]] = %[[VAL_148]] to %[[VAL_151]] step %[[VAL_150]] iter_args(%[[VAL_159:.*]] = %[[VAL_148]]) -> (index) {
// CHECK:               %[[VAL_160:.*]] = addi %[[VAL_158]], %[[VAL_150]] : index
// CHECK:               %[[VAL_161:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_158]]] : memref<?xi64>
// CHECK:               %[[VAL_162:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_160]]] : memref<?xi64>
// CHECK:               %[[VAL_163:.*]] = index_cast %[[VAL_161]] : i64 to index
// CHECK:               %[[VAL_164:.*]] = index_cast %[[VAL_162]] : i64 to index
// CHECK:               %[[VAL_165:.*]] = index_cast %[[VAL_158]] : index to i64
// CHECK:               %[[VAL_166:.*]]:2 = scf.while (%[[VAL_167:.*]] = %[[VAL_163]], %[[VAL_168:.*]] = %[[VAL_146]]) : (index, i1) -> (index, i1) {
// CHECK:                 %[[VAL_169:.*]] = cmpi ult, %[[VAL_167]], %[[VAL_164]] : index
// CHECK:                 %[[VAL_170:.*]] = and %[[VAL_168]], %[[VAL_169]] : i1
// CHECK:                 scf.condition(%[[VAL_170]]) %[[VAL_167]], %[[VAL_168]] : index, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_171:.*]]: index, %[[VAL_172:.*]]: i1):
// CHECK:                 %[[VAL_173:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_171]]] : memref<?xi64>
// CHECK:                 %[[VAL_174:.*]] = cmpi ne, %[[VAL_173]], %[[VAL_165]] : i64
// CHECK:                 %[[VAL_175:.*]] = addi %[[VAL_171]], %[[VAL_150]] : index
// CHECK:                 scf.yield %[[VAL_175]], %[[VAL_174]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_176:.*]] = scf.if %[[VAL_177:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_159]] : index
// CHECK:               } else {
// CHECK:                 %[[VAL_178:.*]] = addi %[[VAL_159]], %[[VAL_150]] : index
// CHECK:                 scf.yield %[[VAL_178]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_179:.*]] : index
// CHECK:             }
// CHECK:             call @vector_resize_pointers(%[[VAL_156]], %[[VAL_148]], %[[VAL_149]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_index(%[[VAL_156]], %[[VAL_148]], %[[VAL_180:.*]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index, index) -> ()
// CHECK:             call @vector_resize_values(%[[VAL_156]], %[[VAL_180]]) : (tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>, index) -> ()
// CHECK:             %[[VAL_181:.*]] = sparse_tensor.pointers %[[VAL_156]], %[[VAL_148]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_182:.*]] = index_cast %[[VAL_151]] : index to i64
// CHECK:             memref.store %[[VAL_182]], %[[VAL_181]]{{\[}}%[[VAL_150]]] : memref<?xi64>
// CHECK:             %[[VAL_183:.*]] = sparse_tensor.indices %[[VAL_156]], %[[VAL_148]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_184:.*]] = sparse_tensor.values %[[VAL_156]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>> to memref<?xi64>
// CHECK:             %[[VAL_185:.*]] = scf.for %[[VAL_186:.*]] = %[[VAL_148]] to %[[VAL_151]] step %[[VAL_150]] iter_args(%[[VAL_187:.*]] = %[[VAL_148]]) -> (index) {
// CHECK:               %[[VAL_188:.*]] = addi %[[VAL_186]], %[[VAL_150]] : index
// CHECK:               %[[VAL_189:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_186]]] : memref<?xi64>
// CHECK:               %[[VAL_190:.*]] = memref.load %[[VAL_152]]{{\[}}%[[VAL_188]]] : memref<?xi64>
// CHECK:               %[[VAL_191:.*]] = index_cast %[[VAL_189]] : i64 to index
// CHECK:               %[[VAL_192:.*]] = index_cast %[[VAL_190]] : i64 to index
// CHECK:               %[[VAL_193:.*]] = index_cast %[[VAL_186]] : index to i64
// CHECK:               %[[VAL_194:.*]]:3 = scf.while (%[[VAL_195:.*]] = %[[VAL_191]], %[[VAL_196:.*]] = %[[VAL_146]], %[[VAL_197:.*]] = %[[VAL_147]]) : (index, i1, i64) -> (index, i1, i64) {
// CHECK:                 %[[VAL_198:.*]] = cmpi ult, %[[VAL_195]], %[[VAL_192]] : index
// CHECK:                 %[[VAL_199:.*]] = and %[[VAL_196]], %[[VAL_198]] : i1
// CHECK:                 scf.condition(%[[VAL_199]]) %[[VAL_195]], %[[VAL_196]], %[[VAL_197]] : index, i1, i64
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_200:.*]]: index, %[[VAL_201:.*]]: i1, %[[VAL_202:.*]]: i64):
// CHECK:                 %[[VAL_203:.*]] = memref.load %[[VAL_153]]{{\[}}%[[VAL_200]]] : memref<?xi64>
// CHECK:                 %[[VAL_204:.*]] = cmpi ne, %[[VAL_203]], %[[VAL_193]] : i64
// CHECK:                 %[[VAL_205:.*]] = scf.if %[[VAL_204]] -> (i64) {
// CHECK:                   scf.yield %[[VAL_202]] : i64
// CHECK:                 } else {
// CHECK:                   %[[VAL_206:.*]] = memref.load %[[VAL_154]]{{\[}}%[[VAL_200]]] : memref<?xi64>
// CHECK:                   scf.yield %[[VAL_206]] : i64
// CHECK:                 }
// CHECK:                 %[[VAL_207:.*]] = addi %[[VAL_200]], %[[VAL_150]] : index
// CHECK:                 scf.yield %[[VAL_207]], %[[VAL_204]], %[[VAL_208:.*]] : index, i1, i64
// CHECK:               }
// CHECK:               %[[VAL_209:.*]] = scf.if %[[VAL_210:.*]]#1 -> (index) {
// CHECK:                 scf.yield %[[VAL_187]] : index
// CHECK:               } else {
// CHECK:                 memref.store %[[VAL_211:.*]]#2, %[[VAL_184]]{{\[}}%[[VAL_187]]] : memref<?xi64>
// CHECK:                 memref.store %[[VAL_193]], %[[VAL_183]]{{\[}}%[[VAL_187]]] : memref<?xi64>
// CHECK:                 %[[VAL_212:.*]] = addi %[[VAL_187]], %[[VAL_150]] : index
// CHECK:                 scf.yield %[[VAL_212]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_213:.*]] : index
// CHECK:             }
// CHECK:             return %[[VAL_156]] : tensor<?xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>
// CHECK:           }
// CHECK:         }

    func @mat_to_vec_arbitrary_csc(%mat: tensor<?x?xi64, #CSC64>) -> tensor<?xi64, #SparseVec64> {
        %vec = graphblas.diag %mat : tensor<?x?xi64, #CSC64> to tensor<?xi64, #SparseVec64>
        return %vec : tensor<?xi64, #SparseVec64>
    }

}
