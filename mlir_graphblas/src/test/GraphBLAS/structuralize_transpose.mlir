// RUN: graphblas-opt %s | graphblas-opt --graphblas-structuralize | tersify_mlir | FileCheck %s

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

module {
// CHECK:         func @transpose_different_compression(%[[VAL_0:.*]]: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSC64> {
// CHECK:           %[[VAL_1:.*]] = graphblas.transpose %[[VAL_0]] : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
// CHECK:           return %[[VAL_1]] : tensor<?x?xi64, #CSC64>
// CHECK:         }
    func @transpose_different_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSC64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
        return %answer : tensor<?x?xi64, #CSC64>
    }

// CHECK:         func @transpose_same_compression(%[[VAL_0:.*]]: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSR64> {
// CHECK:           %[[VAL_1:.*]] = graphblas.convert_layout %[[VAL_0]] : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64>
// CHECK:           %[[VAL_2:.*]] = graphblas.transpose %[[VAL_1]] : tensor<?x?xi64, #CSC64> to tensor<?x?xi64, #CSR64>
// CHECK:           return %[[VAL_2]] : tensor<?x?xi64, #CSR64>
// CHECK:         }
    func @transpose_same_compression(%sparse_tensor: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSR64> {
        %answer = graphblas.transpose %sparse_tensor : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSR64>
        return %answer : tensor<?x?xi64, #CSR64>
    }
   
}
