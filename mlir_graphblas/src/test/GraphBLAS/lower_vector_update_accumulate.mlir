// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL:   func @vector_update_accumulate(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: tensor<?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>) {
// COM: make this a functional test

func @vector_update_accumulate(%input: tensor<?xf64, #CV64>, %output: tensor<?xf64, #CV64>) {
    graphblas.update %input -> %output { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    return
}

// TODO: Check all type combinations
