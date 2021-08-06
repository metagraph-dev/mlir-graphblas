// RUN: graphblas-opt %s | graphblas-opt --graphblas-lower | FileCheck %s

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @vector_update_accumulate(%input: tensor<?xf64, #CV64>, %output: tensor<?xf64, #CV64>) -> index {
    %blah = graphblas.update %input -> %output { accumulate_operator = "plus" } : tensor<?xf64, #CV64> -> tensor<?xf64, #CV64>
    return %blah : index
}