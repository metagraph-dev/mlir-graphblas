// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

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
    func @cast(%m: tensor<?x?xi64, #CSR64>) -> tensor<?x?xi64, #CSC64> {
        %answer = graphblas.cast %m : tensor<?x?xi64, #CSR64> to tensor<?x?xi64, #CSC64> // expected-error {{result must have CSR compression.}}
        return %answer : tensor<?x?xi64, #CSC64>
    }
}

// -----

#CV64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CV32 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

module {
    func @cast(%v: tensor<?xf64, #CV64>) -> tensor<?xf64, #CV32> {
        %answer = graphblas.cast %v : tensor<?xf64, #CV64> to tensor<?xf64, #CV32> // expected-error {{Changing bit width is not yet supported. Input and output pointer bit widths do not match: 64!=32}}
        return %answer : tensor<?xf64, #CV32>
    }
}