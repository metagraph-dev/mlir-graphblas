// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xbf16>, %thunk: bf16) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xbf16>, bf16) to tensor<2x3xbf16, #CSR64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<3xbf16>, %thunk: bf16) -> tensor<3xbf16, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<3xbf16>, bf16) to tensor<3xbf16, #SparseVec64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %answer : tensor<3xbf16, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>, %thunk: bf16) -> tensor<2x3xbf16> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xbf16, #CSR64>, bf16) to tensor<2x3xbf16> // expected-error {{Return value must be a sparse tensor.}}
        return %answer : tensor<2x3xbf16>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<3xbf16, #SparseVec64>, %thunk: bf16) -> tensor<3xbf16> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<3xbf16, #SparseVec64>, bf16) to tensor<3xbf16> // expected-error {{Return value must be a sparse tensor.}}
        return %answer : tensor<3xbf16>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: f16) -> tensor<2x3xf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, f16) to tensor<2x3xf16, #CSR64> // expected-error {{Element type of input tensor does not match type of thunk.}}
        return %answer : tensor<2x3xf16, #CSR64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<6xi8, #SparseVec64>, %thunk: f16) -> tensor<6xf16, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<6xi8, #SparseVec64>, f16) to tensor<6xf16, #SparseVec64> // expected-error {{Element type of input tensor does not match type of thunk.}}
        return %answer : tensor<6xf16, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<2x3xf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<2x3xf16, #CSR64> // expected-error {{Element type of result tensor does not match type of thunk.}}
        return %answer : tensor<2x3xf16, #CSR64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<6xi8, #SparseVec64>, %thunk: i8) -> tensor<6xf16, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<6xi8, #SparseVec64>, i8) to tensor<6xf16, #SparseVec64> // expected-error {{Element type of result tensor does not match type of thunk.}}
        return %answer : tensor<6xf16, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<99x99xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<99x99xi8, #CSR64> // expected-error {{Input shape does not match output shape.}}
        return %answer : tensor<99x99xi8, #CSR64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<6xi8, #SparseVec64>, %thunk: i8) -> tensor<9999xi8, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<6xi8, #SparseVec64>, i8) to tensor<9999xi8, #SparseVec64> // expected-error {{Input shape does not match output shape.}}
        return %answer : tensor<9999xi8, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xi8, #CSR64>, %thunk: i8) -> tensor<2x3xi8, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "BADOPERATOR" } : (tensor<2x3xi8, #CSR64>, i8) to tensor<2x3xi8, #CSR64> // expected-error {{"BADOPERATOR" is not a supported operator.}}
        return %answer : tensor<2x3xi8, #CSR64>
    }
}

// -----

#SparseVec64 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<6xi8, #SparseVec64>, %thunk: i8) -> tensor<6xi8, #SparseVec64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "BADOPERATOR" } : (tensor<6xi8, #SparseVec64>, i8) to tensor<6xi8, #SparseVec64> // expected-error {{"BADOPERATOR" is not a supported operator.}}
        return %answer : tensor<6xi8, #SparseVec64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>, %thunk: bf16) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %thunk { apply_operator = "abs" } : (tensor<2x3xbf16, #CSR64>, bf16) to tensor<2x3xbf16, #CSR64> // expected-error {{"abs" is a unary opertator, but was given a thunk.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor { apply_operator = "min" } : (tensor<2x3xbf16, #CSR64>) to tensor<2x3xbf16, #CSR64> // expected-error {{"min" requires a thunk.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%thunk: bf16) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.apply %thunk, %thunk { apply_operator = "div" } : (bf16, bf16) to tensor<2x3xbf16, #CSR64> // expected-error {{Exactly one operand must be a ranked tensor.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}

// -----

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @apply_wrapper(%sparse_tensor: tensor<2x3xbf16, #CSR64>) -> tensor<2x3xbf16, #CSR64> {
        %answer = graphblas.apply %sparse_tensor, %sparse_tensor { apply_operator = "div" } : (tensor<2x3xbf16, #CSR64>, tensor<2x3xbf16, #CSR64>) to tensor<2x3xbf16, #CSR64> // expected-error {{Exactly one operand must be a ranked tensor.}}
        return %answer : tensor<2x3xbf16, #CSR64>
    }
}
