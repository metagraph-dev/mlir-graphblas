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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand #0 must be a sparse tensor.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_pair" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand #1 must be a sparse tensor.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64> // expected-error {{Return value must be a sparse tensor.}}
        return %answer : tensor<2x2xi64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "BAD_SEMIRING" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64> // expected-error {{"BAD_SEMIRING" is not a supported semiring.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x9xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x9xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand shapes are incompatible.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<9x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<9x2xi64, #CSR64> // expected-error {{Operand shapes incompatible with output shape.}}
        return %answer : tensor<9x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>) -> tensor<2x9xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x9xi64, #CSR64> // expected-error {{Operand shapes incompatible with output shape.}}
        return %answer : tensor<2x9xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>, %mask: tensor<2x2xi64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask{ semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>, tensor<2x2xi64>) to tensor<2x2xi64, #CSR64> // expected-error {{Operand #2 must be a sparse tensor.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>, %mask: tensor<2x999xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask{ semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>, tensor<2x999xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{Mask shape must match output shape.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>, %mask: tensor<999x2xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask{ semiring = "plus_pair" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>, tensor<999x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{Mask shape must match output shape.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x3xi64, #CSR64>, %argB: tensor<3x2xi64, #CSC64>, %mask: tensor<999x999xi64, #CSR64>) -> tensor<2x2xi64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask{ semiring = "plus_times" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>, tensor<999x999xi64, #CSR64>) to tensor<2x2xi64, #CSR64> // expected-error {{Mask shape must match output shape.}}
        return %answer : tensor<2x2xi64, #CSR64>
    }
}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x2xf64, #CSC64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_plus" } : (tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64> // expected-error {{Operand #0 must have CSR compression.}}
        return %answer : tensor<2x2xf64, #CSR64>
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
    func @matrix_multiply_wrapper(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64> // expected-error {{Operand #1 must have CSC compression.}}
        return %answer : tensor<2x2xf64, #CSR64>
    }

}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSC64>) -> tensor<2x2xf64, #CSR64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_plus" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>) to tensor<2x2xf64, #CSR64> // expected-error {{Operand #2 must have CSR compression.}}
        return %answer : tensor<2x2xf64, #CSR64>
    }

}

// -----

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
    func @matrix_multiply_wrapper(%argA: tensor<2x2xf64, #CSR64>, %argB: tensor<2x2xf64, #CSC64>, %mask: tensor<2x2xf64, #CSR64>) -> tensor<2x2xf64, #CSC64> {
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_plus" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSC64> // expected-error {{Return value must have CSR compression.}}
        return %answer : tensor<2x2xf64, #CSC64>
    }

}
