// RUN: graphblas-opt %s -split-input-file -verify-diagnostics

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<*xf64>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<*xf64> to f64 // expected-error {{op operand #0 must be 1D/2D tensor of any type values, but got 'tensor<*xf64>'}}
        return %answer : f64
    }
}

// -----

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64> to f64 // expected-error {{Operand #0 must be a sparse vector or sparse matrix.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "dense" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?x?xf64, #BADENCODING> to f64 // expected-error {{op operand #0 must be 1D/2D tensor of any type values, but got 'tensor<?x?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed", "dense" ], dimOrdering = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, pointerBitWidth = 64, indexBitWidth = 64 }>>'}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "singleton" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "singleton" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "singleton", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "singleton", "dense" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
    }
}

// -----

#BADENCODING = #sparse_tensor.encoding<{
  dimLevelType = [ "singleton", "singleton" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

module {
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #BADENCODING>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #BADENCODING> to f64 // expected-error {{Operand #0 must have CSR or CSC compression, i.e. must have dimLevelType = [ "dense", "compressed" ] in the sparse encoding.}}
        return %answer : f64
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
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> f64 {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "bad_reducer" } : tensor<?x?xf64, #CSR64> to f64 // expected-error {{"bad_reducer" is not a supported aggregator.}}
        return %answer : f64
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
    func @reduce_to_scalar_wrapper(%sparse_tensor: tensor<?x?xf64, #CSR64>) -> index {
        %answer = graphblas.reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xf64, #CSR64> to index // expected-error {{Operand and output types are incompatible.}}
        return %answer : index
    }
}

