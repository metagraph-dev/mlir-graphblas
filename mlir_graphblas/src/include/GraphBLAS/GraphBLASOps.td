//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GraphBLAS_TransposeOp : GraphBLAS_Op<"transpose", [NoSideEffect]> {
    let summary = "transpose operation";
    let description = [{
        Returns a new sparse tensor that's the transpose of the original sparse tensor.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity (i.e. a sparsity of [ "dense", "compressed" ]) or a CSC sparsity (i.e. a sparsity of [ "compressed", "dense" ]).
        This operation is guaranteed to swap the sparsity of CSR -> CSC or CSC -> CSR.

        Example 1:
        ```%answer = graphblas.transpose %sparse_tensor { swap_sizes = false } : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>```

        Example 2:
        ```%answer = graphblas.transpose %sparse_tensor { swap_sizes = true } : tensor<2x3xf64, #CSR64> to tensor<3x2xf64, #CSC64>```
    }];

    let arguments = (ins AnyTensor:$input, BoolAttr:$swap_sizes);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    // TODO add custsom verifier sanity checking the input and output types are sane
    // let verifier = 
}

def GraphBLAS_MatrixSelectOp : GraphBLAS_Op<"matrix_select", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "matrix select operation";
    let description = [{
        Returns a new sparse tensor with a subset of element from the given matrix.
        The elements included in the resulting sparse tensor vary depending on the selector given (one of "triu", "tril", or "gt0").
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity (i.e. a sparsity of [ "dense", "compressed" ]) or a CSC sparsity (i.e. a sparsity of [ "compressed", "dense" ]).
        The resulting sparse tensor will have the same sparsity as the given sparse tensor.

        Example:
        ```%answer = graphblas.matrix_select %sparse_tensor { selector = "triu" } : tensor<100x100xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input, StrAttr:$selector);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    // TODO add custsom verifier sanity checking the selector attribute is sane
    // let verifier = 
}

def GraphBLAS_MatrixReduceToScalarOp : GraphBLAS_Op<"matrix_reduce_to_scalar", [NoSideEffect]> {
    let summary = "matrix reduce to scalar operation";
    let description = [{
        Reduces a sparse tensor to a scalar.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity (i.e. a sparsity of [ "dense", "compressed" ]) or a CSC sparsity (i.e. a sparsity of [ "compressed", "dense" ]).
        The resulting scalar's type will depend on the type of the input tensor.

        Example:
        ```%answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<2x3xi64, #CSR64> to i64```
    }];

    let arguments = (ins AnyTensor:$input, StrAttr:$aggregator);
    let results = (outs AnyType:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    // TODO add custsom verifier sanity checking the type of $output is sane
    // let verifier = 
}

#endif // GRAPHBLAS_OPS
