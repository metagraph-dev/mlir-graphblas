//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GraphBLAS_Op<string mnemonic, list<OpTrait> traits = []> : Op<GraphBLAS_Dialect, mnemonic, traits> {
    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_TransposeOp : GraphBLAS_Op<"transpose", [NoSideEffect]> {
    let summary = "transpose operation";
    let description = [{
        Returns a new sparse tensor that's the transpose of the original sparse tensor.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
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
}

def GraphBLAS_MatrixSelectOp : GraphBLAS_Op<"matrix_select", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "matrix select operation";
    let description = [{
        Returns a new sparse tensor with a subset of element from the given matrix.
        The elements included in the resulting sparse tensor vary depending on the selector given (one of "triu", "tril", or "gt0").
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The resulting sparse tensor will have the same sparsity as the given sparse tensor.

        Example:
        ```%answer = graphblas.matrix_select %sparse_tensor { selector = "triu" } : tensor<100x100xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input, StrAttr:$selector);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];
}

def GraphBLAS_MatrixReduceToScalarOp : GraphBLAS_Op<"matrix_reduce_to_scalar", [NoSideEffect]> {
    let summary = "matrix reduce to scalar operation";
    let description = [{
        Reduces a sparse tensor to a scalar according to the given aggregator.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The resulting scalar's type will depend on the type of the input tensor.

        Example:
        ```%answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<2x3xi64, #CSR64> to i64```
    }];

    let arguments = (ins AnyTensor:$input, StrAttr:$aggregator);
    let results = (outs AnyType:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];
}

def GraphBLAS_MatrixApplyOp : GraphBLAS_Op<"matrix_apply", [NoSideEffect]> {
    let summary = "matrix apply operation";
    let description = [{
        Applies an operator to all elements of the given sparse matrix.
        How the thunk is used depends on the chosen operator.
        Currently, the only valid operator is "min".
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.

        Example:
        ```
        %thunk = constant 100 : i64
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<2x3xi64, #CSR64>, i64) to tensor<2x3xi64, #CSR64>
        ```
    }];

    let arguments = (ins AnyTensor:$input, AnyType:$thunk, StrAttr:$apply_operator);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input `,` $thunk attr-dict `:` `(` type($input) `,` type($thunk) `)` `to` type($output)
    }];
}

def GraphBLAS_MatrixMultiplyOp : GraphBLAS_Op<"matrix_multiply", [NoSideEffect]> {
    let summary = "matrix multiply operation with an optional structural mask";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional structural mask.
        The semiring must be one of "plus_times", "plus_pair", or "plus_plus".
        The given sparse tensors must be a matrix, i.e. have rank 2.
        The given tensors must have a CSR sparsity (i.e. a sparsity of [ "dense", "compressed" ]) or a CSC sparsity (i.e. a sparsity of [ "compressed", "dense" ]).

        No Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSR64>) to tensor<2x2xi64, #CSR64>```
        
        Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSC64>) to tensor<2x2xf64, #CSC64>```
    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, Optional<AnyTensor>:$mask, StrAttr:$semiring);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)
    }];
}

#endif // GRAPHBLAS_OPS
