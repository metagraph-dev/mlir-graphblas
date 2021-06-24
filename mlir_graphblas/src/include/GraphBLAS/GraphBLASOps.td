//===- GraphBLASOps.td - GraphBLAS dialect ops -----------*- tablegen -*-===//
//
// TODO add documentation
//
//===--------------------------------------------------------------------===//

#ifndef GRAPHBLAS_OPS
#define GRAPHBLAS_OPS

include "GraphBLASDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class GraphBLAS_Op<string mnemonic, list<OpTrait> traits = []> : Op<GraphBLAS_Dialect, mnemonic, traits> {
    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ConvertLayoutOp : GraphBLAS_Op<"convert_layout", [NoSideEffect]> {
    let summary = "convert graph storage layout";
    let description = [{
        Returns a new sparse tensor with identical values, but a different storage layout.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The output type must be CSR or CSC.  This operation can convert tensors between these types.

        Example:
        ```%answer = graphblas.convert_layout %sparse_tensor : tensor<2x3xf64, #CSR64> to tensor<2x3xf64, #CSC64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];
}

def GraphBLAS_MatrixSelectOp : GraphBLAS_Op<"matrix_select", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "matrix select operation";
    let description = [{
        Returns new sparse tensor(s) with a subset of element from the given matrix.
        The elements included in the resulting sparse tensor vary depending on the selectors given (one of "triu", "tril", or "gt0").
        Multiple selectors may be given, in which case multiple results will be returned
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The resulting sparse tensors will have the same sparsity as the given sparse tensor.

        Example:
        ```%answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<100x100xf64, #CSR64> to tensor<100x100xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input, StrArrayAttr:$selectors);
    let results = (outs Variadic<AnyTensor>:$outputs);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($outputs)
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
        The first input tensors must be CSR format, while the second input tensor must be CSC format.
        The mask (if provided) must be CSR format.
        This operation also accepts an optional region that specifies element-wise postprocessing to be done on the result of the matrix multiplication.
        The region must use `graphblas.yield` to indicate the result of the element-wise postprocessing.

        No Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to tensor<2x2xi64, #CSR64>```
        
        Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_times" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to tensor<2x2xf64, #CSR64>```

        Body Example:
        ```
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
            ^bb0(%value: f64):
                %result = std.mulf %value, %value: f64
                graphblas.yield %result : f64
        }
        ```

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, Optional<AnyTensor>:$mask, StrAttr:$semiring);
    let results = (outs AnyTensor:$output);
    let regions = (region AnyRegion:$body);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) ($body^)?
    }];
}

def GraphBLAS_MatrixMultiplyReduceToScalarOp : GraphBLAS_Op<"matrix_multiply_reduce_to_scalar", [NoSideEffect]> {
    let summary = "matrix multiply followed by reduction to a scalar with an optional structural mask";
    let description = [{
        Performs a matrix multiply followed by a reduction to scalar.
        The multiplication is done according to the given semiring and optional structural mask.
        The semiring must be one of "plus_times", "plus_pair", or "plus_plus".
        The reduction to scalar is done according to the given aggregator.
        The aggregator must be "sum".
        The given sparse tensors must be a matrix, i.e. have rank 2.
        The first input tensors must be CSR format, while the second input tensor must be CSC format.
        The mask (if provided) must be CSR format.

        No Mask Example:
        ```%answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB { semiring = "plus_plus", aggregator = "sum" } : (tensor<2x3xi64, #CSR64>, tensor<3x2xi64, #CSC64>) to f64```

        Mask Example:
        ```%answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB, %mask { semiring = "plus_times", aggregator = "sum" } : (tensor<2x2xf64, #CSR64>, tensor<2x2xf64, #CSC64>, tensor<2x2xf64, #CSR64>) to f64```

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, Optional<AnyTensor>:$mask, StrAttr:$semiring, StrAttr:$aggregator);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)
    }];
}

def GraphBLAS_YieldOp : GraphBLAS_Op<"yield", [NoSideEffect, ReturnLike, Terminator]>, 
    Arguments<(ins Variadic<AnyType>:$values)> {
    let summary = "graphblas yield operation";
    let description = [{
        `graphblas.yield` is a special terminator operation for blocks inside regions in
        several `graphblas` operations.  It returns a value to the enclosing op, with
        a meaning that depends on the op.

        ```mlir
        graphblas.yield %f0 : f64
        ```
    }];

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = [{
           $values attr-dict `:` type($values)
    }];

}

#endif // GRAPHBLAS_OPS
