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

def GraphBLAS_SizeOp : GraphBLAS_Op<"size", [NoSideEffect]> {
    let summary = "return the size of a vector";
    let description = [{
        Returns the size of a vector

        Example:
        ```%size = graphblas.size %sparse_vector : tensor<?xf64, #CV64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];
}

def GraphBLAS_NumRowsOp : GraphBLAS_Op<"num_rows", [NoSideEffect]> {
    let summary = "return the number of rows in a matrix";
    let description = [{
        Returns the number of rows in a matrix

        Example:
        ```%nrows = graphblas.num_rows %sparse_matrix : tensor<?x?xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];
}

def GraphBLAS_NumColsOp : GraphBLAS_Op<"num_cols", [NoSideEffect]> {
    let summary = "return the number of columns in a matrix";
    let description = [{
        Returns the number of columns in a matrix

        Example:
        ```%ncols = graphblas.num_cols %sparse_matrix : tensor<?x?xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];
}

def GraphBLAS_NumValsOp : GraphBLAS_Op<"num_vals", [NoSideEffect]> {
    let summary = "return the number of values in a sparse tensor";
    let description = [{
        Returns the number of values in a sparse tensor

        Example:
        ```%nnz = graphblas.num_vals %sparse_matrix : tensor<?x?xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];
}

def GraphBLAS_DupOp : GraphBLAS_Op<"dup", [NoSideEffect, AllTypesMatch<["input", "output"]>]> {
    let summary = "return a duplicate of the input sparse tensor";
    let description = [{
        Returns a duplicate copy of the input sparse tensor

        Example:
        ```%B = graphblas.dup %A : tensor<?x?xf64, #CSR64>```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];
}

def GraphBLAS_ConvertLayoutOp : GraphBLAS_Op<"convert_layout", [NoSideEffect]> {
    let summary = "convert graph storage layout";
    let description = [{
        Returns a new sparse tensor with identical values, but a different storage layout.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The output type must be CSR or CSC.  This operation can convert tensors between these types.

        Example:
        ```%answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>```
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

        Single Selector Example:
        ```%answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>```

        Multiple Selector Example:
        ```%answers = graphblas.matrix_select %sparse_tensor { selectors = ["triu", "tril"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>```
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
        ```%answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<?x?xi64, #CSR64> to i64```
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
        %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>
        ```
    }];

    let arguments = (ins AnyTensor:$input, AnyType:$thunk, StrAttr:$apply_operator);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input `,` $thunk attr-dict `:` `(` type($input) `,` type($thunk) `)` `to` type($output)
    }];
}

def GraphBLAS_MatrixApplyGenericOp : GraphBLAS_Op<"matrix_apply_generic", [NoSideEffect]> {
    let summary = "generic matrix apply operation";
    let description = [{
        Applies an arbitrary transformation to every element of a matrix.
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];
}

def GraphBLAS_MatrixMultiplyOp : GraphBLAS_Op<"matrix_multiply", [NoSideEffect]> {
    let summary = "matrix multiply operation with an optional structural mask";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional structural mask.
        The semiring must be one of "plus_times", "plus_pair", or "plus_plus".
        The given sparse tensors must be a matrix (i.e. rank 2) or a vector (i.e. rank 1).
        If the first input is a matrix, it must be CSR format.
        If the second input is a matrix, it must be CSC format.
        Matrix times vector will return a vector.
        Vector times matrix will return a vector.
        Matrix times matrix will return a CSR matrix.
        The mask (if provided) must be the same format as the returned object.
        This operation also accepts an optional region that specifies element-wise postprocessing to be done on the result of the matrix multiplication.
        The region must use `graphblas.yield` to indicate the result of the element-wise postprocessing.

        No Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>```
        
        Mask Example:
        ```%answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>```

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
    let results = (outs AnyType:$output);
    let regions = (region AnyRegion:$body);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) ($body^)?
    }];
}

def GraphBLAS_MatrixMultiplyGenericOp : GraphBLAS_Op<"matrix_multiply_generic", [NoSideEffect]> {
    let summary = "generic matrix multiply operation with an optional structural mask";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional structural mask.
        The given sparse tensors must be a matrix, i.e. have rank 2.
        The first input tensors must be CSR format, while the second input tensor must be CSC format.
        The mask (if provided) must be CSR format.

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, Optional<AnyTensor>:$mask);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions
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
        ```%answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB { semiring = "plus_plus", aggregator = "sum" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to f64```

        Mask Example:
        ```%answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB, %mask { semiring = "plus_times", aggregator = "sum" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to f64```

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, Optional<AnyTensor>:$mask, StrAttr:$semiring, StrAttr:$aggregator);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)
    }];
}

def GraphBLAS_VectorAccumulateOp : GraphBLAS_Op<"vector_accumulate", []> {
    let summary = "vector accumulation operation";
    let description = [{
        Performs vector accumulation with the given accumulate operator.
        This returns zero values and modifies the given vector (arg #0) in place.
        The only supported accumulate operator is "plus".
        The given sparse vectors must have rank 1 and must be sparse.

        Example:
        ```
        graphblas.vector_accumulate %vec, %other_vec { accumulate_operator = "plus" } : (tensor<?xi64, #SparseVec64>, tensor<?xi64, #SparseVec64>)
        ```

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b, StrAttr:$accumulate_operator);
    
    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)`
    }];
}

def GraphBLAS_EqualOp : GraphBLAS_Op<"equal", [NoSideEffect]> {
    let summary = "Equality checking operation for vectors and matrices";
    let description = [{
        Performs an equality check.
        The given tensors must be sparse.
        Checks equality of rank and size of tensors, as well as values and structure.

        Example:
        ```
        %answer = graphblas.equal %vec, %other_vec : (tensor<?xi64, #SparseVec64>, tensor<?xi64, #SparseVec64>)
        ```

    }];

    let arguments = (ins AnyTensor:$a, AnyTensor:$b);
    let results = (outs I1:$output);
    
    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b) `)`
    }];
}

// Generic ops

def YIELD_TRANSFORM_IN_A : I64EnumAttrCase<"TRANSFORM_IN_A", 0, "transform_in_a">;
def YIELD_TRANSFORM_IN_B : I64EnumAttrCase<"TRANSFORM_IN_B", 1, "transform_in_b">;
def YIELD_TRANSFORM_OUT  : I64EnumAttrCase<"TRANSFORM_OUT", 2, "transform_out">;
def YIELD_SELECT_IN_A    : I64EnumAttrCase<"SELECT_IN_A", 3, "select_in_a">;
def YIELD_SELECT_IN_B    : I64EnumAttrCase<"SELECT_IN_B", 4, "select_in_b">;
def YIELD_SELECT_OUT     : I64EnumAttrCase<"SELECT_OUT", 5, "select_out">;
def YIELD_ADD_IDENTITY   : I64EnumAttrCase<"ADD_IDENTITY", 6, "add_identity">;
def YIELD_ADD            : I64EnumAttrCase<"ADD", 7, "add">;
def YIELD_MULT_IDENTITY  : I64EnumAttrCase<"MULT_IDENTITY", 8, "mult_identity">;
def YIELD_MULT           : I64EnumAttrCase<"MULT", 9, "mult">;

def YieldKindAttr : I64EnumAttr<
    "YieldKind", "",
    [YIELD_TRANSFORM_IN_A, YIELD_TRANSFORM_IN_B, YIELD_TRANSFORM_OUT,
     YIELD_SELECT_IN_A,    YIELD_SELECT_IN_B,    YIELD_SELECT_OUT,
     YIELD_ADD_IDENTITY,   YIELD_ADD,
     YIELD_MULT_IDENTITY,  YIELD_MULT]
    > {
  let cppNamespace = "::mlir::graphblas";
}

def GraphBLAS_YieldOp : GraphBLAS_Op<"yield", [NoSideEffect, ReturnLike, Terminator]> {
    let summary = "graphblas yield operation";
    let description = [{
        `graphblas.yield` is a special terminator operation for blocks inside regions in
        several `graphblas` operations.  It returns a value to the enclosing op, with
        a meaning that depends on the op.

        ```mlir
        graphblas.yield transform_out %f0 : f64
        ```
    }];
    
    let arguments = (ins
      YieldKindAttr:$kind,
      Variadic<AnyType>:$values
    );

    /*let builders = [
      OpBuilder<(ins "YieldKind":$kind, "ValueRange":$values), [{
      ::buildYieldOp($_builder, $_state, kind, values);
      }]>
    ];*/

    let assemblyFormat = [{
           $kind $values attr-dict `:` type($values)
    }];
}

#endif // GRAPHBLAS_OPS
