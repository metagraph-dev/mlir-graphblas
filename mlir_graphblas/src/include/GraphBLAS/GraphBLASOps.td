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

class GraphBLAS_Op<string mnemonic, list<OpTrait> traits = []> : Op<GraphBLAS_Dialect, mnemonic, traits> {}

def GraphBlasMatrixOrVectorOperand : TensorRankOf<[AnyType], [1,2]>;
def GraphBlasVectorOperand : 1DTensorOf<[AnyType]>;
def GraphBlasMatrixOperand : 2DTensorOf<[AnyType]>;

def GraphBLAS_SizeOp : GraphBLAS_Op<"size", [NoSideEffect]> {
    let summary = "return the size of a ranked vector";
    let description = [{
        Returns the size of a vector

        Example:
        ```mlir 
          %size = graphblas.size %sparse_vector : tensor<?xf64, #CV64>
        ```
    }];

    let arguments = (ins GraphBlasVectorOperand:$input);
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
        ```mlir 
          %nrows = graphblas.num_rows %sparse_matrix : tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input);
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
        ```mlir 
          %ncols = graphblas.num_cols %sparse_matrix : tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input);
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
        ```mlir
          %nnz = graphblas.num_vals %sparse_matrix : tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs Index:$result);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_DupOp : GraphBLAS_Op<"dup", [NoSideEffect, AllTypesMatch<["input", "output"]>]> {
    let summary = "return a duplicate of the input sparse tensor";
    let description = [{
        Returns a duplicate copy of the input sparse tensor

        Example:
        ```mlir
          %B = graphblas.dup %A : tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input)
    }];

    let builders = [
      OpBuilder<(ins "Value":$tensor)>
    ];

    let verifier = [{ return ::verify(*this); }];
}

// TODO: Is this op still needed (see sparse_tensor.convert)
def GraphBLAS_ConvertLayoutOp : GraphBLAS_Op<"convert_layout", [NoSideEffect]> {
    let summary = "convert graph storage layout";
    let description = [{
        Returns a new sparse tensor with identical values, but a different
        storage layout.  The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.  The output type
        must be CSR or CSC.  This operation can convert tensors between these types.

        Example:
        ```mlir
          %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
        ```
    }];

    let arguments = (ins AnyTensor:$input);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_TransposeOp : GraphBLAS_Op<"transpose", [NoSideEffect]> {
    let summary = "transpose";
    let description = [{
        Returns a new sparse matrix that's the transpose of the input matrix.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The output type must be CSR or CSC.

        Example:
        ```mlir
        %a = graphblas.transpose %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
        %b = graphblas.transpose %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input);
    let results = (outs GraphBlasMatrixOperand:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixSelectOp : GraphBLAS_Op<"matrix_select", [NoSideEffect]> {
    let summary = "matrix select operation";
    let description = [{
        Returns new sparse tensors, each with a subset of elements from the given matrix.
        The elements included in the resulting sparse tensor vary depending on the selectors given (one of "triu", "tril", or "gt").
        Multiple selectors may be given, in which case multiple results will be returned.
        Some selectors, e.g. "gt", require a thunk. The thunks should be passed as extra operands to the op.
        The number of given thunks must match the number of thunk-requiring selectors.
        The ordering of the thunks/selectors determines which thunk is used for which selector, i.e. the nth thunk is used for the nth thunk-requiring selector.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The resulting sparse tensors will have the same sparsity as the given sparse tensor.

        Single Selector Example:
        ```mlir
        %answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
        ```

        Multiple Selector Example:
        ```mlir
        %answers = graphblas.matrix_select %sparse_tensor { selectors = ["triu", "tril"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
        ```

        Thunk-Requiring Selector Example:
        ```mlir
        %thunk_a = constant 0.0 : f64 // used for the first "gt"
        %thunk_b = constant 9.9 : f64 // used for the second "gt"
        %answers = graphblas.matrix_select %sparse_tensor, %thunk_a, %thunk_b { selectors = ["triu", "gt", "tril", "gt"] } : tensor<?x?xf64, #CSR64>, f64, f64 to tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins AnyTensor:$input, Variadic<AnyType>:$thunks, StrArrayAttr:$selectors);
    let results = (outs Variadic<AnyTensor>:$outputs);
    
    let assemblyFormat = [{
           $input (`,` $thunks^)? attr-dict `:` type($input) (`,` type($thunks)^)? `to` type($outputs)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixReduceToVectorOp : GraphBLAS_Op<"matrix_reduce_to_vector", [NoSideEffect]> {
    let summary = "matrix reduce to vector operation";
    let description = [{
        Reduces a sparse tensor to a vector according to the given aggregator.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The resulting sparse vector's element type must be the same as the element type of the input tensor.
        If the axis attribute is 0, the input tensor will be reduced column-wise, so the resulting
        vector's size must be the  number of rows in the input tensor. 
        If the axis attribute is 1, the input tensor will be reduced row-wise, so the resulting
        vector's size must be the  number of columns in the input tensor. 

        Example:
        ```mlir
        %vec1 = graphblas.matrix_reduce_to_vector %matrix { aggregator = "plus", axis = 0 } : tensor<7x9xi32, #CSR64> to tensor<9xi32, #SparseVec64>
        %vec2 = graphblas.matrix_reduce_to_vector %matrix { aggregator = "plus", axis = 1 } : tensor<7x9xi32, #CSR64> to tensor<7xi32, #SparseVec64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input, StrAttr:$aggregator, I64Attr:$axis);
    let results = (outs GraphBlasVectorOperand:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixReduceToScalarOp : GraphBLAS_Op<"matrix_reduce_to_scalar", [NoSideEffect]> {
    let summary = "matrix reduce to scalar operation";
    let description = [{
        Reduces a sparse matrix to a scalar according to the given aggregator.
        The given tensor must have a CSR sparsity or a CSC sparsity.  The resulting
        scalar's type will depend on the type of the input tensor.

        Example:
        ```mlir
        %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "plus" } : tensor<?x?xi64, #CSR64> to i64
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input, StrAttr:$aggregator);
    let results = (outs AnyType:$output);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixReduceToScalarGenericOp : GraphBLAS_Op<"matrix_reduce_to_scalar_generic", [NoSideEffect]> {
    let summary = "matrix reduce to scalar generic operation";
    let description = [{
        Reduces a sparse tensor to a scalar according to the given aggregator
        block.  The given sparse tensor must be a matrix, i.e. have rank 2.  The given
        tensor must have a CSR sparsity or a CSC sparsity.  The resulting scalar's type
        will depend on the type of the input tensor.

        Example:
        ```mlir
          %answer = graphblas.matrix_reduce_to_scalar_generic %sparse_tensor : tensor<?x?xi64, #CSR64> to i64 {
            ^bb0(%a : i64, %b : i64):
              %result = std.addi %a, %b : i64
              graphblas.yield agg %result : i64
        }        
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ApplyOp : GraphBLAS_Op<"apply", [NoSideEffect]> {
    let summary = "tensor apply operation";
    let description = [{
        Applies an operator to all elements of the given sparse tensor.  How
        the thunk is used depends on the chosen operator.  Currently, the only valid
        operator is "min".  The given sparse tensor must either be a CSR matrix, CSC
        matrix, or a sparse vector.

        Example:
        ```mlir
        %thunk = constant 100 : i64
        %matrix_answer = 
          graphblas.apply %sparse_matrix, %thunk { apply_operator = "min" } : 
            (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>
        %vector_answer = 
          graphblas.apply %sparse_vector, %thunk { apply_operator = "min" } : 
            (tensor<?xi64, #SparseVec64>, i64) to tensor<?xi64, #SparseVec64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input, AnyType:$thunk, StrAttr:$apply_operator);
    let results = (outs AnyTensor:$output);
    
    let assemblyFormat = [{
           $input `,` $thunk attr-dict `:` `(` type($input) `,` type($thunk) `)` `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ApplyGenericOp : GraphBLAS_Op<"apply_generic", [NoSideEffect]> {
    let summary = "generic tensor apply operation";
    let description = [{
        Applies an arbitrary transformation to every element of a matrix or vector.
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs AnyTensor:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);
    
    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyOp : GraphBLAS_Op<"matrix_multiply", [NoSideEffect]> {
    let summary = "matrix multiply operation with an optional structural mask";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional
        structural mask.  The semiring must be a string of the form "<ADD_NAME>_<MUL_NAME>",
        e.g. "plus_times". The options for "<ADD_NAME>" are "plus", "any", and "min". The
        options for "<MUL_NAME>" are "pair", "times", "plus", "first", and "second". The
        given sparse tensors must be a matrix (i.e. rank 2) or a vector (i.e. rank 1).
        If the first input is a matrix, it must be CSR format. If the second input
        is a matrix, it must be CSC format.  Matrix times vector will return a vector.
        Vector times matrix will return a vector.  Matrix times matrix will return
        a CSR matrix.  The mask (if provided) must be the same format as the returned
        object.  This operation also accepts an optional region that specifies
        element-wise postprocessing to be done on the result of the matrix multiplication.
        The region must use `graphblas.yield` to indicate the result of the element-wise
        postprocessing.

        No Mask Example:
        ```mlir
          %answer = 
            graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : 
              (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>
        ```
        
        Mask Example:
        ```mlir
          %answer = 
            graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_times" } : 
              (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        ```

        Body Example:
        ```mlir
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : 
          (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
            ^bb0(%value: f64):
                %result = std.mulf %value, %value: f64
                graphblas.yield %result : f64
        }
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, 
                         GraphBlasMatrixOrVectorOperand:$b, Optional<AnyTensor>:$mask, StrAttr:$semiring);
    let results = (outs AnyType:$output);
    let regions = (region AnyRegion:$body);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) ($body^)?
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyGenericOp : GraphBLAS_Op<"matrix_multiply_generic", [NoSideEffect]> {
    let summary = "generic matrix multiply operation with an optional structural mask";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional
        structural mask.  The given sparse tensors must be a matrix, i.e. have rank 2.
        The first input tensors must be CSR format, while the second input tensor must
        be CSC format.  The mask (if provided) must be CSR format.
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, GraphBlasMatrixOrVectorOperand:$b, Optional<AnyTensor>:$mask);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);
    
    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyReduceToScalarGenericOp : 
  GraphBLAS_Op<"matrix_multiply_reduce_to_scalar_generic", [NoSideEffect]> {
    let summary = "matrix multiply followed by reduction to a scalar with an optional structural mask";
    let description = [{
        Performs a matrix multiply followed by a reduction to scalar.  Supports
        same extension blocks as matrix_multiply_generic, and also requires binary
        aggregation block (aggregation assumes same identity as semiring add).

        The given sparse tensors must be a matrix, i.e., have rank 2.  The first
        input tensors must be CSR format, while the second input tensor must be CSC
        format.  The mask (if provided) must be CSR format.
    }];

    let arguments = (ins GraphBlasMatrixOperand:$a, GraphBlasMatrixOperand:$b, Optional<AnyTensor>:$mask);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
      $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_UnionOp : GraphBLAS_Op<"union", [NoSideEffect]> {
    let summary = "element-wise union operation";
    let description = [{
        Performs an element-wise union between two matrices or two vectors.
        The resulting sparse structure with be the union of the two input structures.
        When either object has a non-overlapping element, it is copied to the output.
        When both objects have an overlapping element in a cell, an operation combines the result.

        Example:
        ```mlir
        %combined = graphblas.union %A, %B { union_operator = "plus" } : (
            tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, 
                         GraphBlasMatrixOrVectorOperand:$b, StrAttr:$union_operator);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_IntersectOp : GraphBLAS_Op<"intersect", [NoSideEffect]> {
    let summary = "element-wise intersection operation";
    let description = [{
        Performs an element-wise intersection between two matrices or two vectors.
        The resulting sparse structure with be the intersection of the two input structures.
        Overlapping elements are combined using an operation passed in.

        Example:
        ```mlir
        %combined = graphblas.intersect %A, %B { intersect_operator = "mult" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, 
                         GraphBlasMatrixOrVectorOperand:$b, StrAttr:$intersect_operator);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_UpdateOp : GraphBLAS_Op<"update", []> {
    let summary = "update operation handling accumulation, mask, and replacement";
    let description = [{
        Updates the output tensor based on the input and desired accumulation,
        mask, and replacement.  This returns zero values and modifies the output in
        place.  The only supported accumulate operator is "plus".  The given tensors
        must be sparse.

        Simple Accumulation Example:
        ```mlir
        graphblas.update %other_vec -> %vec { accumulate_operator = "plus" } : 
          tensor<?xi64, #SparseVec64> -> tensor<?xi64, #SparseVec64>
        ```

        Complemented Mask + Accumulate + Replace Example:
        ```mlir
        graphblas.update %other_vec -> %vec(%mask) { accumulate_operator = "plus", replace = true, mask_complement = true } : 
          tensor<?xi64, #SparseVec64> -> tensor<?xi64, #SparseVec64>(tensor<?xi64, #SparseVec64>)
        ```
    }];

    let arguments = (ins
     GraphBlasMatrixOrVectorOperand:$input,
     GraphBlasMatrixOrVectorOperand:$output,
     Optional<AnyTensor>:$mask, OptionalAttr<StrAttr>:$accumulate_operator,
     DefaultValuedAttr<BoolAttr, "false">:$replace, DefaultValuedAttr<BoolAttr, "false">:$mask_complement);
    let results = (outs Index:$fake_output);
    
    let assemblyFormat = [{
           $input `->` $output (`(` $mask^ `)`)? attr-dict `:` type($input) `->` type($output) (`(` type($mask)^ `)`)?
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_EqualOp : GraphBLAS_Op<"equal", [NoSideEffect]> {
    let summary = "Equality checking operation for vectors and matrices";
    let description = [{
        Performs an equality check.  The given tensors must be sparse.  Checks
        equality of rank and size of tensors, as well as values and structure.

        Example:
        ```mlir
        %answer = graphblas.equal %vec, %other_vec : tensor<?xi64, #SparseVec64>, tensor<?xi64, #SparseVec64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, GraphBlasMatrixOrVectorOperand:$b);
    let results = (outs I1:$output);
    
    let assemblyFormat = [{
           $a `,` $b attr-dict `:` type($a) `,` type($b)
    }];
  
    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_VectorArgMinMaxOp : GraphBLAS_Op<"vector_argminmax", [NoSideEffect]> {
    let summary = "vector argmin/argmax operation";
    let description = [{
        Determines the argmin or argmax of a vector depending on the value of
        the minmax string attribute, which may only take on the value of "min" or
        "max".  If there are multiple values that can be the argmin or argmax, an
        arbitrary one is chosen from them.  The given sparse vector must have rank 1
        and must be sparse.

        Example:
        ```mlir
        graphblas.vector_argminmax %vec { minmax = "min" } : tensor<?xi64, #SparseVec64>
        ```

    }];

    let arguments = (ins GraphBlasVectorOperand:$vec, StrAttr:$minmax);
    let results = (outs Index:$output);
    
    let assemblyFormat = [{
           $vec attr-dict `:` type($vec)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_VectorArgMinOp : GraphBLAS_Op<"vector_argmin", [NoSideEffect]> {
    let summary = "vector argmin operation";
    let description = [{
        Determines the argmin of a vector.  If there are multiple values that
        can be the argmin, an arbitrary one is chosen from them.  The given
        sparse vector must have rank 1 and must be sparse.

        Example:
        ```mlir
        graphblas.vector_argmin %vec : tensor<?xi64, #SparseVec64>
        ```

    }];

    let arguments = (ins GraphBlasVectorOperand:$vec);
    let results = (outs Index:$output);
    
    let assemblyFormat = [{
           $vec attr-dict `:` type($vec)
    }];
  
    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_VectorArgMaxOp : GraphBLAS_Op<"vector_argmax", [NoSideEffect]> {
    let summary = "vector argmax operation";
    let description = [{
        Determaxes the argmax of a vector.  If there are multiple values that
        can be the argmax, an arbitrary one is chosen from them.  The given
        sparse vector must have rank 1 and must be sparse.

        Example:
        ```mlir
        graphblas.vector_argmax %vec : tensor<?xi64, #SparseVec64>
        ```

    }];

    let arguments = (ins GraphBlasVectorOperand:$vec);
    let results = (outs Index:$output);
    
    let assemblyFormat = [{
           $vec attr-dict `:` type($vec)
    }];

    let verifier = [{ return ::verify(*this); }];
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
def YIELD_AGG_IDENTITY   : I64EnumAttrCase<"AGG_IDENTITY", 10, "agg_identity">;
def YIELD_AGG            : I64EnumAttrCase<"AGG", 11, "agg">;

def YieldKindAttr : I64EnumAttr<
    "YieldKind", "",
    [YIELD_TRANSFORM_IN_A, YIELD_TRANSFORM_IN_B, YIELD_TRANSFORM_OUT,
     YIELD_SELECT_IN_A,    YIELD_SELECT_IN_B,    YIELD_SELECT_OUT,
     YIELD_ADD_IDENTITY,   YIELD_ADD,
     YIELD_MULT_IDENTITY,  YIELD_MULT, 
     YIELD_AGG_IDENTITY,   YIELD_AGG]
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

def GraphBLAS_CommentOp : GraphBLAS_Op<"comment", []> {
    let summary = "graphblas comment operation";
    let description = [{
        `graphblas.comment` is used to add comments to MLIR assembly.
        It lowers into a no-op.

        ```mlir
        graphblas.comment { comment = "here is a comment!" } 
        ```
    }];
    
    let arguments = (ins StrAttr:$comment);

    let assemblyFormat = [{
           attr-dict
    }];
}

#endif // GRAPHBLAS_OPS
