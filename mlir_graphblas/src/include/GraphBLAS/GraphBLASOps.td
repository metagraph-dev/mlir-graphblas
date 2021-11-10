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
    let summary = "Return the size of a sparse vector.";
    let description = [{
        Returns the size of a vector.

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
    let summary = "Returns the number of rows in a matrix.";
    let description = [{
        Return the return the number of rows in a CSR or CSC matrix.

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
    let summary = "Returns the number of columns in a matrix.";
    let description = [{
        Return the return the number of columns in a CSR or CSC matrix.

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
    let summary = "Returns the number of values in a sparse tensor.";
    let description = [{
        Returns the number of values in a CSC matrix, CSR matrix, or sparse vector.

        Example:
        ```mlir
        %csr_nnz = graphblas.num_vals %csr_matrix : tensor<?x?xf64, #CSR64>
        %vector_nnz = graphblas.num_vals %sparse_vector : tensor<?xf64, #CV64>
        %csc_nnz = graphblas.num_vals %csc_matrix : tensor<?x?xf64, #CSC64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
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
    let summary = "Returns a duplicate of the input sparse tensor.";
    let description = [{
        Returns a duplicate copy of the input CSC matrix, CSR matrix, or sparse tensor.

        Example:
        ```mlir
          %B = graphblas.dup %A : tensor<?x?xf64, #CSR64>
          %new_vec = graphblas.dup %vec : tensor<?xf64, #CV64>
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
    let summary = "Converts graph storage layout.";
    let description = [{
        Rewrite the contents of a sparse tensor to change it from CSR to CSC, or vice versa.

        Vacuous conversions (e.g. CSC → CSC or CSR → CSR) are equivalent to no-ops and are removed by the `--graphblas-lower` pass.

        Example:
        ```mlir
          %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input);
    let results = (outs GraphBlasMatrixOperand:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_CastOp : GraphBLAS_Op<"cast", [NoSideEffect]> {
    let summary = "Changes graph storage parameters: dtype and bitwidths.";
    let description = [{
        Rewrite the contents of a sparse tensor to use a new dtype or a new pointer or index bitwidth.
        Layout changes (ex. CSR->CSC) are not supported by this operation. Use `convert_layout` instead.

        Example:
        ```mlir
          %a_int = graphblas.cast %a : tensor<?x?xf64, #CSR64> to tensor<?x?xi32, #CSR64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_TransposeOp : GraphBLAS_Op<"transpose", [NoSideEffect]> {
    let summary = "Transpose operation.";
    let description = [{
        Returns a new sparse matrix that's the transpose of the input matrix.
        The given sparse tensor must be a matrix, i.e. have rank 2.
        The given tensor must have a CSR sparsity or a CSC sparsity.
        The output type must be CSR or CSC.

        Note that the behavior of this op differs depending on the sparse
        encoding of the specified output tensor type.

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

def GraphBLAS_SelectOp : GraphBLAS_Op<"select", [NoSideEffect]> {
    let summary = "Select operation.";
    let description = [{
        Returns a new sparse tensor with a subset of elements from the given tensor.

        The elements included in the resulting sparse tensor vary depending on the selectors given
        (one of "triu", "tril", "ge", "gt", or "probability").

        Some selectors, e.g. "gt", require a thunk.
        Some selectors, e.g. "probability" require a thunk and random number generator context.

        The given tensor must be a sparse vector or a matrix with CSR or CSC sparsity.
        The resulting sparse tensors will have the same encoding as the input tensor.

        Selector Example:
        ```mlir
        %result = graphblas.select %sparse_tensor { selector = "triu" } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
        ```

        Selector with Thunk Example:
        ```mlir
        %thunk = constant 0.0 : f64
        %result = graphblas.select %sparse_tensor, %thunk { selector = "gt" } : tensor<?x?xf64, #CSR64>, f64 to tensor<?x?xf64, #CSR64>
        ```

        Random Selector Example:
        ```mlir
        %percentage = constant 0.5 : f64
        %result = graphblas.select %sparse_tensor, %percentage, %rng_context { selector = "probability" } : tensor<?x?xf64, #CSR64>, f64, !llvm.ptr<i8> to tensor<?x?xf64, #CSR64>
        ```
    }];

    let arguments = (ins
      GraphBlasMatrixOrVectorOperand:$input,
      Variadic<AnyType>:$thunks,
      StrAttr:$selector);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $input (`,` $thunks^)? attr-dict `:` type($input) (`,` type($thunks)^)? `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_SelectGenericOp : GraphBLAS_Op<"select_generic", [NoSideEffect]> {
    let summary = "Generic select operation.";
    let description = [{
        Returns a new sparse tensor with a subset of elements from the given tensor.

        The given tensor must be a sparse vector or a matrix with CSR or CSC sparsity.
        The resulting sparse tensors will have the same encoding as the input tensor.

        Selector Example (upper triangle):
        ```mlir
        %result = graphblas.select_generic %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64> {
          ^bb0(%val: f64, %row: index, %col: index):
            %result = arith.cmpi "ugt", %col, %row : index
            return %result : i1
        }
        ```

        Selector with Thunk Example:
        ```mlir
        %thunk = constant 0.0 : f64
        %result = graphblas.select_generic %sparse_tensor : tensor<?x?xf64, #CSR64>, f64 to tensor<?x?xf64, #CSR64> {
          ^bb0(%val: f64):
            %result = arith.cmpf "olt", %val, %thunk : f64
            return %result : i1
        }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ReduceToVectorOp : GraphBLAS_Op<"reduce_to_vector", [NoSideEffect]> {
    let summary = "Matrix reduce to vector operation.";
    let description = [{
        Reduces a CSR or CSC matrix to a vector according to the given aggregator and axis.

        The resulting sparse vector's element type varies according to the given aggregator.
        The supported aggregators are "plus", "count", "argmin", and "argmax". Aggregating via
        "plus" and "count" will cause the output's element type to match that of the input
        tensor. Aggregating via "argmin" and "argmax" will cause the output's element to be a
        64-bit integer.

        If the axis attribute is 0, the input tensor will be reduced column-wise, so the
        resulting vector's size must be the number of columns in the input tensor.

        If the axis attribute is 1, the input tensor will be reduced row-wise, so the resulting
        vector's size must be the number of rows in the input tensor.

        Example:
        ```mlir
        %vec1 = graphblas.reduce_to_vector %matrix_1 { aggregator = "plus", axis = 0 } : tensor<7x9xf16, #CSR64> to tensor<9xf16, #CV64>
        %vec2 = graphblas.reduce_to_vector %matrix_2 { aggregator = "count", axis = 1 } : tensor<7x9xf16, #CSR64> to tensor<7xf16, #CV64>
        %vec3 = graphblas.reduce_to_vector %matrix_3 { aggregator = "argmin", axis = 1 } : tensor<7x9xf32, #CSR64> to tensor<7xi64, #CV64>
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input, StrAttr:$aggregator, I64Attr:$axis);
    let results = (outs GraphBlasVectorOperand:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ReduceToVectorGenericOp : GraphBLAS_Op<"reduce_to_vector_generic", [NoSideEffect]> {
    let summary = "Matrix reduce to vector generic operation.";
    let description = [{
        Reduces a CSR or CSC matrix to a vector according to the given axis using the
        specified aggregator block.

        If the axis attribute is 0, the input tensor will be reduced column-wise, so the
        resulting vector's size must be the number of columns in the input tensor.

        If the axis attribute is 1, the input tensor will be reduced row-wise, so the resulting
        vector's size must be the number of rows in the input tensor.

        Example:
        ```mlir
        %vec1 = graphblas.reduce_to_vector %matrix_1 { axis = 0 } : tensor<7x9xf16, #CSR64> to tensor<9xf16, #CV64> {
              graphblas.yield agg_identity %cf0 : f16
          },  {
            ^bb0(%a : f16, %b : f16):
              %result = arith.addf %a, %b : f16
              graphblas.yield agg %result : f16
          }
        %vec2 = graphblas.reduce_to_vector %matrix_2 { axis = 1 } : tensor<7x9xi64, #CSR64> to tensor<7xi64, #CV64> {
              graphblas.yield agg_identity %ci0 : i64
          },  {
            ^bb0(%a : i64, %b : i64):
              %cmp = arith.cmpi "slt", %a, %b : i64
              %result = select %cmp, %a, %b : i64
              graphblas.yield agg %result : i64
          }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$input, I64Attr:$axis);
    let results = (outs GraphBlasVectorOperand:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ReduceToScalarOp : GraphBLAS_Op<"reduce_to_scalar", [NoSideEffect]> {
    let summary = "Reduce to scalar operation.";
    let description = [{
        Reduces a sparse tensor (CSR matrix, CSC matrix, or sparse vector) to a scalar
        according to the given aggregator. Matrices must have a CSR sparsity or a CSC
        sparsity.

        The resulting scalar's type will depend on the type of the input tensor.

        The supported aggregators are "plus", "count", "argmin", and "argmax".

        When using the aggregators "argmin" and "argmax", the output must be a 64-bit integer,
        and the input type must be a vector.

        Example:
        ```mlir
        %answer_1 = graphblas.reduce_to_scalar %sparse_matrix { aggregator = "plus" } : tensor<?x?xf32, #CSR64> to f32
        %answer_2 = graphblas.reduce_to_scalar %sparse_vector { aggregator = "argmax" } : tensor<?x?xf64, #CSR64> to i64
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input, StrAttr:$aggregator);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ReduceToScalarGenericOp : GraphBLAS_Op<"reduce_to_scalar_generic", [NoSideEffect]> {
    let summary = "Reduce to scalar generic operation.";
    let description = [{
        Reduces a sparse tensor to a scalar according to the given aggregator
        block.  If the tensor is a matrix, it must have a CSR sparsity or a CSC sparsity.
        The resulting scalar's type will depend on the type of the input tensor.
        Only one aggregator block is allowed.

        Example:
        ```mlir
          %ci0 = arith.constant 0 : i64
          %answer = graphblas.reduce_to_scalar_generic %sparse_vector : tensor<?xi64, #CV64> to i64 {
              graphblas.yield agg_identity %ci0 : i64
          },  {
            ^bb0(%a : i64, %b : i64):
              %result = arith.addi %a, %b : i64
              graphblas.yield agg %result : i64
          }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ApplyOp : GraphBLAS_Op<"apply", [NoSideEffect]> {
    let summary = "Tensor element-wise apply operation.";
    let description = [{
        Applies in an element-wise fashion the function indicated by the ``apply_operator``
        attribute to each element of the given sparse tensor. The operator can be unary or
        binary. Binary operators require a thunk. The supported binary operators are "min",
        "div", "first", and "second". Unary operators cannot take a thunk. The supported
        unary operators are "abs", "minv" (i.e. multiplicative inverse or *1/x*),
        "ainv" (i.e. additive inverse or *-x*), and "identity".

        The given sparse tensor must either be a CSR matrix, CSC matrix, or a sparse vector.

        Using "minv" with integer types uses signed integer division and rounds towards
        zero. For example, *minv(-2) == 1 / -2 == 0*.

        Some binary operators, e.g. "div", are not symmetric. The sparse tensor and thunk
        should be given in the order they should be given to the binary operator. For
        example, to divide every element of a matrix by 2, use the following:

        ```mlir
        %thunk = constant 2 : i64
        %matrix_answer = graphblas.apply %sparse_matrix, %thunk { apply_operator = "div" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>
        ```

        As another example, to divide 10 by each element of a sparse vector, use the
        following:


        ```mlir
        %thunk = constant 10 : i64
        %vector_answer = graphblas.apply %thunk, %sparse_vector { apply_operator = "div" } : (i64, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
        ```

        Note that the application only takes place for elements that are present in the
        matrix. Thus, the operation will not apply when the values are missing in the
        tensor. For example, *1.0 / [ _ , 2.0,  _ ] == [ _ , 0.5,  _ ]*.

        Note that using the "identity" operator does not create a copy of the input tensor.

        The shape of the output tensor will match that of the input tensor.


        Example:
        ```mlir
        %thunk = constant 100 : i64
        %matrix_answer = graphblas.apply %sparse_matrix, %thunk { apply_operator = "min" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>
        %vector_answer = graphblas.apply %sparse_vector { apply_operator = "abs" } : (tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>
        ```
    }];

    let arguments = (ins AnyType:$left, Optional<AnyType>:$right, StrAttr:$apply_operator);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $left (`,` $right^)? attr-dict `:` `(` type($left) (`,` type($right)^)? `)` `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_ApplyGenericOp : GraphBLAS_Op<"apply_generic", [NoSideEffect]> {
    let summary = "Generic tensor apply operation.";
    let description = [{
        Applies an arbitrary transformation to every element of a CSR or CSC matrix or sparse vector
        according to the given transformation block. Only one transformation block is allowed.

        Example:
        ```mlir
        %answer = graphblas.apply_generic %m : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64> {
          ^bb0(%val: f64):
            %result = arith.negf %val : f64
            graphblas.yield transform_out %result : f64
        }
        ```

        ```mlir
        %thunk = constant 0.0 : f64
        %answer = graphblas.apply_generic %sparse_tensor : tensor<?xf64, #CV64> to tensor<?xf64, #CV64> {
          ^bb0(%val: f64):
            %pick = cmpf olt, %val, %thunk : f64
            %result = select %pick, %val, %thunk : f64
            graphblas.yield transform_out %result : f64
        }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyOp : GraphBLAS_Op<"matrix_multiply", [NoSideEffect]> {
    let summary = "Matrix multiply operation with an optional structural mask.";
    let description = [{
        Performs a matrix multiply according to the given semiring and optional structural
        mask. The structural mask specifies which values in the output are to be computed
        and thus must have the same shape as the expected output.

        The semiring must be a string of the form "<ADD_NAME>_<MUL_NAME>", e.g. "plus_times".
        The options for "<ADD_NAME>" are "plus", "any", and "min". The options for "<MUL_NAME>"
        are "pair", "times", "plus", "firsti", "secondi", "overlapi", "first", and "second".

        If the first input is a matrix, it must be CSR format. If the second input is a matrix,
        it must be CSC format. Matrix times vector will return a vector. Vector times matrix
        will return a vector. Matrix times matrix will return a CSR matrix. Vector times
        vector will return a scalar.

        The mask (if provided) must be the same format as the returned object. There's an
        optional boolean `mask_complement` attribute (which has a default value of `false`)
        that will make the op use the complement of the mask.

        It should be noted that masks are not allowed for vector times vector multiplication.

        Examples:
        ```mlir
        %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>
        ```
        ```mlir
        %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "min_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
        ```
        ```mlir
        %answer = graphblas.matrix_multiply %mat, %vec, %mask { semiring = "any_first", mask_complement = true } : (tensor<?x?xf64, #CSR64>, tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
        ```
        ```mlir
        %answer = graphblas.matrix_multiply %vec, %mat { semiring = "min_second", mask_complement = true } : (tensor<?xf64, #CV64>, tensor<?x?xf64, #CSC64>) to tensor<?xf64, #CV64>
        ```
        ```mlir
        %answer = graphblas.matrix_multiply %vecA, %vecB { semiring = "any_pair" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>
        ```

    }];

    let arguments = (ins
     GraphBlasMatrixOrVectorOperand:$a,
     GraphBlasMatrixOrVectorOperand:$b,
     Optional<GraphBlasMatrixOrVectorOperand>:$mask,
     StrAttr:$semiring,
     DefaultValuedAttr<BoolAttr, "false">:$mask_complement);
    let results = (outs AnyType:$output);

    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyGenericOp : GraphBLAS_Op<"matrix_multiply_generic", [NoSideEffect]> {
    let summary = "Generic matrix multiply operation with an optional structural mask.";
    let description = [{

        This op performs computations over the two sparse tensor inputs using the same access pattern
        as a conventional matrix multiply with the given blocks allowing us to modify the behavior.

        This op takes as input 2 sparse tensor inputs and an optional structural mask.

        Additionally, this op takes 3 required blocks (we'll refer to them as the  "mult",
        "add", and "add_identity" blocks) and 1 optional block (we'll refer to it as the "transform_out"
        block).

        In a conventional matrix multiply where the multiplication between two elements takes place, this
        op instead performs the behavior specified in the "mult" block. The "mult" block takes two scalar
        arguments and uses the ``graphblas.yield`` terminator op (with the "kind" attribute set to "mult")
        to return the result of the element-wise computation.

        In a conventional matrix multiply where the summation over the products from the element-wise
        multiplications take place, this op instead performs the behavior specified in the "add" block
        to aggregate the results. The "add" block takes two scalar arguments (the first representing the
        current aggregation and the second representing the next value to be aggregated) and uses the
        ``graphblas.yield`` terminator op (with the "kind" attribute set to "add") to return the result
        of the current aggregation.

        The aggregation taking place in the "add" block requires an initial value (for conventional
        matrix multiplication, this value is zero). Using the ``graphblas.yield`` terminator op (with
        the "kind" attribute set to "add_identity") in the "add_identity" block let's us specify this
        initial value. This block takes no arguments.

        This op additionally takes an optional "transform_out" block that performs an element-wise
        transformation on the final aggregated values from the "add" block. The "transform_out" block
        takes one argument and returns one value via the ``graphblas.yield`` terminator op (with
        the "kind" attribute set to "transform_out").

        The mask (if provided) must be the same format as the returned object. There's an optional
        boolean `mask_complement` attribute (which has a default value of `false`) that will make
        the op use the complement of the mask.


        Example:
        ```mlir
          %answer = graphblas.matrix_multiply_generic %a, %b {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
              ^bb0:
                   %identity = constant 0.0 : f64
                   graphblas.yield add_identity %identity : f64
            },{
              ^bb0(%add_a: f64, %add_b: f64):
                   %add_result = arith.addf %add_a, %add_b : f64
                   graphblas.yield add %add_result : f64
            },{
              ^bb0(%mult_a: f64, %mult_b: f64):
                   %mult_result = arith.mulf %mult_a, %mult_b : f64
                   graphblas.yield mult %mult_result : f64
            },{
               ^bb0(%value: f64):
                   %result = arith.addf %value, %c100_f64: f64
                   graphblas.yield transform_out %result : f64
            }
        ```
    }];

    let arguments = (ins
      GraphBlasMatrixOrVectorOperand:$a,
      GraphBlasMatrixOrVectorOperand:$b,
      Optional<GraphBlasMatrixOrVectorOperand>:$mask,
      BoolAttr:$mask_complement);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixMultiplyReduceToScalarGenericOp :
      GraphBLAS_Op<"matrix_multiply_reduce_to_scalar_generic", [NoSideEffect]> {
    let summary = "Matrix multiply followed by reduction to a scalar with an optional structural mask.";
    let description = [{
        Performs a matrix multiply followed by a reduction to scalar. Supports the same extension blocks
        as `graphblas.matrix_multiply_generic` and also requires a binary aggregation block and aggregation
        identity block. These latter two blocks are used for reducing the result of the matrix multiply
        to a scalar.



        Example:
        ```mlir
        %answer = graphblas.matrix_multiply_reduce_to_scalar_generic %a, %b : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to f64 {
            ^bb0:
                %identity = constant 0.0 : f64
                graphblas.yield add_identity %identity : f64
        },{
            ^bb0(%add_a: f64, %add_b: f64):
                %add_result = arith.addf %add_a, %add_b : f64
                graphblas.yield add %add_result : f64
        },{
            ^bb0(%mult_a: f64, %mult_b: f64):
                %mult_result = arith.mulf %mult_a, %mult_b : f64
                graphblas.yield mult %mult_result : f64
        },{
            %agg_identity = constant 0.0 : f64
            graphblas.yield agg_identity %agg_identity : f64
        },{
            ^bb0(%lhs: f64, %rhs: f64):
                %agg_result = arith.addf %lhs, %rhs: f64
                graphblas.yield agg %agg_result : f64
        }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOperand:$a, GraphBlasMatrixOperand:$b, Optional<GraphBlasMatrixOperand>:$mask);
    let results = (outs AnyType:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
      $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_UnionOp : GraphBLAS_Op<"union", [NoSideEffect]> {
    let summary = "Element-wise union operation.";
    let description = [{
        Performs an element-wise union between two matrices or two vectors.
        The resulting sparse structure will be the union of the two input structures.
        When either object has a non-overlapping element, it is copied to the output.
        When both objects have an overlapping element in a cell, an operation combines the result.
        The supported values for this operation are "plus", "times", "min", "max", "first",
        and "second".

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

def GraphBLAS_UnionGenericOp : GraphBLAS_Op<"union_generic", [NoSideEffect]> {
    let summary = "Element-wise union operation.";
    let description = [{
        Performs an element-wise union between two matrices or two vectors.
        The resulting sparse structure will be the union of the two input structures.
        When either object has a non-overlapping element, it is copied to the output.
        When both objects have an overlapping element in a cell, an operation combines
        the result according to the given binary operator.

        Example:
        ```mlir
          %combined = graphblas.union_generic %A, %B : (
            tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64) to tensor<?x?xf64, #CSR64> {
            ^bb0(%a : f64, %b : f64):
              %result = arith.addf %a, %b : f64
              graphblas.yield mult %result : f64
          }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a,
                         GraphBlasMatrixOrVectorOperand:$b);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_IntersectOp : GraphBLAS_Op<"intersect", [NoSideEffect]> {
    let summary = "Element-wise intersection operation.";
    let description = [{
        Performs an element-wise intersection between two matrices or two vectors.
        The resulting sparse tensor will be the intersection of the two input structures.
        When both objects have an overlapping element in a cell, an operation
        combines the result according to the given operator. The supported
        values for the intersect operator are "plus", "minus", "times", "div",
        "min", "max", "first", and "second".

        Example:
        ```mlir
        %combined = graphblas.intersect %A, %B { intersect_operator = "mult" } : (
            tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>
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

def GraphBLAS_IntersectGenericOp : GraphBLAS_Op<"intersect_generic", [NoSideEffect]> {
    let summary = "Element-wise intersection operation.";
    let description = [{
        Performs an element-wise intersection between two matrices or two vectors.
        The resulting sparse structure will be the union of the two input structures.
        When both objects have an overlapping element in a cell, an operation combines
        the result according to the given binary operator.

        Example:
        ```mlir
          %combined = graphblas.intersect_generic %A, %B : (
            tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64) to tensor<?x?xf64, #CSR64> {
            ^bb0(%a : f64, %b : f64):
              %result = arith.addf %a, %b : f64
              graphblas.yield mult %result : f64
          }
        ```
    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a,
                         GraphBlasMatrixOrVectorOperand:$b);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output) $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_UpdateOp : GraphBLAS_Op<"update", []> {
    let summary = "Update operation handling accumulation, mask, and replacement.";
    let description = [{
        Updates the output tensor based on the input and desired accumulation,
        mask, and replacement.  This returns zero values and modifies the output in
        place.  The supported accumulate operators are "plus", "times", "min", and "max".
        The given tensors must be sparse.

        There's an optional boolean *mask_complement* attribute (which has a default
        value of *false*) that will make the op use the complement of the mask.

        Example:
        ```mlir
        graphblas.update %other_mat -> %mat(%mask) { accumulate_operator = "times", replace = true, mask_complement = true } : tensor<?x?xi64, #CSR64> -> tensor<?x?xi64, #CSR64>(tensor<?x?xi64, #CSR64>)
        ```
        ```mlir
        graphblas.update %other_vec -> %vec { accumulate_operator = "plus" } : tensor<?xi64, #CV64> -> tensor<?xi64, #CV64>
        ```
        ```mlir
        graphblas.update %other_mat -> %mat(%mask) { accumulate_operator = "max", replace = true } : tensor<?x?xi64, #CSC64> -> tensor<?x?xi64, #CSC64>(tensor<?x?xi64, #CSC64>)
        ```
    }];

    let arguments = (ins
     GraphBlasMatrixOrVectorOperand:$input,
     GraphBlasMatrixOrVectorOperand:$output,
     Optional<GraphBlasMatrixOrVectorOperand>:$mask,
     OptionalAttr<StrAttr>:$accumulate_operator,
     DefaultValuedAttr<BoolAttr, "false">:$replace,
     DefaultValuedAttr<BoolAttr, "false">:$mask_complement);
    let results = (outs Index:$fake_output);

    let assemblyFormat = [{
           $input `->` $output (`(` $mask^ `)`)? attr-dict `:` type($input) `->` type($output) (`(` type($mask)^ `)`)?
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_UpdateGenericOp : GraphBLAS_Op<"update_generic", []> {
    let summary = "Update operation handling accumulation, mask, and replacement.";
    let description = [{
        Updates the output tensor based on the input and desired accumulation,
        mask, and replacement.  This returns zero values and modifies the output in
        place. The given tensors must be sparse. Accumulation is performed based on
        the given accumulator block.

        There's an optional boolean *mask_complement* attribute (which has a default
        value of *false*) that will make the op use the complement of the mask.

        Example:
        ```mlir
        graphblas.update %other_mat -> %mat(%mask) { replace = true, mask_complement = true } :
            tensor<?x?xi64, #CSR64> -> tensor<?x?xf64, #CSR64>(tensor<?x?xf64, #CSR64>) {
            ^bb0(%a : f64, %b : f64):
              %result = arith.addf %a, %b : f64
              graphblas.yield accumulate %result : f64
        }
        ```
        ```mlir
        graphblas.update %other_vec -> %vec : tensor<?xi64, #CV64> -> tensor<?xi64, #CV64> {
            ^bb0(%a : i64, %b : i64):
              %result = arith.muli %a, %b : i64
              graphblas.yield accumulate %result : i64
        }
        ```
    }];

    let arguments = (ins
     GraphBlasMatrixOrVectorOperand:$input,
     GraphBlasMatrixOrVectorOperand:$output,
     Optional<GraphBlasMatrixOrVectorOperand>:$mask,
     DefaultValuedAttr<BoolAttr, "false">:$replace,
     DefaultValuedAttr<BoolAttr, "false">:$mask_complement);
    let results = (outs Index:$fake_output);
    let regions = (region VariadicRegion<SizedRegion<1>>:$extensions);

    let assemblyFormat = [{
           $input `->` $output (`(` $mask^ `)`)? attr-dict `:` type($input) `->` type($output) (`(` type($mask)^ `)`)? $extensions
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_EqualOp : GraphBLAS_Op<"equal", [NoSideEffect]> {
    let summary = "Equality checking operation for vectors and matrices.";
    let description = [{
        Performs an equality check. The given tensors must be sparse vectors, CSR matrices,
        or CSC matrices. Checks equality of rank and size of tensors, as well as values and
        structure. Returns a single boolean value.

        Example:
        ```mlir
        %answer = graphblas.equal %vec, %other_vec : tensor<?xi64, #CV64>, tensor<?xi64, #CV64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$a, GraphBlasMatrixOrVectorOperand:$b);
    let results = (outs I1:$output);

    let assemblyFormat = [{
           $a `,` $b attr-dict `:` type($a) `,` type($b)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_DiagOp : GraphBLAS_Op<"diag", [NoSideEffect]> {
    let summary = "Diagonal operation.";
    let description = [{
        When given a square CSR or CSC matrix, returns the diagonal
        as a sparse vector. When given a sparse vector, returns a
        square CSR or CSC matrix with the vector's values along the
        diagonal.

        Example:
        ```mlir
        %csr_matrix_answer = graphblas.diag %vec : tensor<?xi64, #CV64> to tensor<?x?xi64, #CSR64>
        %csc_matrix_answer = graphblas.diag %vec : tensor<?xi64, #CV64> to tensor<?x?xi64, #CSC64>
        %vector_answer = graphblas.diag %mat : tensor<?x?xi64, #CSR64> to tensor<?xi64, #CV64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOrVectorOperand:$input);
    let results = (outs GraphBlasMatrixOrVectorOperand:$output);

    let assemblyFormat = [{
           $input attr-dict `:` type($input) `to` type($output)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def GraphBLAS_MatrixSelectRandomOp : GraphBLAS_Op<"matrix_select_random", []> {
    let summary = "Random selection of elements from each row of matrix.";
    let description = [{
        Selects a random subset of up to *n* elements in each row of a CSR
        matrix.  If there are less than *n* elements in a row, all elements in the row
        are included in the output.

        An external function must be provided via the *choose_n* attribute to
        the op with the following signature:
        ```mlir
        func @my_choose_n(%context: !llvm.ptr<i8>,
                          %n: IndexType, %max_i: IndexType,
                          %output_indices: memref<?xIndexType>,
                          %row_values: memref<?xValueType)
        ```

        where *IndexType* corresponds to the index element type of the sparse
        tensor input and *ValueType* corresponds to the value element type of
        the sparse tensor input.  This external function selects *n* random
        indices from the interval *[0, max_i)* and writes them to the
        *output_indices* memref in increasing order.  If desired, the
        distribution of selected indices can be biased by the values in
        *row_values*, which will have length *max_i*.  A uniform choice
        function will ignore this last argument.

        The implementation of the *choose_n* function is not specified by this
        op because it will differ significantly depending on use case (uniform
        or weighted sampling) and desired execution target (serial, parallel,
        GPU, etc).

        Example:
        ```mlir
        %output = graphblas.matrix_select_random %a, %n, %rng_context { choose_n = @uniform_choose_n } : (tensor<?x?xf64, #CSR64>, i64, !llvm.ptr<i8>) to tensor<?x?xf64, #CSR64>
        ```

    }];

    let arguments = (ins GraphBlasMatrixOperand:$input,
                         AnyInteger:$n,
                         AnyType:$rng_context,
                         SymbolRefAttr:$choose_n);
    let results = (outs GraphBlasMatrixOperand:$output);

    let assemblyFormat = [{
           $input `,` $n `,` $rng_context attr-dict `:` `(` type($input) `,` type($n) `,` type($rng_context)  `)` `to` type($output)
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
def YIELD_MULT           : I64EnumAttrCase<"MULT", 8, "mult">;
def YIELD_AGG_IDENTITY   : I64EnumAttrCase<"AGG_IDENTITY", 9, "agg_identity">;
def YIELD_AGG            : I64EnumAttrCase<"AGG", 10, "agg">;
def YIELD_ACCUMULATE     : I64EnumAttrCase<"ACCUMULATE", 11, "accumulate">;

def YieldKindAttr : I64EnumAttr<
    "YieldKind", "",
    [YIELD_TRANSFORM_IN_A, YIELD_TRANSFORM_IN_B, YIELD_TRANSFORM_OUT,
     YIELD_SELECT_IN_A,    YIELD_SELECT_IN_B,    YIELD_SELECT_OUT,
     YIELD_ADD_IDENTITY,   YIELD_ADD, YIELD_MULT,
     YIELD_AGG_IDENTITY,   YIELD_AGG, YIELD_ACCUMULATE]
    > {
  let cppNamespace = "::mlir::graphblas";
}

def GraphBLAS_YieldOp : GraphBLAS_Op<"yield", [NoSideEffect, ReturnLike, Terminator]> {
    let summary = "GraphBLAS yield operation.";
    let description = [{
        `graphblas.yield` is a special terminator operation for blocks inside regions in
        several `graphblas` operations.  It returns a value to the enclosing op, with
        a meaning that depends on the op.

        Special terminator operation for blocks inside regions in several GraphBLAS dialect operations,
        e.g. `graphblas.matrix_multiply_generic`. It returns a value to the enclosing op,
        with a meaning that depends on the required "kind" attribute. It must be one of the following:

        - transform_in_a
        - transform_in_b
        - transform_out
        - select_in_a
        - select_in_b
        - select_out
        - add_identity
        - add
        - mult
        - accumulate

        Example:
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
    let summary = "Comment operation.";
    let description = [{
        `graphblas.comment` is intended to be a no-op and
        returns zero values. It merely contains a string
        attribute intended to hold code comments.

        Example:
        ```mlir
        graphblas.comment { comment = "here is a comment!" }
        ```
    }];

    let arguments = (ins StrAttr:$comment);

    let assemblyFormat = [{
           attr-dict
    }];
}

def GraphBLAS_PrintOp : GraphBLAS_Op<"print", []> {
    let summary = "Print operation.";
    let description = [{
        `graphblas.print` is used to pretty print values
        to stdout.
        This is intended to be used for debugging only.
        The strings attribute is a list of strings.
        This op is variadic and takes an arbitrary number of inputs.
        The printing alternates between printing the strings
        and the input values.

        ```mlir
        %c9_9_f32 = constant 9.9 : f32
        %c1_i32 = constant 1 : i32

        // prints "start 9.9 middle 1 end ".
        graphblas.print %c9_9_f32, %c1_i32 { strings = ["start ", " middle ", " end"] } : f32, i32

        // prints "start 9.9 middle   end  z y x ".
        graphblas.print %c9_9_f32 { strings = ["start ", " middle ", " end", " z", "y", "x"] } : f32

        // prints "start 9.9 1 1 1 1".
        graphblas.print %c9_9_f32, %c1_i32, %c1_i32, %c1_i32, %c1_i32 { strings = ["start "] } : f32, i32, i32, i32, i32

        // prints "9.9".
        graphblas.print %c9_9_f32 { strings = [] } : f32

        %dense_vec_fixed = arith.constant dense<[0.0, 10.0, 20.0, 0.0]> : tensor<4xf64>
        %dense_vec = tensor.cast %dense_vec_fixed : tensor<4xf64> to tensor<?xf64>
        %vec = sparse_tensor.convert %dense_vec : tensor<?xf64> to tensor<?xf64, #CV64>
        
        // prints "vec [0, 10, 20, 0]".
        graphblas.print %dense_vec_fixed { strings = ["vec "] } : tensor<4xf64>

        // prints "vec [0, 10, 20, 0]".
        graphblas.print %dense_vec { strings = ["vec "] } : tensor<?xf64>
      
        // prints "vec [_, 10, 20, _]".
        graphblas.print %vec { strings = ["vec "] } : tensor<?xf64, #CV64>

        %dense_mat_fixed = arith.constant dense<[
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0]
          ]> : tensor<2x4xf64>
        %dense_mat = tensor.cast %dense_mat_fixed : tensor<2x4xf64> to tensor<?x?xf64>
        %mat = sparse_tensor.convert %dense_mat : tensor<?x?xf64> to tensor<?x?xf64, #CSR64>
        %mat_csc = graphblas.convert_layout %mat : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>

        // prints "mat [
        //   [0, 1, 2, 0],
        //   [0, 0, 0, 3],
        // ]".
        graphblas.print %dense_mat_fixed { strings = ["mat "] } : tensor<2x4xf64>
  
        // prints "mat [
        //   [0, 1, 2, 0],
        //   [0, 0, 0, 3],
        // ]".
        graphblas.print %dense_mat { strings = ["mat "] } : tensor<?x?xf64>
        
        // prints "mat [
        //   [_, 1, 2, _],
        //   [_, _, _, 3],
        // ]".
        graphblas.print %mat { strings = ["mat "] } : tensor<?x?xf64, #CSR64>

        // prints "mat_csc [
        //   [_, 1, 2, _],
        //   [_, _, _, 3],
        // ]".
        graphblas.print %mat_csc { strings = ["mat_csc "] } : tensor<?x?xf64, #CSC64>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values, StrArrayAttr:$strings);

    let assemblyFormat = [{
        $values attr-dict `:` type($values)
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // GRAPHBLAS_OPS
