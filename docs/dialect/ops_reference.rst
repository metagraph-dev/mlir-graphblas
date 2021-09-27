
..    include:: <isonum.txt>

.. _graphblas_ops_reference:

GraphBLAS Ops
=============

The ``graphblas`` dialect describes standard sparse tensor operations that are
found in the `GraphBLAS spec`_.  The ops are not one-to-one equivalents of
GraphBLAS function calls in order to fit into MLIR's SSA requirements.

This document is not intended to be a tutorial and acts more as a reference manual for the
ops in the GraphBLAS dialect. For tutorials, see our :ref:`graphblas_dialect_tutorials`.

.. _GraphBLAS spec: http://people.eecs.berkeley.edu/~aydin/GraphBLAS_API_C_v13.pdf

Assumptions
-----------

Although the `sparse tensor encoding
<https://mlir.llvm.org/docs/Dialects/SparseTensorOps/#sparsetensorencodingattr>`_
in MLIR is extremely flexible, the ``graphblas`` dialect and associated
lowering pass only supports three encodings currently.

The `CSR64` encoding is usually defined with the alias::

    #CSR64 = #sparse_tensor.encoding<{
      dimLevelType = [ "dense", "compressed" ],
      dimOrdering = affine_map<(i,j) -> (i,j)>,
      pointerBitWidth = 64,
      indexBitWidth = 64
    }>

The `CSC64` encoding can be defined with the alias::

    #CSC64 = #sparse_tensor.encoding<{
      dimLevelType = [ "dense", "compressed" ],
      dimOrdering = affine_map<(i,j) -> (j,i)>,
      pointerBitWidth = 64,
      indexBitWidth = 64
    }>

In terms of data structure contents CSR and CSC are identical (with index,
pointer, and value arrays), just the indexing is reversed for CSC.  The sparse
tensor is then defined in the same way as a regular MLIR tensor, but with this
additional encoding attribute::

    tensor<?x?xf64, #CSC64>

The `CV64` encoding (for sparse vectors) is usually defined with the alias::

    #CV64 = #sparse_tensor.encoding<{
      dimLevelType = [ "compressed" ],
      pointerBitWidth = 64,
      indexBitWidth = 64
    }>


Note that the :ref:`graphblas-lower` only supports tensors with unknown
dimensions (indicated by the ``?``).

``graphblas.size``
------------------

Return the size of a sparse vector.

Example:
^^^^^^^^

.. code-block:: 

    %size = graphblas.size %sparse_vector : tensor<?xf64, #CV64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.size` $input attr-dict `:` type($input)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse vector


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``result``
     - Vector size as an `index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_


``graphblas.num_rows``
----------------------

Return the return the number of rows in a CSR or CSC matrix.

Example:
^^^^^^^^

.. code-block:: 

    %nrows = graphblas.num_rows %sparse_matrix : tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.num_rows` $input attr-dict `:` type($input)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)



Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``result``
     - Row count as an `index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_


``graphblas.num_cols``
----------------------

Return the return the number of columns in a CSR or CSC matrix.

Example:
^^^^^^^^

.. code-block:: 

    %ncols = graphblas.num_cols %sparse_matrix : tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.num_cols` $input attr-dict `:` type($input)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)



Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``result``
     - Column count as an `index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_


``graphblas.num_vals``
----------------------

Returns the number of values present in a CSC matrix, CSR matrix, or sparse vector.

Example:
^^^^^^^^

.. code-block:: 

    %csr_nnz = graphblas.num_vals %csr_matrix : tensor<?x?xf64, #CSR64>
    %vector_nnz = graphblas.num_vals %sparse_vector : tensor<?xf64, #CV64>
    %csc_nnz = graphblas.num_vals %csc_matrix : tensor<?x?xf64, #CSC64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.num_vals` $input attr-dict `:` type($input)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Sparse input tensor (CSR matrix, CSC matrix, or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``result``
     - Column count as an `index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_


``graphblas.dup``
-----------------

Returns a duplicate copy of the input CSC matrix, CSR matrix, or sparse tensor.

Example:
^^^^^^^^

.. code-block:: 

    %B = graphblas.dup %A : tensor<?x?xf64, #CSR64>
    %new_vec = graphblas.dup %vec : tensor<?xf64, #CV64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.dup` $input attr-dict `:` type($input)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Sparse input tensor (CSR matrix, CSC matrix, or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (same type as input tensor)


``graphblas.convert_layout``
----------------------------

Rewrite the contents of a sparse tensor to change it from CSR to CSC, or vice versa.

Vacuous conversions (e.g. CSC |rarr| CSC or CSR |rarr| CSR) are equivalent to no-ops and are removed by the ``--graphblas-lower`` pass. 

Example:
^^^^^^^^

.. code-block:: 

    %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.convert_layout` $input attr-dict `:` type($input) `to` type($output)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)



Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR or CSC)


``graphblas.transpose``
-----------------------

Returns a new sparse matrix that's the transpose of the input matrix.
Note that the behavior of this op differs depending on the sparse encoding
of the specified output tensor type.

Example:
^^^^^^^^

.. code-block::  text

    %a = graphblas.transpose %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
    %b = graphblas.transpose %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.transpose` $input attr-dict `:` type($input) `to` type($output)


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)



Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR or CSC)


``graphblas.matrix_select``
---------------------------

Returns new sparse tensor(s) with a subset of elements from the given matrix. 
The elements included in the resulting sparse tensor vary depending on the
selectors given (one of "triu", "tril", "ge", or "gt"). Some selectors, e.g.
"gt", require a thunk value. The ordering of the thunks/selectors determines
which thunk is used for which selector, i.e. the n\ :sup:`th` thunk is used
for the n\ :sup:`th` thunk-requiring selector. Multiple selectors may be given,
in which case multiple results will be returned. The given sparse tensor must
be a matrix, i.e. have rank 2. The input tensor must have a CSR sparsity or
a CSC sparsity. The resulting sparse tensors will have the same sparsity as
the given sparse tensor.

Example:
^^^^^^^^

.. code-block::  text

    %answer_triu = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>
    
    %thunk_a = constant 0.0 : f64 // used for the first "gt"
    %thunk_b = constant 9.9 : f64 // used for the second "gt"
    %answers = graphblas.matrix_select %sparse_tensor, %thunk_a, %thunk_b { selectors = ["triu", "gt", "tril", "gt"] } : tensor<?x?xf64, #CSR64>, f64, f64 to tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.matrix_select` $input (`,` $thunks^)? attr-dict `:` type($input) (`,` type($thunks)^)? `to` type($outputs)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``selectors``
     - ``::mlir::ArrayAttr`` (of string)
     - List of selectors.  Allowed: "triu" (upper triangle), "tril" (lower triangle), "gt" (values greater than the given thunk), and "ge" (values greater than or equal to the given thunk).

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)

   * - ``thunks``
     - Variadic list of thunk values, matching number of thunk-requiring selectors.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``outputs``
     - Variadic list of output tensors, matching number of selectors


``graphblas.reduce_to_vector``
------------------------------

Reduces a CSR or CSC matrix to a vector according to the given aggregator and axis.

The resulting sparse vector's element type varies according to the given aggregator.
The supported aggregators are "plus", "count", "argmin", and "argmax".
Aggregating via "plus" and "count" will cause the output's element type to match that
of the input tensor.
Aggregating via "argmin" and "argmax" will cause the output's element type to be
`i64 <https://mlir.llvm.org/docs/Dialects/Builtin/#integertype>`_.

If the axis attribute is 0, the input tensor will be reduced column-wise, so the resulting
vector's size must be the number of columns in the input tensor.

If the axis attribute is 1, the input tensor will be reduced row-wise, so the resulting
vector's size must be the  number of rows in the input tensor.

Example:
^^^^^^^^

.. code-block::  text

    %vec1 = graphblas.reduce_to_vector %matrix_1 { aggregator = "plus", axis = 0 } : tensor<7x9xf16, #CSR64> to tensor<9xf16, #CV64>
    %vec2 = graphblas.reduce_to_vector %matrix_2 { aggregator = "count", axis = 1 } : tensor<7x9xf16, #CSR64> to tensor<7xf16, #CV64>
    %vec3 = graphblas.reduce_to_vector %matrix_3 { aggregator = "argmin", axis = 1 } : tensor<7x9xf32, #CSR64> to tensor<7xi64, #CV64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.reduce_to_vector` $input attr-dict `:` type($input) `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``aggregator``
     - ``::mlir::StringAttr``
     - Aggregation method.  Allowed: "plus", "count", "argmin", "argmax"
   * - ``axis``
     - ``::mlir::I64Attr``
     - Aggregation axis.  Allowed: 0, 1

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse matrix (CSR or CSC)



Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Vector resulting from aggregation.


``graphblas.reduce_to_scalar``
------------------------------

Reduces a sparse tensor (CSR matrix, CSC matrix, or sparse vector) to a scalar according to the given aggregator. 

The supported aggregators are "plus", "count", "argmin", and "argmax".

"plus" and "count" require the result type to match the input tensor's element type.

"argmin" and "argmax" require the result type to be `i64 <https://mlir.llvm.org/docs/Dialects/Builtin/#integertype>`_.

Example:
^^^^^^^^

.. code-block::  text

    %answer_1 = graphblas.reduce_to_scalar %sparse_matrix { aggregator = "plus" } : tensor<?x?xf32, #CSR64> to f32
    %answer_2 = graphblas.reduce_to_scalar %sparse_vector { aggregator = "argmax" } : tensor<?xf64, #CSR64> to i64

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.reduce_to_scalar` $input attr-dict `:` type($input) `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``aggregator``
     - ``::mlir::StringAttr``
     - Aggregation method.  Allowed: "plus", "count", "argmin", "argmax".

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR matrix, CSC matrix, or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Scalar resulting from aggregation.


``graphblas.reduce_to_scalar_generic``
--------------------------------------

Reduces a sparse tensor (CSR matrix, CSC matrix, or sparse vector) to a scalar according to the given aggregator block. Only one aggregator block is allowed.

Example:
^^^^^^^^

.. code-block::  text

    %answer = graphblas.reduce_to_scalar_generic %sparse_vector : tensor<?xi64, #CV64> to i64 {
                ^bb0(%a : i64, %b : i64):
                  %result = std.addi %a, %b : i64
                  graphblas.yield agg %result : i64
    }

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.reduce_to_scalar_generic` $input attr-dict `:` type($input) `to` type($output) $extensions

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR matrix, CSC matrix, or sparse vector)
   * - ``extensions``
     - Variadic list of regions describing the aggregation behavior. Must have size 1.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Scalar resulting from aggregation block.


``graphblas.apply``
-------------------

Applies in an element-wise fashion the function indicated by the ``apply_operator``
attribute to each element of the given sparse tensor. The operator can be unary or
binary. Binary operators require a thunk. The supported binary operators are "min",
"div", and "fill". Unary operators cannot take a thunk. Unary operators cannot
take a thunk. The supported unary operators are "abs", "minv" (i.e. multiplicative
inverse or `1/x`), "ainv" (i.e. additive inverse or `-x`), and "identity".

The given sparse tensor must either be a CSR matrix, CSC matrix, or a sparse vector.

Using "minv" with integer types uses signed integer division and rounds towards
zero. For example, `minv(-2) == 1 / -2 == 0`.

Some binary operators, e.g. "div", are not symmetric. The sparse tensor and thunk
should be given in the order they should be given to the binary operator. For
example, to divide every element of a matrix by 2, use the following:

.. code-block:: text

    %thunk = constant 2 : i64
    %matrix_answer = graphblas.apply %sparse_matrix, %thunk { apply_operator = "div" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>

As another example, to divide 10 by each element of a sparse vector, use the
following:

.. code-block:: text
        
    %thunk = constant 10 : i64
    %vector_answer = graphblas.apply %thunk, %sparse_vector { apply_operator = "div" } : (i64, tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>

Note that the application only takes place for elements that are present in the
matrix. Thus, the operation will not apply when the values are missing in the
tensor. For example, `1.0 / [ _ , 2.0,  _ ] == [ _ , 0.5,  _ ]`.

Note that using the "identity" operator does not create a copy of the input tensor.

The shape of the output tensor will match that of the input tensor.

Example:
^^^^^^^^

.. code-block:: text

    %thunk = constant 100 : i64
    %matrix_answer = graphblas.apply %sparse_matrix, %thunk { apply_operator = "min" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>
    %vector_answer = graphblas.apply %sparse_vector { apply_operator = "abs" } : (tensor<?xi64, #CV64>) to tensor<?xi64, #CV64>

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.apply` $left (`,` $right^)? attr-dict `:` `(` type($left) (`,` type($right)^)? `)` `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``apply_operator``
     - ``::mlir::StringAttr``
     - Operator to apply.  Allowed: "min", "div", "fill", "abs", "minv", "ainv", and "identity"

Operands:
^^^^^^^^^

The allowed types of the inputs varies depending on the aggregator. Exactly one of the operands must be a sparse vector, CSR matrix, or CSC matrix.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``left``
     - Left-hand value
   * - ``right``
     -  Optional right-hand value


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor

 
``graphblas.apply_generic``
---------------------------

Applies an arbitrary transformation to every element of a CSR or CSC matrix or sparse vector according to the given transformation block. Only one transformation block is allowed.

Example:
^^^^^^^^

.. code-block:: text

    %thunk = constant 0.0 : f64
    %answer = graphblas.apply_generic %sparse_tensor : tensor<?xf64, #CV64> to tensor<?xf64, #CV64> {
      ^bb0(%val: f64):
        %pick = cmpf olt, %val, %thunk : f64
        %result = select %pick, %val, %thunk : f64
        graphblas.yield transform_out %result : f64
    }

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.apply_generic` $input attr-dict `:` type($input) `to` type($output) $extensions


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR matrix, CSC matrix, or sparse vector)
   * - ``extensions``
     - Variadic list of regions describing the transformation behavior. Must have size 1.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor


``graphblas.matrix_multiply``
-----------------------------

Performs a matrix multiply according to the given semiring and optional structural mask.
The structural mask specifies which values in the output are to be computed and thus must
have the same shape as the expected output.

The semiring must be a string of the form "<ADD_NAME>_<MUL_NAME>", e.g. "plus_times".
The options for "<ADD_NAME>" are "plus", "any", and "min". The options for
"<MUL_NAME>" are "pair", "times", "plus", "first", and "second". 

If the first input is a matrix, it must be CSR format. If the second input
is a matrix, it must be CSC format.  Matrix times vector will return a vector.
Vector times matrix will return a vector.  Matrix times matrix will return
a CSR matrix.

The mask (if provided) must be the same format as the returned object. There's an optional
boolean ``mask_complement`` attribute (which has a default value of ``false``) that will make
the op use the complement of the mask.

It should be noted that masks are not allowed for vector times vector multiplication.
	
Example:
^^^^^^^^

.. code-block:: text

    %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "min_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %mat, %vec, %mask { semiring = "any_first", mask_complement = true } : (tensor<?x?xf64, #CSR64>, tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %vec, %mat { semiring = "min_second", mask_complement = true } : (tensor<?xf64, #CV64>, tensor<?x?xf64, #CSC64>) to tensor<?xf64, #CV64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %vecA, %vecB { semiring = "any_pair" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_multiply` $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``semiring``
     - ``::mlir::StringAttr``
     - Semiring.
   * - ``mask_complement``
     - ``::mlir::BoolAttr``
     - (Optional) Whether to use the mask or the complement of the mask. Is false by default.

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand matrix multiply tensor operand (CSR matrix or sparse vector)
   * - ``b``
     - Right-hand matrix multiply tensor operand (CSC matrix or sparse vector)
   * - ``mask``
     -  Optional structural mask (CSR matrix or sparse vector)

Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR matrix or sparse vector or scalar).


``graphblas.matrix_multiply_generic``
-------------------------------------

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
boolean ``mask_complement`` attribute (which has a default value of ``false``) that will make
the op use the complement of the mask.


Example:
^^^^^^^^

.. code-block:: text

    %answer = graphblas.matrix_multiply_generic %a, %b {mask_complement = false} : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
        ^bb0:
             %identity = constant 0.0 : f64
             graphblas.yield add_identity %identity : f64
      },{
        ^bb0(%add_a: f64, %add_b: f64):
             %add_result = std.addf %add_a, %add_b : f64
             graphblas.yield add %add_result : f64
      },{
        ^bb0(%mult_a: f64, %mult_b: f64):
             %mult_result = std.mulf %mult_a, %mult_b : f64
             graphblas.yield mult %mult_result : f64
      },{
         ^bb0(%value: f64):
             %result = std.addf %value, %c100_f64: f64
             graphblas.yield transform_out %result : f64
      }

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_multiply_generic` $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``mask_complement``
     - ``::mlir::BoolAttr``
     - (Optional) Whether to use the mask or the complement of the mask. Is false by default.

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand matrix multiply tensor operand (CSR matrix or sparse vector)
   * - ``b``
     - Right-hand matrix multiply tensor operand (CSC matrix or sparse vector)
   * - ``mask``
     -  Optional structural mask (CSR matrix or sparse vector)
   * - ``extensions``
     -  Region containing blocks dictating the behavior of this op.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR matrix or sparse vector).


``graphblas.matrix_multiply_reduce_to_scalar_generic``
------------------------------------------------------

Performs a matrix multiply followed by a reduction to scalar.  Supports the 
same extension blocks as ``graphblas.matrix_multiply_generic`` and also requires
a binary aggregation block and aggregation identity block. These latter two blocks
are used for reducing the result of the matrix multiply to a scalar.

Example:
^^^^^^^^

.. code-block:: text

    %answer = graphblas.matrix_multiply_reduce_to_scalar_generic %a, %b : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to f64 {
        ^bb0:
            %identity = constant 0.0 : f64
            graphblas.yield add_identity %identity : f64
    },{
        ^bb0(%add_a: f64, %add_b: f64):
            %add_result = std.addf %add_a, %add_b : f64
            graphblas.yield add %add_result : f64
    },{
        ^bb0(%mult_a: f64, %mult_b: f64):
            %mult_result = std.mulf %mult_a, %mult_b : f64
            graphblas.yield mult %mult_result : f64
    },{
        %agg_identity = constant 0.0 : f64
        graphblas.yield agg_identity %agg_identity : f64
    },{
        ^bb0(%lhs: f64, %rhs: f64):
            %agg_result = std.addf %lhs, %rhs: f64
            graphblas.yield agg %agg_result : f64
    }

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_multiply_reduce_to_scalar_generic` $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) $extensions

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand matrix multiply tensor operand (CSR)
   * - ``b``
     - Right-hand matrix multiply tensor operand (CSC)
   * - ``mask``
     -  Optional structural mask (CSR)
   * - ``extensions``
     -  Region containing blocks dictating the behavior of this op.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Scalar resulting from aggregation.


``graphblas.union``
-------------------

Performs an element-wise union between two CSR matrices, two CSC matrices, or two sparse vectors.
The resulting sparse tensor with be the union of the two input structures.
When either object has a non-overlapping element, it is copied to the output.
When both objects have an overlapping element in a cell, an operation combines the result according
to the given operator.


Example:
^^^^^^^^

.. code-block::  text

    %combined = graphblas.union %A, %B { union_operator = "plus" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>

.. code-block::  text

    %combined = graphblas.union %vecA, %vecB { union_operator = "min" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.union` $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``union_operator``
     - ``::mlir::StringAttr``
     - Union operator. Allowed: "plus", "times", "min", "max", "first", and "second".

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand tensor operand (CSR matrix or CSC matrix or sparse vector)
   * - ``b``
     - Right-hand tensor operand (CSR matrix or CSC matrix or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Result sparse tensor.


``graphblas.intersect``
-----------------------

Performs an element-wise intersection between two CSR matrices, two CSC matrices, or two sparse vectors.
The resulting sparse tensor will be the intersection of the two input structures.
When either object has a non-overlapping element, it is not included the output.
When both objects have an overlapping element in a cell, an operation combines the result according
to the given operator.


Example:
^^^^^^^^

.. code-block::  text

    %combined = graphblas.intersect %vecA, %vecB { intersect_operator = "max" } : (tensor<?xf64, #CV64>, tensor<?xf64, #CV64>) to tensor<?xf64, #CV64>

.. code-block::  text

    %combined = graphblas.intersect %A, %B { intersect_operator = "minus" } : (tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSC64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.intersect` $a `,` $b attr-dict `:` `(` type($a) `,` type($b)  `)` `to` type($output)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``intersect_operator``
     - ``::mlir::StringAttr``
     - Intersect operator. Allowed: "plus", "minus", "times", "div", "min", "max", "first", and "second".

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand tensor operand (CSR matrix or CSC matrix or sparse vector)
   * - ``b``
     - Right-hand tensor operand (CSR matrix or CSC matrix or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Result sparse tensor.


``graphblas.update``
--------------------

Updates the output tensor based on the input and desired accumulation,
mask, and replacement.  This returns zero values and modifies the output in
place.  The supported accumulate operators are "plus", "times", "min", and "max".
The given tensors must be sparse.

There's an optional boolean `mask_complement` attribute (which has a default
value of `false`) that will make the op use the complement of the mask.

Example:
^^^^^^^^

.. code-block::  text

    graphblas.update %other_mat -> %mat(%mask) { accumulate_operator = "times", replace = true, mask_complement = true } : tensor<?x?xi64, #CSR64> -> tensor<?x?xi64, #CSR64>(tensor<?x?xi64, #CSR64>)

.. code-block::  text

    graphblas.update %other_vec -> %vec { accumulate_operator = "plus" } : tensor<?xi64, #CV64> -> tensor<?xi64, #CV64>

.. code-block::  text

    graphblas.update %other_mat -> %mat(%mask) { accumulate_operator = "max", replace = true } : tensor<?x?xi64, #CSC64> -> tensor<?x?xi64, #CSC64>(tensor<?x?xi64, #CSC64>)

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.update` $input `->` $output (`(` $mask^ `)`)? attr-dict `:` type($input) `->` type($output) (`(` type($mask)^ `)`)?


Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``accumulate_operator``
     - ``::mlir::StringAttr``
     - Update operator. Allowed: "plus", "times", "min", and "max".
   * - ``replace``
     - ``::mlir::BoolAttr``
     - (Optional) Whether or not the new values will replace the old ones (rather than being combined with them). Is false by default.
   * - ``mask_complement``
     - ``::mlir::BoolAttr``
     - (Optional) Whether to use the mask or the complement of the mask. Is false by default.


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor operand (CSR matrix or CSC matrix or sparse vector)
   * - ``output``
     - Output tensor operand (must have same shape, element type, and sparse encoding as the input tensor)
   * - ``mask``
     -  Optional structural mask (must have same shape, element type, and sparse encoding as the input tensor)


``graphblas.equal``
-------------------

Performs an equality check.  The given tensors must be sparse vectors, CSR matrices, or CSC matrices.
Checks equality of rank and size of tensors, as well as values and structure. Returns a single
boolean value.


Example:
^^^^^^^^

.. code-block::  text

    %answer = graphblas.equal %vec, %other_vec : tensor<?xi64, #CV64>, tensor<?xi64, #CV64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.equal` $a `,` $b attr-dict `:` type($a) `,` type($b)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``a``
     - Left-hand tensor operand (CSR matrix or CSC matrix or sparse vector)
   * - ``b``
     - Right-hand tensor operand (CSR matrix or CSC matrix or sparse vector)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Boolean value indicating equality.


``graphblas.vector_argminmax``
------------------------------

Determines the argmin or argmax of a vector depending on the value of
the minmax string attribute, which may only take on the value of "min" or
"max".  If there are multiple values that can be the argmin or argmax, an
arbitrary one is chosen from them.  The given sparse vector must have rank 1
and must be sparse.


Example:
^^^^^^^^

.. code-block::  text

    %ans = graphblas.vector_argminmax %vec { minmax = "min" } : tensor<?xi64, #CV64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.vector_argminmax` $vec attr-dict `:` type($vec)

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``minmax``
     - ``::rmlir::StringAttr``
     - Which extremum index to return.  Allowed: "min" and "max"


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``vec``
     - Input sparse vector.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - `Index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_ representing the position of the specified extremum.


``graphblas.vector_argmin``
---------------------------

Determines the argmin of a vector.  If there are multiple values that
can be the argmin, an arbitrary one is chosen from them.  The given
sparse vector must have rank 1 and must be sparse.


Example:
^^^^^^^^

.. code-block::  text

    %ans = graphblas.vector_argmin %vec : tensor<?xi64, #CV64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.vector_argmin` $vec attr-dict `:` type($vec)


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``vec``
     - Input sparse vector.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - `Index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_ representing the position of a minimum value.


``graphblas.vector_argmax``
---------------------------

Determines the argmax of a vector.  If there are multiple values that
can be the argmax, an arbitrary one is chosen from them.  The given
sparse vector must have rank 1 and must be sparse.


Example:
^^^^^^^^

.. code-block::  text

    %ans = graphblas.vector_argmax %vec : tensor<?xi64, #CV64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.vector_argmax` $vec attr-dict `:` type($vec)


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``vec``
     - Input sparse vector.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - `Index <https://mlir.llvm.org/docs/Dialects/Builtin/#indextype>`_ representing the position of a maximum value.


``graphblas.diag``
------------------

When given a square CSR or CSC matrix, returns the diagonal as a sparse vector.
When given a sparse vector, returns a square CSR or CSC matrix with the
vector's values along the diagonal.


Example:
^^^^^^^^

.. code-block::  text

    %csr_matrix_answer = graphblas.diag %vec : tensor<?xi64, #CV64> to tensor<?x?xi64, #CSR64>
    %csc_matrix_answer = graphblas.diag %vec : tensor<?xi64, #CV64> to tensor<?x?xi64, #CSC64>
    %vector_answer = graphblas.diag %mat : tensor<?x?xi64, #CSR64> to tensor<?xi64, #CV64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.diag` $input attr-dict `:` type($input) `to` type($output)


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input sparse tensor.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output sparse tensor.


``graphblas.matrix_select_random``
----------------------------------

Selects a random subset of up to `n` elements in each row of a CSR
matrix.  If there are less than `n` elements in a row, all elements in the row
are included in the output.

An external function must be provided via the `choose_n` attribute to
the op with the following signature:

.. code-block::  text

    func @my_choose_n(%context: !llvm.ptr<i8>,
                      %n: IndexType, %max_i: IndexType,
                      %output_indices: memref<?xIndexType>,
                      %row_values: memref<?xValueType)

where `IndexType` corresponds to the index element type of the sparse
tensor input and `ValueType` corresponds to the value element type of
the sparse tensor input.  This external function selects `n` random
indices from the interval `[0, max_i)` and writes them to the
`output_indices` memref in increasing order.  If desired, the
distribution of selected indices can be biased by the values in
`row_values`, which will have length `max_i`.  A uniform choice
function will ignore this last argument.

The implementation of the `choose_n` function is not specified by this
op because it will differ significantly depending on use case (uniform
or weighted sampling) and desired execution target (serial, parallel,
GPU, etc).

Example:
^^^^^^^^

.. code-block::  text

    %output = graphblas.matrix_select_random %a, %n, %rng_context { choose_n = @uniform_choose_n } : (tensor<?x?xf64, #CSR64>, i64, !llvm.ptr<i8>) to tensor<?x?xf64, #CSR64>


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.matrix_select_random` $input `,` $n `,` $rng_context attr-dict `:` `(` type($input) `,` type($n) `,` type($rng_context)  `)` `to` type($output)


Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``choose_n``
     - ``::mlir::SymbolRefAttr``
     - Symbol corresponding to an external function selecting random indices from an interval.


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - CSR matrix.
   * - ``n``
     - Integer representing number of random values to select for each row.
   * - ``rng_context``
     - Random number generator context used by the external "choose_n" function.


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output CSR matrix.


``graphblas.yield``
-------------------

Special terminator operation for blocks inside regions in several ``graphblas`` operations,
e.g. ``graphblas.matrix_multiply_generic``. It returns a value to the enclosing op, with a
meaning that depends on the required "kind" attribute.  It must be one of the following:

* transform_in_a
* transform_in_b
* transform_out
* select_in_a
* select_in_b
* select_out
* add_identity
* add
* mult_identity
* mult

Example:
^^^^^^^^

.. code-block:: text

    graphblas.yield transform_out %result : f64

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.yield` $kind $values attr-dict `:` type($values)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``values``
     - Variadic list of output values


``graphblas.comment``
---------------------

This op is intended to be a no-op and returns zero values. It merely contains a string attribute intended to hold code comments.


Example:
^^^^^^^^

.. code-block::  text

    graphblas.comment { comment = "here is a comment!" }


Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.comment` attr-dict

Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``comment``
     - ``::rmlir::StringAttr``
     - A string comment.


``graphblas.print``
-------------------

`graphblas.print` is used to pretty print values to stdout.

This is intended to be used for debugging only.
The strings attribute is a list of strings.
This op is variadic and takes an arbitrary number of inputs.
The printing alternates between printing the strings
and the input values.

Example:
^^^^^^^^

.. code-block::  text

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

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.print` $values attr-dict `:` type($values)


Attributes:
^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Attribute
     - MLIRType
     - Description
   * - ``strings``
     - ``::mlir::ArrayAttr`` (of string)
     - List of strings to print.


Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``values``
     - Variadic list of values to print.
