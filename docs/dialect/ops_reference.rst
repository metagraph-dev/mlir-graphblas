
..    include:: <isonum.txt>

GraphBLAS Ops
=============

The ``graphblas`` dialect describes standard sparse tensor operations that are
found in the `GraphBLAS spec`_.  The ops are not one-to-one equivalents of
GraphBLAS function calls in order to fit into MLIR's SSA requirements with
immutable sparse tensors.

This document is not intended to be a tutorial and acts more as a reference manual for the
ops in the GraphBLAS dialect. For tutorials, see our :ref:`graphblas_dialect_tutorials`.

.. _GraphBLAS spec: http://people.eecs.berkeley.edu/~aydin/GraphBLAS_API_C_v13.pdf

Assumptions
-----------

Although the `sparse tensor encoding
<https://mlir.llvm.org/docs/Dialects/SparseTensorOps/#sparsetensorencodingattr>`_
in MLIR is extremely flexible, the ``graphblas`` dialect and associated
lowering pass only supports two encodings currently.

The `CSR64` encoding is usually defined with the alias::

    #CSR64 = #sparse_tensor.encoding<{
      dimLevelType = [ "dense", "compressed" ],
      dimOrdering = affine_map<(i,j) -> (i,j)>,
      pointerBitWidth = 64,
      indexBitWidth = 64
    }>

The `CSC64` encoding can be defined with the alias::

    #CSR64 = #sparse_tensor.encoding<{
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

Note that the :ref:`graphblas-lower` only supports rank 2 tensors with unknown
dimensions (indicated by the ``?``).

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
     - Input tensor (CSR or CSC)


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

Returns new sparse tensor(s) with a subset of element from the given matrix.  
The elements included in the resulting sparse tensor vary depending on the
selectors given (one of "triu", "tril", or "gt0"). Multiple selectors may be
given, in which case multiple results will be returned The given sparse tensor
must be a matrix, i.e. have rank 2. The input tensor must have a CSR sparsity
or a CSC sparsity. The resulting sparse tensors will have the same sparsity as
the given sparse tensor.

Example:
^^^^^^^^

.. code-block::  text

    %answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.matrix_select` $input attr-dict `:` type($input) `to` type($outputs)

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
     - List of selectors.  Allowed: "triu" (upper triangle), "tril" (lower triangle), and "gt0" (values greater than 0)

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR or CSC)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``outputs``
     - Variadic list of output tensors, matching number of selectors


``graphblas.matrix_reduce_to_scalar``
-------------------------------------

Reduces a sparse tensor to a scalar according to the given aggregator. Currently, the only available aggregator is "sum".


Example:
^^^^^^^^

.. code-block::  text

    %answer = graphblas.matrix_reduce_to_scalar %sparse_tensor { aggregator = "sum" } : tensor<?x?xi64, #CSR64> to i64

Syntax:
^^^^^^^

.. code-block::  text

    operation ::= `graphblas.matrix_reduce_to_scalar` $input attr-dict `:` type($input) `to` type($output)

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
     - Aggregation method.  Allowed: "sum"

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR or CSC)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Scalar resulting from aggregation.


``graphblas.matrix_apply``
--------------------------

Applies in an element-wise fashion the function indicated by the ``apply_operator`` attribute to each element and the thunk. Currently, the only valid operator is "min".


Example:
^^^^^^^^

.. code-block:: text

    %thunk = constant 100 : i64
    %answer = graphblas.matrix_apply %sparse_tensor, %thunk { apply_operator = "min" } : (tensor<?x?xi64, #CSR64>, i64) to tensor<?x?xi64, #CSR64>

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_apply` $input `,` $thunk attr-dict `:` `(` type($input) `,` type($thunk) `)` `to` type($output)

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
     - Operator to apply.  Allowed: "min"

Operands:
^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Operand
     - Description
   * - ``input``
     - Input tensor (CSR or CSC)
   * - ``thunk``
     -  Thunk value (scalar)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR or CSC).


``graphblas.matrix_multiply``
-----------------------------

Performs a matrix multiply according to the given semiring and optional structural mask.
The structural mask specifies which values in the output are to be computed and thus must
have the same shape as the expected output. The semiring must be one of "plus_times",
"plus_pair", or "plus_plus".
``graphblas.matrix_multiply`` also accepts an optional region that specifies element-wise
postprocessing to be done on the result of the matrix multiplication. The region must use
``graphblas.yield`` to indicate the result of the element-wise postprocessing.

Example:
^^^^^^^^

.. code-block:: text

    %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_plus" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to tensor<?x?xi64, #CSR64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %argA, %argB, %mask { semiring = "plus_pair" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to tensor<?x?xf64, #CSR64>

.. code-block:: text

    %answer = graphblas.matrix_multiply %argA, %argB { semiring = "plus_times" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to tensor<?x?xf64, #CSR64> {
        ^bb0(%value: f64):
            %result = std.mulf %value, %value: f64
            graphblas.yield transform_out %result : f64
    }

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_multiply` $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output) ($body^)?

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
     - Semiring.  Allowed: "plus_times", "plus_pair", and "plus_plus"

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
   * - ``body``
     -  Optional region specifying postprocessing (must use ``graphblas.yield``)


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Output tensor (CSR).


``graphblas.matrix_multiply_reduce_to_scalar``
----------------------------------------------

Performs a matrix multiply followed by a reduction to scalar.
The multiplication is done according to the given semiring and optional structural mask.
The semiring must be one of "plus_times", "plus_pair", or "plus_plus".
The reduction to scalar is done according to the given aggregator.
The aggregator must be "sum".
Unlike ``graphblas.matrix_multiply``, ``graphblas.matrix_multiply_reduce_to_scalar`` does not
accept a region.

Example:
^^^^^^^^

.. code-block:: text

    %answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB { semiring = "plus_plus", aggregator = "sum" } : (tensor<?x?xi64, #CSR64>, tensor<?x?xi64, #CSC64>) to f64

.. code-block:: text

    %answer = graphblas.matrix_multiply_reduce_to_scalar %argA, %argB, %mask { semiring = "plus_times", aggregator = "sum" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>, tensor<?x?xf64, #CSR64>) to f64

Syntax:
^^^^^^^

.. code-block:: text

    operation ::= `graphblas.matrix_multiply_reduce_to_scalar` $a `,` $b (`,` $mask^)? attr-dict `:` `(` type($a) `,` type($b)  (`,` type($mask)^)? `)` `to` type($output)

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
     - Semiring.  Allowed: "plus_times", "plus_pair", and "plus_plus"
   * - ``aggregator``
     - ``::mlir::StringAttr``
     - Aggregation method.  Allowed: "sum"

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


Results:
^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Result
     - Description
   * - ``output``
     - Scalar resulting from aggregation.


``graphblas.yield``
-------------------

Special terminator operation for blocks inside regions in several ``graphblas`` operations,
e.g. ``graphblas.matrix_multiply``. It returns a value to the enclosing op, with a meaning
that depends on the required "kind" attribute.  It must be one of the following:

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
