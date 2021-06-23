GraphBLAS Ops
=============

The ``graphblas`` dialect describes standard sparse tensor operations that are
found in the `GraphBLAS spec`_.  The ops are not one-to-one equivalents of
GraphBLAS function calls in order to fit into MLIR's SSA requirements with
immutable sparse tensors.

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

Example:
^^^^^^^^

.. code-block::

    %answer = graphblas.convert_layout %sparse_tensor : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>

Syntax:
^^^^^^^

.. code-block::

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
----------------------------

Returns new sparse tensor(s) with a subset of element from the given matrix.  
The elements included in the resulting sparse tensor vary depending on the
selectors given (one of "triu", "tril", or "gt0"). Multiple selectors may be
given, in which case multiple results will be returned The given sparse tensor
must be a matrix, i.e. have rank 2. The input tensor must have a CSR sparsity
or a CSC sparsity. The resulting sparse tensors will have the same sparsity as
the given sparse tensor.

Example:
^^^^^^^^

.. code-block::

    %answer = graphblas.matrix_select %sparse_tensor { selectors = ["triu"] } : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSR64>

Syntax:
^^^^^^^

.. code-block::

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
   * - ``outputs``
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

