GraphBLAS Passes and graphblas-opt
==================================

The mlir-graphblas package includes an drop-in replacement for ``mlir-opt``
called ``graphblas-opt`` which includes the custom ``graphblas`` dialect along
with all the other dialects that ship with MLIR.

This document is not intended to be a complete tutorial on ``graphblas-opt`` and acts more
as a reference manual for the passes exclusively available for ``graphblas-opt``. For tutorials
on these passes specific to ``graphblas-opt``, see our :ref:`graphblas_dialect_tutorials`.

There are five passes specific to ``graphblas-opt`` with two intended usages:

Standard Lowering

* ``--graphblas-structuralize``: Lowers higher-level ``graphblas`` ops
  into lower-level generic ``graphblas`` ops
* ``--graphblas-dwim``: Adds ``convert-layout`` calls to inputs to match the expected
  shape and type of generic ``graphblas`` ops
* ``--graphblas-optimize``: Fuses lower-level generic ``graphblas`` ops together
  to eliminate temporary tensors and redundant loops.
* ``--graphblas-lower``: Convert ``graphblas`` ops to the ``scf``, ``std``,
  and ``memref`` dialects.

Linalg Lowering

* ``--graphblas-structuralize``: Lowers higher-level ``graphblas`` ops
  into lower-level generic ``graphblas`` ops
* ``--graphblas-optimize``: Fuses lower-level generic ``graphblas`` ops together
  to eliminate temporary tensors and redundant loops.
* ``--graphblas-linalg-lower``: Convert ``graphblas`` ops to the ``linalg`` dialects,
  specifically to ``linalg.generic`` calls. These will be further lowered by the
  ``sparse_tensor`` dialect.

.. _graphblas-structuralize:

``--graphblas-structuralize`` Pass
----------------------------------

The structuralization pass performs eight generic transformations:

* Transform ``graphblas.matrix_multiply`` ops into equivalent
  ``graphblas.matrix_multiply_generic`` ops.
* Transform ``graphblas.apply`` ops into equivalent
  ``graphblas.apply_generic`` ops.
* Transform ``graphblas.select`` ops into equivalent
  ``graphblas.select_generic`` ops.
* Transform ``graphblas.union`` ops into equivalent
  ``graphblas.union_generic`` ops.
* Transform ``graphblas.intersect`` ops into equivalent
  ``graphblas.intersect_generic`` ops.
* Transform ``graphblas.update`` ops into equivalent
  ``graphblas.update_generic`` ops.
* Transform ``graphblas.reduce_to_vector`` ops into equivalent
  ``graphblas.reduce_to_vector_generic`` ops.
* Transform ``graphblas.reduce_to_scalar`` ops into equivalent
  ``graphblas.reduce_to_scalar_generic`` ops.

.. _graphblas-dwim:

``--graphblas-dwim`` Pass
-------------------------

The DWIM (Do-What-I-Mean) pass automatically converts the layout of input matrices
to conform to the requirements each pass. DWIM transformations are included for:

* ``graphblas.transpose``
* ``graphblas.reduce_to_vector``
* ``graphblas.matrix_multiply``
* ``graphblas.matrix_multiply_reduce_to_scalar``

.. _graphblas-optimize: 

``--graphblas-optimize`` Pass
-----------------------------

The optimization pass performs two transformations:

* Combine ``graphblas.matrix_multiply`` followed by ``graphblas.apply``
  into a single ``graphblas.matrix_multiply`` with an additional scalar 
  transformation of the output elements as they are written to memory.
* Combine ``graphblas.matrix_multiply`` followed by ``graphblas.matrix_reduce_to_scalar``
  into a new ``graphblas.matrix_multiply_reduce_to_scalar`` op that does the
  sparse multiply and accumulate in a single pass (rather than two passes for
  ``graphblas.matrix_multiply``).

.. _graphblas-lower: 

``--graphblas-lower`` Pass
--------------------------

The lowering pass is intended to allow the ``graphblas`` ops to be executed on
a CPU target.  Although the lowering assumes the same `C++ structure
<https://mlir.llvm.org/doxygen/SparseUtils_8cpp_source.html>`_ as is used in
the existing sparse tensor related passes in MLIR, ``--graphblas-lower`` does
not use the ``linalg`` dialect, as there is not yet support for sparse output.
The ``graphblas`` lowering pass requires a small runtime based on the MLIR
sparse runtime, but with some additional functions that allow for the
allocation, resizing, and deallocation of the sparse tensor data structure.

In general, this pass uses ``scf`` ops to handle traversing the sparse data
structure, with `scf.parallel` used when possible to indicate loops where the
loop body iterations are independent of each other, or are performing a
well-defined reduction.

.. _graphblas-linalg-lower

``--graphblas-linalg-lower`` Pass
---------------------------------

The linalg lowering will eventually be a full replacement for the standard lowering pass,
once improvements are made to the ``sparse_tensor`` handling of ``linalg.generic``.

One benefit of linalg lowering is a much smaller and simpler lowering output to maintain.
Another is the ability to handle more types of inputs, eliminating the need for the DWIM
pass.