GraphBLAS Passes and graphblas-opt
==================================

The mlir-graphblas package includes an drop-in replacement for ``mlir-opt``
called ``graphblas-opt`` which includes the custom ``graphblas`` dialect along
with all the other dialects that ship with MLIR.

This document is not intended to be a complete tutorial on ``graphblas-opt`` and acts more
as a reference manual for the passes exclusively available for ``graphblas-opt``. For tutorials
on these passes specific to ``graphblas-opt``, see our :ref:`graphblas_dialect_tutorials`.

The three passes specific to ``graphblas-opt`` are:

* ``--graphblas-structuralize``: Lowers higher-level ``graphblas`` ops
  into lower-level generic ``graphblas`` ops to enable the
  ``--graphblas-optimize`` pass to perform optimizations such as op fusion.
* ``--graphblas-optimize``: Fuses lower-level generic ``graphblas`` ops together
  to eliminate temporary tensors and redundant loops.
* ``--graphblas-lower``: Convert ``graphblas`` ops to the ``scf``, ``std``,
  and ``memref`` dialects.

The ``--graphblas-structuralize`` and ``--graphblas-optimize`` passes are optional
and unoptimized code can be generated and executed with ``--graphblas-lower``.

Also, ``--graphblas-optimize`` can be used without uses of ``--graphblas-structuralize``,
but in these sorts of situations, ``--graphblas-optimize`` may not be able to find all
the available optimizations as it only transforms lower-level generic ``graphblas`` ops
(all of which may not yet be present without uses of ``--graphblas-structuralize``).
It's best practice to always use ``--graphblas-structuralize`` prior to uses of
``--graphblas-optimize``.

.. _graphblas-structuralize: 

``--graphblas-structuralize`` Pass
----------------------------------

The structuralization pass performs three transformations:

* Transform ``graphblas.matrix_multiply`` ops into equivalent
  ``graphblas.matrix_multiply_generic`` ops.
* Transform ``graphblas.apply`` ops into equivalent
  ``graphblas.apply_generic`` ops.
* Transform ``graphblas.reduce_to_scalar`` ops into equivalent
  ``graphblas.reduce_to_scalar_generic`` ops.

.. _graphblas-optimize: 

``--graphblas-optimize`` Pass
-----------------------------

The optimization pass performs three transformations:

* Combine ``graphblas.matrix_select`` ops on the same input sparse tensor
  into a single loop with multiple output tensors.
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
well-defined reduction.  Note that ``scf.parallel`` does not automatically
enable parallel, multithreaded execution without the use of additional passes.
Unfortunately, the ``--convert-scf-to-openmp`` pass cannot handle the code
``--graphblas-lower`` generates as OpenMP reductions are not yet supported.
