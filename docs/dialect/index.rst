.. _dialect:

GraphBLAS Dialect
=================

The ``graphblas`` dialect is designed to make it possible to express
`GraphBLAS`_ algorithms in `MLIR`_ in a compact way.  The dialect does not
define any new types, but rather operates on `MLIR sparse tensors`_.

.. _GraphBLAS: https://graphblas.github.io/
.. _MLIR: https://mlir.llvm.org/
.. _MLIR sparse tensors: https://mlir.llvm.org/docs/Dialects/SparseTensorOps/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ops
   passes
