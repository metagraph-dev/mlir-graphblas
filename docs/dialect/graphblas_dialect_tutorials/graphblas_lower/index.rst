.. _graphblas_lowering_pass:

GraphBLAS Lowering Pass
=======================

These tutorials and examples will cover how to use the GraphBLAS dialect's ``--graphblas-lower`` pass by using the :ref:`engine` to lower several ops from the GraphBLAS dialect.

The main purpose of the ``--graphblas-lower`` pass is to lower from the GraphBLAS dialect into a lower level dialect, e.g. the `SCF dialect <https://mlir.llvm.org/docs/Dialects/SCFDialect/>`_ or the `Sparse Tensor dialect <https://mlir.llvm.org/docs/Dialects/SparseTensorOps/>`_. Rather than simply showing the code transformations, we'll use the :ref:`engine` to take some example MLIR code using the GraphBLAS dialect and create executable Python code from it. Since we won't be able to go over all of the ops in the GraphBLAS dialect and since all of the ops are documented with examples in the `GraphBLAS Ops Reference <../../ops_reference.ipynb>`_, our examples in this section will mostly cover ops that are commonly used.

The content of the tutorials are somewhat sequentially dependent as some later tutorials assume completion of previous tutorials. They also assume familiarity with the content of the `GraphBLAS Ops Reference <../../ops_reference.ipynb>`_.

.. toctree::
   :maxdepth: 1

   python_utilities
   sparse_layouts
   vector_ops
   matrix_ops
   debugging_ops
