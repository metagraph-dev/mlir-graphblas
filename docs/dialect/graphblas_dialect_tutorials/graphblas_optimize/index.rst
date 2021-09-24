.. _graphblas_optimizing_pass:

GraphBLAS Optimizing Pass
=========================

These tutorials and examples will cover how to use the GraphBLAS dialect's ``--graphblas-structuralize`` and ``--graphblas-optimize`` passes by using the :ref:`engine` to lower several ops from the GraphBLAS dialect.

These tutorials assume the completion of the :ref:`graphblas_lowering_pass` and :ref:`graphblas_structuralizing_pass` tutorials.

The content of the tutorials are somewhat sequentially dependent as some later tutorials assume completion of previous tutorials.

Rather than using the JIT engine to lower the MLIR code examples down to something executable, these tutorials will use the ``--graphblas-structuralize`` and ``--graphblas-optimize`` passes to demonstrate the code transformations in order to demonstrate the expected behavior.

.. toctree::
   :maxdepth: 1

   fuse_select
   fuse_multiply_reduce
   fuse_multiply_apply
