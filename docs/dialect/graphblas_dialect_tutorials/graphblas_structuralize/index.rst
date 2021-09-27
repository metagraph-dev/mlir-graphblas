.. _graphblas_structuralizing_pass:

GraphBLAS Structuralizing Pass
==============================

These tutorials and examples will cover how to use the GraphBLAS dialect's ``--graphblas-structuralize`` passes by using the :ref:`engine` to lower several ops from the GraphBLAS dialect into a form that can be more easily optimized by the ``--graphblas-optimize`` pass. This mostly happens by lowering certain ops into their more generic equivalents, e.g. lowering a ``graphblas.apply`` op into a ``graphblas.apply_generic`` op, that are more easily optimizable by the ``--graphblas-optimize`` pass via op fusion and similar optimizations.

These tutorials assume the completion of the :ref:`graphblas_lowering_pass` tutorials and knowledge of the content in the :ref:`graphblas_ops_reference` documentation (in particular, the ``graphblas.*_generic`` ops).

The content of the tutorials are somewhat sequentially dependent as some later tutorials assume completion of previous tutorials.

Rather than using the JIT engine to lower the MLIR code examples down to something executable, these tutorials will solely use the ``--graphblas-structuralize`` pass to demonstrate the code transformations in order to demonstrate the expected behavior. Since the transformations are pretty simple, these tutorials will not do a deep dive but will act more as example demonstrations.

.. toctree::
   :maxdepth: 1

   lower_matrix_multiply_rewrite
   lower_apply_rewrite
   lower_reduce_to_scalar_rewrite
