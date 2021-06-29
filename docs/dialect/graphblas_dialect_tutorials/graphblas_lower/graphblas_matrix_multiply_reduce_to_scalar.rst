.. _graphblas_matrix_multiply_reduce_to_scalar:

graphblas.matrix_multiply_reduce_to_scalar
==========================================

Here, we'll go over the ``graphblas.matrix_multiply_reduce_to_scalar`` op. 

``graphblas.matrix_multiply_reduce_to_scalar`` is behaviorally equivalent to sequential calls to ``graphblas.matrix_multiply`` and ``graphblas.matrix_reduce_to_scalar``. The purpose of ``graphblas.matrix_multiply_reduce_to_scalar`` is to allow the lowering to add additional performance optimizations that wouldn't be available when using ``graphblas.matrix_multiply`` and ``graphblas.matrix_reduce_to_scalar`` independently.

Here's an example use of the ``graphblas.matrix_multiply_reduce_to_scalar`` op::

  %answer = graphblas.matrix_multiply_reduce_to_scalar %a, %b { semiring = "plus_times", aggregator = "sum" } : (tensor<?x?xf64, #CSR64>, tensor<?x?xf64, #CSC64>) to f64


The options for the ``semiring`` and ``aggregator`` attributes are the same as those for ``graphblas.matrix_multiply`` and ``graphblas.matrix_reduce_to_scalar``, respectively.

It is important to note that ``graphblas.matrix_multiply_reduce_to_scalar`` will rarely be handwritten. It is mostly used by ``graphblas-opt`` to optimize MLIR code using the GraphBLAS dialect (via the ``--graphblas-optimize`` pass) to combine sequential calls of ``graphblas.matrix_multiply`` and ``graphblas.matrix_reduce_to_scalar``.

Since the use of ``graphblas.matrix_multiply_reduce_to_scalar`` is fairly self-explanatory, we won't go over examples of how to use it here and will refer to our previous examples on ``graphblas.matrix_multiply`` and ``graphblas.matrix_reduce_to_scalar``.
