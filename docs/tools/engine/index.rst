.. _engine:

JIT Engine
==========

``mlir-graphblas.MlirJitEngine`` provides a way to go from MLIR code and a set of passes to runnable Python code
using a Just-in-Time compiler strategy.

.. code-block:: python

    engine = MlirJitEngine(llvmlite_engine=None)

An optional ``llvmlite`` engine can be passes in. Otherwise a new one will be created.

The workflows for the JIT engine is:

- mlir-opt converts MLIR code into LLVM IR through a series of passes
- mlir-translate converts LLVM IR into LLVM code
- llvmlite compiles LLVM code into machine code
- pymlir is used to inspect the original MLIR code signatures for type information
- Python functions are created which accept numeric or numpy types

The mechanism to trigger this workflow is

.. code-block:: python

    engine.add(mlir_code, passes)

If an error is not raised, the functions defined in ``mlir_code`` will be available by indexing into the engine.

.. code-block:: python

    some_func = engine["some_func"]


Example
-------

.. code-block:: python

    >>> mlir_code = b"""
    #trait_1d_scalar = {
      indexing_maps = [
        affine_map<(i) -> (i)>,  // A
        affine_map<(i) -> (i)>   // X (out)
      ],
      iterator_types = ["parallel"],
      doc = "X(i) = A(i) OP Scalar"
    }
    func @scale_array(%input: tensor<?xf64>, %scale: f64) -> tensor<?xf64> {
      %0 = linalg.generic #trait_1d_scalar
         ins(%input: tensor<?xf64>)
         outs(%input: tensor<?xf64>) {
          ^bb(%a: f64, %s: f64):
            %0 = mulf %a, %scale  : f64
            linalg.yield %0 : f64
      } -> tensor<?xf64>
      return %0 : tensor<?xf64>
    }
    """
    >>> passes = [
        '--linalg-bufferize',
        '--func-bufferize',
        '--finalizing-bufferize',
        '--convert-linalg-to-affine-loops',
        '--lower-affine',
        '--convert-scf-to-std',
        '--convert-std-to-llvm',
    ]
    >>> from mlir_graphblas import MlirJitEngine
    >>> engine = MlirJitEngine()
    >>> engine.add(mlir_code, passes)
    ['scale_array']
    >>> import numpy as np
    >>> x = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    >>> engine['scale_array'](x, 20.0)
    array([ 22.,  44.,  66.,  88., 110.])

More Examples
-------------

Here is a series of tutorials and examples for the JIT engine.

They assume knowledge of the MLIR's `linalg dialect <https://mlir.llvm.org/docs/Dialects/Linalg/>`_ and go over how to compile and use MLIR code via the JIT engine.

The content of the tutorials are somewhat sequentially dependent as some later tutorials assume completion of previous tutorials.

Much of the complexity when using the JIT engine in practice comes from writing the MLIR code itself. While some of these tutorials go over features specific to the JIT engine, many of them are simply example uses of the JIT engine plus some MLIR code that can be useful as a template to learn from. 

.. toctree::
   :maxdepth: 1

   scalar_plus_scalar
   tensor_plus_tensor
   matrix_plus_broadcasted_vector
   scalar_times_tensor
   tensor_sum
   spmv
   sparse_vector_times_sparse_vector
   sparse_tensor_sum
