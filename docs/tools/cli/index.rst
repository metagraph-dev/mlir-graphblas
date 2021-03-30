.. _cli:

mlir-opt CLI Wrapper
====================

``mlir-opt`` is the standard command line tool for applying passes to MLIR and lowering from one dialect
to another. 

``mlir_graphblas.MlirOptCli`` is a wrapper around the ``mlir-opt`` command line executable.

.. code-block:: python

    cli = MlirOptCli(executable=None, options=None)


The executable defaults to ``mlir-opt``, but can be specified to include the full path or a differently
named executable if needed.

``options``, if provided, must be a list of strings to pass to ``mlir-opt`` with every call. These options
are in addition to the passes that will be applied.

Applying Passes
---------------

The first way to apply passes is by calling

.. code-block:: python

    result = cli.apply_passes(input_mlir, list_of_passes)

This will return a string containing the final result of applying the list of passes to the input.

If any errors occur, ``MlirOptError`` will be raised. This error contains a ``.debug_result`` attribute,
which is explained below.

The second way to apply passes is by calling

.. code-block:: python

    result = cli.debug_passes(input_mlir, list_of_passes)

This always returns a ``DebugResult`` object.

DebugResult
-----------

A ``DebugResult`` object contains a list of ``.passes`` applied (or attempted to apply) and a list of
``.stages`` which resulted, including the original. As a result, there is always one more stage than
pass.

These stages and passes can be inspected manually, but the easiest way to interact with them is through
the :ref:`explorer`. To open the explorer, call

.. code-block:: python

    result.explore()

A new browser tab will appear showing the explorer.

Examples
--------

Here are some examples of our CLI tool:

.. toctree::
   :maxdepth: 1

   apply_passes_to_string_or_file
   using_debugresult
