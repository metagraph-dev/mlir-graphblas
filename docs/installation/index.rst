.. _installation:

Installation
============

mlir-graphblas is implemented via Python and Cython.

Building from source or installing from ``pip`` is possible, but the recommended method to install is using ``conda``.

Python version support
----------------------

Python 3.8 and above is supported.

Installing using conda
----------------------

::

    conda install -c conda-forge -c metagraph mlir-graphblas

Installing from source
----------------------

Instructions for building from source are currently a work in progress.

Installing using venv
---------------------

Instructions for installing via venv are currently a work in progress.

Required Dependencies
---------------------

These should be automatically installed when ``mlir-graphblas`` is installed

  - `NumPy <https://numpy.org>`__
  - `SciPy <https://scipy.org/>`__
  - `PyMLIR <https://github.com/metagraph-dev/pymlir>`__
  - `llvmlite <https://llvmlite.readthedocs.io/en/latest/>`__
  - `pygments <https://pygments.org/>`__
  - `donfig <https://donfig.readthedocs.io/>`__
  - `panel <https://panel.holoviz.org/>`__
  - `bokeh <https://bokeh.org/>`__
  - `MLIR <https://mlir.llvm.org/>`__
  - `Cython <https://cython.org/>`__
  - `CMake <https://cmake.org/>`__
  - `Ninja <https://ninja-build.org/>`__
  - `lit <https://llvm.org/docs/CommandGuide/lit.html>`__
  - `Jinja <https://jinja.palletsprojects.com/>`__
