.. _installation:

Installation
============

mlir-graphblas is implemented via Python and Cython.

Building from source is possible, but the recommended method to install is via ``conda``.

It is assumed by mlir-graphblas that it is installed using ``conda`` or ``venv``. Thus, installing using one of those two methods is recommended.

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

These should be automatically installed when mlir-graphblas is installed

  - `numpy <https://numpy.org>`__
  - `PyMLIR <https://github.com/metagraph-dev/pymlir>`__
  - `lvmlite <https://llvmlite.readthedocs.io/en/latest/>`__
  - `pygments <https://pygments.org/>`__
  - `donfig <https://donfig.readthedocs.io/>`__
  - `panel <https://panel.holoviz.org/>`__
  - `bokeh <https://bokeh.org/>`__

