Installation
============

Required Dependencies
---------------------
- Python 3.8 or later
- numpy
- scipy
- matplotlib

.. _installation:

you can use the ``pyhsi.beam()`` function:

.. autofunction:: pyhsi.beam

Instructions
------------
The easiest way to install `PyHSI` is to use the python package index: ::

    pip install pyhsi

For users wishing to develop: ::

    git clone https://github.com/MBriggs27810518/pyhsi.git
    cd pyhsi
    pip install -e .

For contributions, first fork the repo and clone from your fork. `Here <https://www.dataschool.io/how-to-contribute-on-github/>`_ is a good guide on this workflow.

Tests
-----
`PyHSI` comes with ``pytest`` functions to verify the correct functioning of the package.
Users can test this using: ::

    python -m pytest

from the root directory of the package.
