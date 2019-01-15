Installation under virtualenv
=============================

.. note::

    If you prefer locations other than ``$HOME/packages`` and ``$HOME/virtual``, modify the following accordingly.


.. code::


Create a fresh virtual environment:

.. code::

   cd $HOME/virtual
   virtualenv mtuq_env


Download MTUQ source code:

.. code::

   cd $HOME/packages
   git clone https://github.com/uafseismo/mtuq.git


Install MTUQ:

.. code::

   source $HOME/virtual/mtuq_env/bin/activate
   cd $HOME/packages/mtuq
   pip install -e .



