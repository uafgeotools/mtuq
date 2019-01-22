Installation under virtualenv
=============================

Confirm that you have ``virtualenv`` utility installed:

.. code::

   which virtualenv

If it is not present, `install it yourself <https://virtualenv.pypa.io/en/latest/installation/>`_ or ask a system administrator to.


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



