Installation under virtualenv
=============================

Install the <virtualenv https://virtualenv.pypa.io/en/latest/installation/>`_ program, if not already present.

Create a new virtual environment:

.. code::

   cd $HOME/virtual
   virtualenv --python=python2.7 mtuq_env


Download MTUQ source code:

.. code::

   cd $HOME/packages
   git clone https://github.com/uafseismo/mtuq.git


Install MTUQ:

.. code::

   source $HOME/virtual/mtuq_env/bin/activate
   cd $HOME/packages/mtuq
   pip install -e .



