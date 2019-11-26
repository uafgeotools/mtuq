Installation under conda
========================

Download and install miniconda following `these instructions <https://conda.io/docs/user-guide/install/index.html>`_.


Create a conda virtual environment:

.. code::

   conda create -n mtuq_env python=2
   conda install numpy obspy h5py


Install MTUQ: 

.. code::

   conda activate mtuq_env
   git clone https://github.com/uafseismo/mtuq.git
   cd mtuq
   pip install -e .

