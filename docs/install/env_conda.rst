Installation under conda
========================

Download and install miniconda following `these instructions <https://conda.io/docs/user-guide/install/index.html>`_.


Create a conda virtual environment:

.. code::

   conda config --prepend channels conda-forge
   conda create -n mtuq python=3
   conda install numpy h5py obspy instaseis


Install MTUQ: 

.. code::

   conda activate mtuq
   git clone https://github.com/uafgeotools/mtuq.git
   cd mtuq
   pip install -e .



