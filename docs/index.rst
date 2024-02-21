.. cuDecomp documentation master file, created by
   sphinx-quickstart on Wed Jun  1 13:44:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cuDecomp: An Adaptive Pencil Decomposition Library for NVIDIA GPUs
=============================================================================
These pages contain the documentation for cuDecomp, an adaptive pencil
decomposition library for NVIDIA GPUs.


  * **Disclaimer**:
    This library is currently in a research-oriented state, and has been
    released as a companion to a paper presented at the PASC22 conference (`link <https://dl.acm.org/doi/10.1145/3539781.3539797>`_).
    We are making it available here as it can be useful in other applications
    outside of this study or as benchmarking tool and usage example for various
    GPU communication libraries to perform transpose and halo communication. 

Please contact us or open a GitHub issue if you are interested in using this library
in your own solvers and have questions on usage and/or feature requests.  


Table of Contents
=================
.. toctree::
   :maxdepth: 2

   overview
   basic_usage
   autotuning
   nvshmem
   api
   env_vars


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
