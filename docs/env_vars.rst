.. _env-var-section-ref:

cuDecomp Environment Variables
==============================

The following section lists the environment variables available to configure the cuDecomp library.

CUDECOMP_NCCL_ENABLE_UBR
------------------------
(since v0.4.0)

This variable controls whether cuDecomp registers its communication buffers with the NCCL library using :code:`ncclCommRegister`/:code:`ncclCommDeregister` (i.e., user buffer registration).
Registration can improve NCCL send/receive performance in some scenarios. See `User Buffer Registration <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html>`_
section of the NCCL documentation for more details.

Default setting is off (:code:`0`). Setting this variable to :code:`1` will enable this feature.

