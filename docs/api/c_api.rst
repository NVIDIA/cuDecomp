##############
cuDecomp C API
##############

These are all the types and functions available in the cuDecomp C API.

Types
==================

Internal types
--------------------------------

.. _cudecompHandle_t-ref:

cudecompHandle_t
________________
.. doxygentypedef:: cudecompHandle_t

------

.. _cudecompGridDesc_t-ref:

cudecompGridDesc_t
__________________
.. doxygentypedef:: cudecompGridDesc_t

------

Grid Descriptor Configuration
-----------------------------

.. _cudecompGridDescConfig_t-ref:

cudecompGridDescConfig_t
________________________
.. doxygenstruct:: cudecompGridDescConfig_t
  :members:

------

.. _cudecompGridDescAutotuneOptions_t-ref:

cudecompGridDescAutotuneOptions_t
_________________________________
.. doxygenstruct:: cudecompGridDescAutotuneOptions_t
  :members:

------

Pencil Information
-----------------------------

.. _cudecompPencilInfo_t-ref:

cudecompPencilInfo_t
____________________
.. doxygenstruct:: cudecompPencilInfo_t
  :members:


Communication Backends
---------------------------------

.. _cudecompTransposeCommBackend_t-ref:

cudecompTranposeCommBackend_t
_____________________________
.. doxygenenum :: cudecompTransposeCommBackend_t

------

.. _cudecompHaloCommBackend_t-ref:

cudecompHaloCommBackend_t
_________________________
.. doxygenenum :: cudecompHaloCommBackend_t

------

Additional Enumerators
---------------------------------

.. _cudecompDataType_t-ref:

cudecompDataType_t
__________________
.. doxygenenum :: cudecompDataType_t

------

.. _cudecompAutotuneGridMode_t-ref:

cudecompAutotuneGridMode_t
__________________________
.. doxygenenum :: cudecompAutotuneGridMode_t

------

.. _cudecompResult_t-ref:

cudecompResult_t
________________
.. doxygenenum :: cudecompResult_t


Functions
==================

Library Initialization/Finalization
-----------------------------------


.. _cudecompInit-ref:

cudecompInit
____________
.. doxygenfunction:: cudecompInit

------

.. _cudecompFinalize-ref:

cudecompFinalize
________________
.. doxygenfunction:: cudecompFinalize

------

Grid Descriptor Management
-----------------------------------

.. _cudecompGridDescCreate-ref:

cudecompGridDescCreate
______________________
.. doxygenfunction:: cudecompGridDescCreate

------

.. _cudecompGridDescDestroy-ref:

cudecompGridDescDestroy
_______________________
.. doxygenfunction:: cudecompGridDescDestroy

------

.. _cudecompGridDescConfigSetDefaults-ref:

cudecompGridDescConfigSetDefaults
_________________________________
.. doxygenfunction:: cudecompGridDescConfigSetDefaults

------

.. _cudecompGridDescAutotuneOptionsSetDefaults-ref:

cudecompGridDescAutotuneOptionsSetDefaults
__________________________________________
.. doxygenfunction:: cudecompGridDescAutotuneOptionsSetDefaults

------

Workspace Management
----------------------------------------

.. _cudecompGetTransposeWorkspaceSize-ref:

cudecompGetTransposeWorkspaceSize
_________________________________
.. doxygenfunction:: cudecompGetTransposeWorkspaceSize

------

.. _cudecompGetHaloWorkspaceSize-ref:

cudecompGetHaloWorkspaceSize
____________________________
.. doxygenfunction:: cudecompGetHaloWorkspaceSize

------

.. _cudecompGetDataTypeSize-ref:

cudecompGetDataTypeSize
_______________________
.. doxygenfunction:: cudecompGetDataTypeSize

------

.. _cudecompMalloc-ref:

cudecompMalloc
______________
.. doxygenfunction:: cudecompMalloc

------

.. _cudecompFree-ref:

cudecompFree
____________
.. doxygenfunction:: cudecompFree

------

Helper Functions
----------------

.. _cudecompGetPencilInfo-ref:

cudecompGetPencilInfo
_____________________
.. doxygenfunction:: cudecompGetPencilInfo

------

.. _cudecompTransposeCommBackendToString-ref:

cudecompTranposeCommBackendToString
___________________________________
.. doxygenfunction:: cudecompTransposeCommBackendToString

------

.. _cudecompHaloCommBackendToString-ref:

cudecompHaloCommBackendToString
_______________________________
.. doxygenfunction:: cudecompHaloCommBackendToString

------

.. _cudecompGetGridDescConfig-ref:

cudecompGetGridDescConfig
_________________________
.. doxygenfunction:: cudecompGetGridDescConfig

------

.. _cudecompGetShiftedRank-ref:

cudecompGetShiftedRank
______________________
.. doxygenfunction:: cudecompGetShiftedRank

------

Transposition Functions
-----------------------

.. _cudecompTransposeXToY-ref:

cudecompTransposeXToY
_____________________
.. doxygenfunction:: cudecompTransposeXToY

------

.. _cudecompTransposeYToZ-ref:

cudecompTransposeYtoZ
_____________________
.. doxygenfunction:: cudecompTransposeYToZ

------

.. _cudecompTransposeZToY-ref:

cudecompTransposeZToY
_____________________
.. doxygenfunction:: cudecompTransposeZToY

------

.. _cudecompTransposeYToX-ref:

cudecompTransposeYToX
_____________________
.. doxygenfunction:: cudecompTransposeYToX

------

Halo Exchange Functions
-----------------------

.. _cudecompUpdateHalosX-ref:

cudecompUpdateHalosX
____________________
.. doxygenfunction:: cudecompUpdateHalosX

------

.. _cudecompUpdateHalosY-ref:

cudecompUpdateHalosY
____________________
.. doxygenfunction:: cudecompUpdateHalosY

------

.. _cudecompUpdateHalosZ-ref:

cudecompUpdateHalosZ
____________________
.. doxygenfunction:: cudecompUpdateHalosZ
