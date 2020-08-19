Dataset Requirements
====================

PyDesigner can process input DWIs in NifTi (.nii), compressed NifTi (.nii.gz), MRTrix3
file format (.mif), and DICOM (.dcm) file formats. With the exception of :code:`.mif`
and :code:`.dcm` filetypes, all other input formats are required to be accompanied with
:code:`.bval`, :code:`.bvec`, and :code:`.json` files.

**Note**: With the exception of extensions, all files additional accompanying a DWI need
to have the same name as DWI. For example, the input DWI file :code:`DKI_64_dir.nii` will
be accompanied by :code:`DKI_64_dir.bval`, :code:`DKI_64_dir.bvec` and :code:`DKI_64_dir.json`
files

Separate or Combined Shells
---------------------------

Having B-value shells in separate or single 4D volumes doesn't matter as long as each 4D DWI
has it's own accompanying files.

JSON File
---------

Every DWI will NEED a :code:`.json` file of the same name; PyDesigner will refuse to process
any input that fails to meet this criterion. This behavior is intentional to prevent unintentional
corrections from being exectuted when they are incompatible. Users must create a JSON file if
their DICOM to NifTi conversion software fails to create it.

PyDesigner primarily looks for partial Fourier information within a JSON information. This information
if encoded in the fields :code:`PartialFourier`; or :code:`PhaseEncodingSteps` and
:code:`AcquisitionMatrixPE`. Users need to have have these fields at the bare minimum to process DWIs.

