"""
Runs the PyDesigner pipeline
"""

#---------------------------------------------------------------------- 
# Package Management
#----------------------------------------------------------------------
import os # path
import shutil # which

# Locate mrtrix3 via which-ing dwidenoise
dwidenoise_location = shutil.which('dwidenoise')
if dwidenoise_location == None:
    raise Exception('Cannot find mrtrix3, please see '
        'https://github.com/m-ama/PyDesigner/wiki'
        ' to troubleshoot.')

# Extract mrtrix3 path from dwidenoise_location
mrtrix3path = os.path.dirname(dwidenoise_location)

# Locate FSL via which-ing fsl
fsl_location = shutil.which('fsl')
if fsl_location == None:
    raise Exception('Cannot find FSL, please see '
        'https://github.com/m-ama/PyDesigner/wiki'
        ' to troubleshoot.')

# Extract FSL path from fsl_location
fslpath = os.path.dirname(fsl_location)
