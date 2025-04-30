__mindtibval__ = 0.5  # minimum DTI B-value
__maxdtibval__ = 1.5  # maximum DTI B-value
__mindkibval__ = 0.5  # minimum DKI B-value
__maxdkibval__ = 3.0  # maximum DKI B-value
__minfbibval__ = 4.0  # minimum FBI B-value
__maxfbibval__ = 12.0  # maximum FBI B-value
__d0__ = 3.0  # diffusivity of free water at body temp (37 deg C) for upper bound on Da
__dn__ = 1.5  # estimated intra-neurite diffusivity in mm^2/ms
__pkT__ = 0.4  # peak thresholding for white matter fiber tracking
__minZero__ = 10e-8  # threshold under which all numbers are zero
__dirs__ = 256  # Define number of directions to resample after computing all tensors
# DTI windows
__fa__ = [0, 1]
__md__ = [0, 5]
__rd__ = [0, 5]
__ad__ = [0, 5]
# DKI windows
__kfa__ = [0, 1]
__mk__ = [0, 10]
__rk__ = [0, 10]
__ak__ = [0, 10]
# WMTI windows
__wawf__ = [0, 1]
__eas_rd__ = [0, 5]
__eas_ad__ = [0, 5]
__tort__ = [0, 100]
__ias_da__ = [0, 5]
 #FBI windows
__zeta__ = [0, 10]
__faa__ = [0, 1]
# FBWM windows
__fawf__ = [0, 1]
__da__ = [0, 5]
__de_mean__ = [0, 5]
__de_rad__ = [0, 5]
__de__ax__ = [0, 5]
__de_fa__ = [0, 1]
