#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cpuinfo

def isIntel():
    """
    Inquires whether CPU is manufactured by Intel

    Parameters
    ----------
    (none)

    Returns
    -------
    ans:    bool
            True if Intel; False otherwise
    """
    sysString = cpuinfo.get_cpu_info()['brand']
    if 'Intel' in sysString:
        ans = True
    else:
        ans = False
    return ans

def isAMD():
    """
    Inquires whether CPU is manufactured by AMD

    Parameters
    ----------
    (none)

    Returns
    -------
    ans:    bool
            True if AMD; False otherwise
    """
    sysString = cpuinfo.get_cpu_info()['brand']
    if 'AMD' in sysString:
        ans = True
    else:
        ans = False
    return ans

def setenv(envlist):
    """
    Sets system variables while a Python for the execution of a Python
    script

    Parameters
    ----------
    envlist:    string list
                List containing environment variables to set. Each entry in
                the list is a [(ENV_VARIABLE, ENV_VAL)]

    Returns
    -------
    (none)      sets system variable

    """
    environ = dict(os.environ)
    i = 0
    while i < len(envlist):
        os.environ[envlist[i][]0] = envlist[i][1]