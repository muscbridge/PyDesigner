#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cpuinfo

def isIntel():
    """
    Inquires whether CPU is manufactured by Intel

    Returns
    -------
    ans : bool
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

    Returns
    -------
    ans : bool
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
    envlist : list of str
    List containing environment variables to set. Each entry in the
    list is a [(ENV_VARIABLE, ENV_VAL)]

    Returns
    -------
    None; sets environment variables defined in `envlist`

    """
    environ = dict(os.environ)
    i = 0
    while i < len(envlist):
        # If env variable does not exist
        if envlist[i][0] not in environ:
            os.environ[envlist[i][0]] = envlist[i][1]
        # if env variable exists but has a different value
        elif (envlist[i][0] in environ) and \
                (envlist[i][1] not in environ[envlist[i][0]]):
            os.environ[envlist[i]] = envlist[i][1]
        i += 1