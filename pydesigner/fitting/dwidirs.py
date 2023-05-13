#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

working_dir = os.path.abspath(os.path.dirname(__file__))
dirs30 = np.genfromtxt(os.path.join(
   working_dir,'dirs30.csv'), delimiter=",")
dirs256 = np.genfromtxt(os.path.join(
   working_dir, 'dirs256.csv'), delimiter=",")
dirs10000 = np.genfromtxt(os.path.join(
   working_dir, 'dirs10000.csv'), delimiter=",")
sh_grid = np.genfromtxt(os.path.join(
   working_dir, 'spherical_grid.csv'), delimiter=",")
