# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:32:42 2026

@author: HP
"""

from scipy.stats import qmc
import matplotlib.pyplot as plt

q = qmc.Sobol(d=2)
p = q.random_base2(8)
plt.scatter(p[:, 0], p[:, 1])
plt.show()