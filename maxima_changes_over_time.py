# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:25:33 2026

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from data.bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import PowerTransformer
from bbo_s2_ndims_methods import fetch_function_data

f_number = range(1, 9)

fig, ax_arr = plt.subplots(2, 4, figsize=(10, 4), layout="tight",
                           dpi=800)
base_points = range(0, 14)

for fn in f_number:
    inp, out = fetch_function_data(fn)
    
    best = []
    best.append(out.max())
    for point in new_outputs[fn]:
        if point > best[-1]:
            best.append(point)
        else:
            best.append(best[-1])
        
    ax_arr[int(fn>4)][(fn-1)%4].plot(base_points, best)
    ax_arr[int(fn>4)][(fn-1)%4].set_title("F%s" %fn)
    ax_arr[int(fn>4)][(fn-1)%4].vlines(7, 
                                       np.min(best), np.max(best),
                                       linestyle="-",
                                       color="grey")
    print("Function %s best point: %s" %(f_number, best[-1]))