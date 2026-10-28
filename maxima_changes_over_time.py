# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:25:33 2026

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from data.bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import FunctionTransformer
from bbo_s2_ndims_methods import fetch_function_data
import numpy as np

f_number = range(1, 9)

fig, ax_arr = plt.subplots(2, 4, layout="tight")
base_points = range(0, 14)

for fn in f_number:
    inp, out = fetch_function_data(fn)
    
    lt = FunctionTransformer(np.log1p, validate=True)
    lt.fit(np.concat([out, new_outputs[fn]]).reshape((-1, 1)))
    out = lt.transform(out).reshape((-1))
    new_out = lt.transform(new_outputs[fn]).reshape((-1))
    
    best = []
    best.append(out.max())
    for point in new_out:
        if point > best[-1]:
            best.append(point)
        else:
            best.append(best[-1])
        
    ax_arr[int(fn>4)][(fn-1)%4].plot(base_points, best)
    ax_arr[int(fn>4)][(fn-1)%4].set_title("F%s" %fn)