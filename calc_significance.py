# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:47:12 2026

@author: HP
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bbo_data_dict import new_inputs, new_outputs
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm


fn = 2
base = Path("C:\\Users\\HP\\Documents\\y4\\ml_course\\bbo_s2\\")

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\initial_outputs.npy" %fn

inp = np.load(base / ip)
out = np.load(base / op)

inp = np.concat([inp, new_inputs[fn]])
out = np.concat([out, new_outputs[fn]])

combined = np.concat([inp, out.reshape(-1, 1)], axis=1)
data = pd.DataFrame(data=combined)
data_inp = data.drop(len(inp[1]), axis=1)
data_out = data[len(inp[1])]

z_scores = data_inp.apply(zscore)

X_const = sm.add_constant(data_inp) ## x intercept
model = sm.OLS(data_out, X_const).fit()

# print(z_scores)
print(model.summary())