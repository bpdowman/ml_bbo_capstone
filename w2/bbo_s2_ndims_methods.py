# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:10:41 2026

@author: HP
"""
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.stats import norm

def fetch_function_data(n):
    base = Path("C:\\Users\\HP\\Documents\\y4\\ml_course\\bbo_s2\\")

    ip = "initial_data\\function_%s\\initial_inputs.npy" %n
    op = "initial_data\\function_%s\\initial_outputs.npy" %n

    inp = np.load(base / ip)
    out = np.load(base / op)
    
    return inp, out

def get_predictions(inp, out, noise_std, n_grid, n_random=1e6):
    noise_std = .05
    lengthscale = len(inp)**(-1/np.shape(inp)[-1])

    kernel = kernels.RBF(length_scale=lengthscale, length_scale_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2)

    dims = np.shape(inp)[-1]
    if n_grid:
        eval_grid_c = np.linspace(0, 1, n_grid)
        axes = np.meshgrid(*[eval_grid_c]*dims,) ## meshgrid each item
        ## then stack them (combine axes into coordinate pairs)
        ## then reshape from (n_grid, n_grid,... dims) to (n_grid^dims, dims)
        eg = np.stack(axes, axis=-1).reshape((-1, dims))
    else:
        eg = np.random.uniform(size=(int(n_random), dims))

    model.fit(inp, out)
    mean, std = model.predict(eg, return_std=True)
    
    return mean, std, eg

def query_point(mean, std, ymax, tune, grid, acq_type="ucb"):
    if acq_type == "ucb":
        acquisition_func = mean + tune*std
    elif acq_type == "pi":
        acquisition_func = norm.cdf((mean - ymax - tune)/(std + 1e-12))
    else:
        raise ValueError("Invalid acquisition function %s" %acq_type)
    
    
    idx = np.argmax(acquisition_func)
    next_query = grid[idx]
    next_query = [i-1e-6 if i==1. else i for i in next_query]
    return next_query

def rank_guess_values(initial, new_guesses):
    ## how does each guess' magnitude rank against all points known magnitude (is it a potential peak?)
    join = np.concat([initial, new_guesses])
    indices = len(join) - 1 - join.argsort().argsort()
    ## now remap back to previous array
    return indices[:len(initial)], indices[len(initial):]