from bbo_s2_ndims_methods import fetch_function_data, get_predictions, query_point
from time import time
import numpy as np

fn = range(1, 9)
fn_res = [100, 100, 100, 50, 
          50, 20, 10, 5]
fn_stds = [1e-16, .05, 0.025, 2,
           10, .5, .5, .001, .05]

for (f_number, grid_res, noise_std) in zip(fn, fn_res, fn_stds):
    ts = time()
    inp, out = fetch_function_data(f_number)
    mean, std, grid = get_predictions(inp, out, noise_std, grid_res)
    qp = query_point(mean, std, 1., grid)
    tend = time()
    
    print("Suggested point for function %s (res %s) is " %(f_number, grid_res), end=None)
    divs = ["-"]*(len(qp)-1)
    divs = np.concat([divs, ["\n"]])
    for (point, divider) in zip(qp, divs):
        print("%.6f" %point, end=divider)
    print("Took %s seconds" %(round(tend-ts, 3)))
    print("")