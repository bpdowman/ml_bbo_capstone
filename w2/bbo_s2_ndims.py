from bbo_s2_ndims_methods import fetch_function_data, get_predictions, \
    query_point, rank_guess_values
from time import time
import numpy as np

fn = range(1, 9)
fn_res = [100, 100, 100, 50, 
          50, 0, 0, 0]
n_random = 1e6
fn_stds = [1e-16, .05, 0.025, 2,
           10, .5, .5, .001, .05]
fn_tunes = [1., 1., 0.5, 0.05,
            0.05, 0.5, 1., 1.]
fn_types = ["ucb", "ucb", "ucb", "pi",
            "pi", "ucb", "ucb", "ucb"]

fn_new_points = {
    1: [[[0.000000, 0.999999]], 
        [0]],
    2: [[[0.050505, 0.858586]],
        [0.09280926108610274]],
    3: [[[0.999999, 0.000000, 0.848485]],
        [-0.11964464525688387]],
    4: [[[0.489796, 0.510204, 0.387755, 0.448980]],
        [-2.5895118693066865]],
    5: [[[0.367347, 0.857143, 0.999999, 0.999999]],
        [3042.6593829979597]],
    6: [[[0.052632, 0.578947, 0.000000, 0.999999, 0.000000]],
        	[-1.5068013636321904]],
    7: [[[0.000000, 0.222222, 0.333333, 0.000000, 0.333333, 0.999999]],
        [0.5042175846203388]],
    8: [[[0.250000, 0.250000, 0.250000, 0.250000, 0.500000, 0.500000, 0.500000, 0.500000]],
        [9.6658]]
    }

ranking_new_guesses=False


for (f_number, grid_res, noise_std, tune, atype) in zip(fn, fn_res, fn_stds, fn_tunes, fn_types):
    inp, out = fetch_function_data(f_number)
    if ranking_new_guesses:
        indices_init, indices_new = rank_guess_values(out, fn_new_points[f_number][1])
        print("New point for function %s was ranked at #%s out of %s points" 
              %(f_number, indices_new[-1], len(indices_init+indices_new)))
        if indices_new[-1] == 0:
            n_best = fn_new_points[f_number][1][-1]
            if 1 in indices_init:
                p_best = out[np.where(indices_init == 1)[0]][0]
            else:
                p_best = fn_new_points[f_number][1][np.where(indices_new==1)[0]]
            print("\tNew point of value %s beat the old best %s by %s" 
                  %(n_best, p_best, n_best - p_best))
    
    if not ranking_new_guesses: ## finding new points
        ts = time()
        inputs = np.concat([inp, fn_new_points[f_number][0]])
        outputs = np.concat([out, fn_new_points[f_number][1]])
        mean, std, grid = get_predictions(inputs, outputs, noise_std, grid_res, n_random)
        
        qp = query_point(mean, std, outputs.max(), tune, grid, acq_type=atype)
        tend = time()
        
        print("Suggested point for function %s (res %s) is " %(f_number, grid_res), end=None)
        divs = ["-"]*(len(qp)-1)
        divs = np.concat([divs, ["\n"]])
        for (point, divider) in zip(qp, divs):
            print("%.6f" %point, end=divider)
        print("Took %s seconds" %(round(tend-ts, 3)))
        print("")