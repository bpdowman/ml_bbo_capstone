from bbo_s2_ndims_methods import fetch_function_data, rank_guess_values, get_next_query
from time import time
import numpy as np
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import PowerTransformer


fn = range(1, 9)
fn_res = [100, 100, 100, 50, 
          50, 0, 0, 0]
n_random = 23 ## approx 1e7
fn_tunes = [1., 1., 0.05, 0.05,
            0.05, 0.025, .05, .05]
# fn_types = ["ucb", "ucb", "pi", "pi",
#             "pi", "pi", "pi", "pi"]
fn_types = ["ei", "ei", "ei", "ei",
            "ei", "ei", "ei", "ei"]
fn_lengths = [[1., 1.], ## 1
              [1.5, 2.], ## 2
              [5., 1., .75], ## 3
              [1., 1.5, 2., 1.], ## 4
              [4., 2., 1., 1.],  ## 5
              [3., 5., 3., .75, .75], ## 6
              [1., 1., 1., 1., 1., 1.], ## 7
              [1., 1., 1., 2., 3., 5., 1., 3.]] ## 8
fn_cs = [1., 0.5, 1, 0.5, 
          1., 3., 0.4, 1.]
fn_nus = [1.5, 1.5, 1.5, 2.5,
          1.5, 2.5, 2.5, 1.5]
fn_vars = [.001, .5, .5, .007,
           .002, .07, .025, .007]

ranking_new_guesses=False

for (f_number, grid_res, tune, atype, length_mod, length_c, nu, noise_var) in \
        zip(fn, fn_res, fn_tunes, fn_types, fn_lengths, fn_cs, fn_nus, fn_vars):
    inp, out = fetch_function_data(f_number)
    inp = np.concat([inp, new_inputs[f_number]])
    out = np.concat([out, new_outputs[f_number]])
    n_new = len(new_outputs[f_number])
    
    lengthscale = len(inp)**(-1/np.shape(inp)[-1])
    scales = lengthscale*np.array(length_mod)*length_c

    if ranking_new_guesses:
        indices = rank_guess_values(out)
        print("New point for function %s was ranked at #%s out of %s points" 
              %(f_number, indices[-1]+1, len(indices)))
        if indices[-1] == 0:
            n_best = out[-1]
            p_best = out[np.where(indices == 1)[0][0]]
            delta = n_best - p_best
            print("\tNew point of value %.2f beat the old best %.2f by %.2f - %.2f%% increase" 
                  %(n_best, p_best, delta, np.abs(delta/p_best)*100))
        else:
            c_best = np.max(out)
            n_guess = out[-1]
            delta = c_best - n_guess
            print("\tNew point of value %.2f underperformed the best guess %.2f by %.2f (%.2f%%)" 
                  %(n_guess, c_best, delta, np.abs(delta/c_best)*100))
    
    if not ranking_new_guesses: ## finding new points

        transformer = PowerTransformer()
        out = transformer.fit_transform(out.reshape((-1, 1)))
        ts = time()
        
        qp = get_next_query(inp, out, scales, grid_res,
                            tune, n_random, nu, noise_var, acq_type=atype)
        
        tend = time()

        print("Suggested point for function %s (res %s) is " %(f_number, grid_res), end=None)
        divs = ["-"]*(len(qp)-1)
        divs = np.concat([divs, ["\n"]])
        for (point, divider) in zip(qp, divs):
            print("%.6f" %point, end=divider)
        print("Took %s seconds" %(round(tend-ts, 3)))
        print("")