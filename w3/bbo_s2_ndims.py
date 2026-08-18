from bbo_s2_ndims_methods import fetch_function_data, get_predictions, \
    query_point, rank_guess_values
from time import time
import numpy as np
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler


fn = range(1, 9)
fn_res = [100, 100, 100, 50, 
          50, 0, 0, 0]
n_random = 1e7
fn_tunes = [1., 1., 0.5, 0.05,
            0.05, 0.5, 1., 1.]
fn_types = ["ucb", "ucb", "ucb", "pi",
            "pi", "ucb", "ucb", "ucb"]
fn_lengths = [[1., 1.], ## 1
              [1., 1.], ## 2
              [5., 1., 1.], ## 3
              [1., 1.5, 5., 1.], ## 4
              [5., 2., 1., 1.],  ## 5
              [3., 5., 3., 1., 1.], ## 6
              [2., 3., 5., 3., 1., 2.], ## 7
              [1., 1., 1., 2., 3., 5., 1., 3.]] ## 8


ranking_new_guesses=False

for (f_number, grid_res, tune, atype, length_mod) in zip(fn, fn_res, fn_tunes, fn_types, fn_lengths):
    inp, out = fetch_function_data(f_number)
    inp = np.concat([inp, new_inputs[f_number]])
    out = np.concat([out, new_outputs[f_number]])
    n_new = len(new_outputs[f_number])
    
    lengthscale = len(inp)**(-1/np.shape(inp)[-1])
    scales = lengthscale*np.array(length_mod)
    
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
    
    if not ranking_new_guesses: ## finding new points
        ##preprocessing
        if f_number == 1:
            out = np.array([i if i>=0. else 0. for i in out])
            kerneltype = "matern"
        else:
            kerneltype = "rbf"
            
        scaler = StandardScaler()
        out = scaler.fit_transform(out.reshape((-1, 1)))
        ts = time()
        mean, std, grid = get_predictions(inp, out, scales, 
                                          grid_res, kerneltype, n_random)
        
        qp = query_point(mean, std, out.max(), tune, grid, acq_type=atype)
        tend = time()

        print("Suggested point for function %s (res %s) is " %(f_number, grid_res), end=None)
        divs = ["-"]*(len(qp)-1)
        divs = np.concat([divs, ["\n"]])
        for (point, divider) in zip(qp, divs):
            print("%.6f" %point, end=divider)
        print("Took %s seconds" %(round(tend-ts, 3)))
        print("")