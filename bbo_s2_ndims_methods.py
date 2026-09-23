from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.stats import norm
from scipy.stats import qmc

def fetch_function_data(n):

    ip = Path("initial_data\\function_%s\\initial_inputs.npy" %n)
    op = ("initial_data\\function_%s\\initial_outputs.npy" %n)

    inp = np.load(ip)
    out = np.load(op)
    
    return inp, out

def get_next_query(inp, out, lengthscale, n_grid, tune, 
                   n_random=20, nu=2.5, noise_var=.05, acq_type="ucb"):
    
    kernel = kernels.Matern(length_scale=lengthscale, length_scale_bounds="fixed",
                                nu=nu)
    
    model = GaussianProcessRegressor(kernel=kernel, alpha=noise_var)

    dims = np.shape(inp)[-1]
    if n_grid:
        eval_grid_c = np.linspace(0, 1, n_grid)
        axes = np.meshgrid(*[eval_grid_c]*dims,) ## meshgrid each item
        ## then stack them (combine axes into coordinate pairs)
        ## then reshape from (n_grid, n_grid,... dims) to (n_grid^dims, dims)
        grid = np.stack(axes, axis=-1).reshape((-1, dims))
    else:
        gen = qmc.Sobol(d=dims)
        grid = gen.random_base2(n_random)
        # grid = np.random.uniform(size=(int(n_random), dims))

    model.fit(inp, out)
    mean, std = model.predict(grid, return_std=True)

    ymax = out.max()

    if acq_type == "ucb":
        acquisition_func = mean + tune*std
    elif acq_type == "pi":
        acquisition_func = norm.cdf((mean - ymax - tune)/(std + 1e-12))
    elif acq_type == "ei":
        m = std > 0
        z = (mean - ymax)/std[m]
        ei = np.zeros_like(mean)
        ei[m] = (mean[m] - ymax)*norm.cdf(z) + std[m]*norm.pdf(z)
        acquisition_func = ei
    else:
        raise ValueError("Invalid acquisition function %s" %acq_type)
    
    idx = np.argmax(acquisition_func)
    next_query = grid[idx]
    next_query = [i-1e-6 if i==1. else i for i in next_query]
    
    if (pred := model.predict([next_query])[0]) < ymax:
        print("Query evaluated to %s - known best is %s" %(pred, ymax))
    
    return next_query

def rank_guess_values(guesses):
    ## how does each guess' magnitude rank against all points known magnitude (is it a potential peak?)
    indices = len(guesses) - 1 - guesses.argsort().argsort()
    ## now remap back to previous array
    return indices