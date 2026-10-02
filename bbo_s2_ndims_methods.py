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

def gen_sobol_constrained(d, lims, n_random):
    gen = qmc.Sobol(d)
    grid = np.array(gen.random_base2(n_random))
    grid *= (lims[:, 1] - lims[:, 0])
    grid += lims[:, 0]
    return grid

def get_next_query(inp, out, lengthscale, 
                   n_random=20, nu=2.5, noise_var=.05, correction=0.5):
    
    kernel = kernels.Matern(length_scale=lengthscale, length_scale_bounds="fixed",
                                nu=nu)
    
    model = GaussianProcessRegressor(kernel=kernel, alpha=noise_var)
    model.fit(inp, out)

    dims = np.shape(inp)[-1]
    lims = np.array([[0., 1.]]*dims)
    grid = gen_sobol_constrained(dims, lims, n_random)
    
    for _ in range(3):
        mean, std = model.predict(grid, return_std=True)
        ymax = out.max()
    
        m = std > 0
        z = (mean - ymax)/std[m]
        ei = np.zeros_like(mean)
        ei[m] = (mean[m] - ymax)*norm.cdf(z) + std[m]*norm.pdf(z)
    
        threshold = max(2**(n_random-4), 2**2)
        
        p = np.argsort(ei, descending=True)
        prelc = grid[p[:threshold]]
        lmins = prelc.min(axis=0)
        lmaxs = prelc.max(axis=0)

        lmins = correction*lmins
        lmaxs = np.clip(1/correction * lmaxs, 0., 1.)
    
        lims = np.hstack([lmins.reshape((-1, 1)), lmaxs.reshape((-1, 1))])
        
        grid = gen_sobol_constrained(dims, lims, n_random)
        
    mean, std = model.predict(grid, return_std=True)
    ymax = out.max()
    
    m = std > 0
    z = (mean - ymax)/std[m]
    ei = np.zeros_like(mean)
    ei[m] = (mean[m] - ymax)*norm.cdf(z) + std[m]*norm.pdf(z)

    idx = np.argmax(ei)
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