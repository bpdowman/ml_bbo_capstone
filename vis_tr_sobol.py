from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.stats import norm
from bbo_data_dict import new_inputs, new_outputs

n_random = 12
n = dims = 2

l = [1.5, 2.] ## 2

c = .5
nu = 4.5
noise_var = .5

correction = 0.5

ip = "initial_data\\function_2\\initial_inputs.npy"
op = "initial_data\\function_2\\initial_outputs.npy"


inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[2]])
out = np.concat([out, new_outputs[2]])


ls = len(inp)**(-1/2)
lengthscale = ls*np.array(l)*c

kernel = kernels.Matern(length_scale=lengthscale, length_scale_bounds="fixed",
                            nu=nu)

model = GaussianProcessRegressor(kernel=kernel, alpha=noise_var)

def gen_sobol_constrained(d, lims, n_random):
    gen = qmc.Sobol(d)
    grid = np.array(gen.random_base2(n_random))
    grid *= (lims[:, 1] - lims[:, 0])
    grid += lims[:, 0]
    return grid

lims = np.array([[0., 1.]]*dims)

for _ in range(3):
    grid = gen_sobol_constrained(dims, lims, n_random)
    plt.scatter(grid[:, 0], grid[:, 1])
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    plt.show()
    
    model.fit(inp, out)
    mean, std = model.predict(grid, return_std=True)
    ymax = out.max()

    m = std > 0
    z = (mean - ymax)/std[m]
    ei = np.zeros_like(mean)
    ei[m] = (mean[m] - ymax)*norm.cdf(z) + std[m]*norm.pdf(z)

    l = int(np.sqrt(len(ei)))
    x = grid[..., 0]
    y = grid[..., 1]
    plt.scatter(x, y, c=ei)
    plt.show()
    
    
    m = ei.max()
    n_items = len(grid)
    a = lims[:, 0]
    b = lims[:, 1]
    threshold = max(2**(n_random-4), 2**2)
    
    p = np.argsort(ei, descending=True)
    prelc = grid[p[:threshold]]
    lmins = prelc.min(axis=0)
    lmaxs = prelc.max(axis=0)
    
    lmins = correction*lmins
    lmaxs = np.clip(1/correction * lmaxs, 0., 1.)

    lims = np.hstack([lmins.reshape((-1, 1)), lmaxs.reshape((-1, 1))])
    print(lims)
    print("\n")
    

