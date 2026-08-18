from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

base = Path("C:\\Users\\HP\\Documents\\y4\\ml_course\\bbo_s2\\")

ip = "initial_data\\function_3\\initial_inputs.npy"
op = "initial_data\\function_3\\initial_outputs.npy"


inp = np.load(base / ip)
out = np.load(base / op)


#%%
noise_std = .05
lengthscale = len(inp)**(-1/np.shape(inp)[-1])

kernel = kernels.RBF(length_scale=lengthscale, length_scale_bounds="fixed")
model = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2)

#%%

n_grid = 100
dims = np.shape(inp)[-1]
eval_grid_c = np.linspace(0, 1, n_grid)
axes = np.meshgrid(*[eval_grid_c]*dims,) ## meshgrid each item
## then stack them (combine axes into coordinate pairs)
## then reshape from (n_grid, n_grid,... dims) to (n_grid^dims, dims)
eg = np.stack(axes, axis=-1).reshape((-1, dims))
print(len(eg))

#%%

beta = .05

model.fit(inp, out)
mean, std = model.predict(eg, return_std=True)

acquisition_func = mean + beta*std
#%%
idx = np.argmax(acquisition_func)
next_query = eg[idx]
print(next_query)