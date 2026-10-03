import numpy as np
import matplotlib.pyplot as plt
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import PowerTransformer
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.colors as mc


fn_lengths = [[1., 1.], ## 1
              [1.5, 2.], ## 2
              [5., 1., .75], ## 3
              [1., 1.5, 2., 1.], ## 4
              [4., 2., 1., 1.],  ## 5
              [3., 5., 3., .75, .75], ## 6
              [1., 1., 1., 1., 1., 1.], ## 7
              [1., 1., 1., 2., 3., 5., 1., 3.]] ## 8

_eval_var = np.geomspace(1e-6, 5., 25)
_eval_ls = np.geomspace(1e-1, 10, 25)
_eval_matern = np.arange(0.5, 10.5, 0.5)
eval_grid = np.array(np.meshgrid(_eval_ls, _eval_matern, _eval_var)).T.reshape((-1, 3))

fn_arr = range(1, 9)

def nearest(array, value):
    return array[(np.abs(array - value)).argmin()]


for fn, length_mod in zip(fn_arr, fn_lengths):
    ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
    op = "initial_data\\function_%s\initial_outputs.npy" %fn
    inp = np.load(ip)
    out = np.load(op)
    new_inp = new_inputs[fn]
    new_out = new_outputs[fn]
    
    lengthscale = len(inp)**(-1/np.shape(inp)[-1])
    length_mod = np.array(length_mod)*lengthscale
    
    errors = []
    
    dims = inp.shape[-1]
    scales = np.array(length_mod)
    
    method = "box-cox" if min(np.sign(np.concat([out, new_out]))) == 1. else "yeo-johnson"
    if fn != 1:
        transformer = PowerTransformer(method=method)
        out = transformer.fit_transform(out.reshape((-1, 1)))
        new_out = transformer.transform(new_out.reshape((-1, 1))).flatten()
    else:
        out = out.reshape((-1, 1))
    
    for i, (scale_length, nu, var) in enumerate(eval_grid):
        kernel = kernels.Matern(length_scale=scales*scale_length, length_scale_bounds="fixed",
                                    nu=nu)
        model = GaussianProcessRegressor(kernel=kernel, alpha=var)
        ############################# REMEMBER TO FIX
        # kernel = kernels.Matern(length_scale=scales, length_scale_bounds="fixed",
        #                             nu=nu)
        # model = GaussianProcessRegressor(kernel=kernel, alpha=noise_var)
        model.fit(inp, out)
        sub_err = []
        
        for x, y in zip(new_inp, new_out):
            # samples = model.sample_y([x], n_samples=10)
            # sub_err.append(np.mean([(y-s)**2 for s in samples]))
            pred = model.predict([x])
            sub_err.append((y - pred[0])**2)
            model.fit([x], [y])
        errors.append(sub_err)
    
    errors = np.array(errors)
    errors = np.sqrt(errors.mean(axis=-1))
    delta = errors.max()-errors.min()
    
    best_band = errors < errors.min() + delta*0.05 ## within 95% of the best
    banded_ls = eval_grid[best_band, 0]
    banded_nu = eval_grid[best_band, 1]
    banded_var = eval_grid[best_band, 2]
    ls = banded_ls.mean()
    nu = banded_nu.mean()
    var = banded_var.mean()
        
    print("Best performance for function %s:" %fn)
    print("\tRMSE of %.3f vs mean of %.3f" %(errors.min(), errors.mean()))
    print("\tLength modifier: %s" %ls)
    print("\tKernel nu: %s" %nu)
    print("\tNoise var: %s" %var)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    
    norm = mc.Normalize(vmin=min(errors), vmax=max(errors))
    ax.scatter(eval_grid[:, 0], eval_grid[:, 1], eval_grid[:, 2], c=errors, norm=norm)
    ax.scatter(ls, nu, var, color="red")
    ax.set_xlabel("Length modifier")
    ax.set_ylabel("nu")
    ax.set_zlabel("Noise var")
    ax.set_xscale("log")
    ax.set_zscale("log")
    
    best_noise_level = nearest(eval_grid[:, 2], var)
    best_noise_mask = eval_grid[:,  2] == best_noise_level
    
    norm2 = mc.Normalize(vmin=min(errors[best_noise_mask]), vmax=max(errors[best_noise_mask]))
    ax2.scatter(eval_grid[best_noise_mask, 0], eval_grid[best_noise_mask, 1], c=errors[best_noise_mask], norm=norm2)
    ax2.scatter(ls, nu, color="red")
    ax2.set_xlabel("Length mod")
    ax2.set_ylabel("nu")
    ax2.set_xscale("log")
    
    plt.show()
    
