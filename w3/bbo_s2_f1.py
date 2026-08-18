from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler


base = Path("C:\\Users\\HP\\Documents\\y4\\ml_course\\bbo_s2\\")

ip = "initial_data\\function_1\\initial_inputs.npy"
op = "initial_data\\function_1\\initial_outputs.npy"


inp = np.load(base / ip)
out = np.load(base / op)

inp = np.concat([inp, new_inputs[1]])
out = np.concat([out, new_outputs[1]])


#%%
noise_std = .05
beta = 1
lengthscale = len(inp)**(-1/np.shape(inp)[-1])

kernel = kernels.Matern(length_scale=lengthscale, length_scale_bounds="fixed")
model = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=10)

n_grid = 100
eval_grid_c = np.linspace(0, 1, n_grid)
eg_x, eg_y = np.meshgrid(eval_grid_c, eval_grid_c)
eg = []
for i in range(len(eg_x)):
    for j in range(len(eg_y)):
        eg.append([eg_x[i][j], eg_y[i][j]])

#%%

scaler = StandardScaler()
out = np.array([i if i>=0. else 0. for i in out])
out = scaler.fit_transform(out.reshape((-1, 1)))
model.fit(inp, out)
mean, std = model.predict(eg, return_std=True)

#%%

eg_grid = np.reshape(eg, (n_grid, n_grid, 2))
x = eg_grid[:, :, 0]
y = eg_grid[:, :, 1]
std_plt = np.reshape(std, (n_grid, n_grid))
mean_plt = np.reshape(mean, (n_grid, n_grid))

plt.scatter(inp[:, 0], inp[:, 1], c=out)
# plt.contour(x, y, mean_plt)
# plt.title("Function 1 GP means")
plt.show()

# plt.scatter(inp[:, 0], inp[:, 1], c=out)
# plt.contour(x, y, std_plt)
# plt.title("Function 1 GP STDs")
# plt.show()

#%%
fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')


acquisition_func = mean + beta*std
acq_plt = np.reshape(acquisition_func, (n_grid, n_grid))

ax1.plot_surface(x, y, mean_plt, cmap="coolwarm")
ax1.set_title("Function 1 GP means")
ax1.set_xlabel("x")

ax2.plot_surface(x, y, std_plt, cmap="coolwarm")
ax2.set_title("Function 1 GP STDs")
ax2.set_xlabel("x")

ax3.plot_surface(x, y, acq_plt, cmap="coolwarm")
ax3.set_title("Function 1 GP acq. func")
ax3.set_xlabel("x")

plt.show()

#%%

idx = np.argmax(acquisition_func)
next_query = eg[idx]
print(next_query)