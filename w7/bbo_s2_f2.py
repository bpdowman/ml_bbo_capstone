import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler

ip = "initial_data\\function_2\\initial_inputs.npy"
op = "initial_data\\function_2\\initial_outputs.npy"


inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[2]])
out = np.concat([out, new_outputs[2]])



noise_var = .6
beta = 1.
lengthscales = np.array([1.5, 2.])

kernel = kernels.Matern(length_scale=lengthscales*0.5, 
                        length_scale_bounds="fixed",
                        nu=2.5)
model = GaussianProcessRegressor(kernel=kernel, alpha=noise_var,
                                 n_restarts_optimizer=10)

n_grid = 100
eval_grid_c = np.linspace(0, 1, n_grid)
eg_x, eg_y = np.meshgrid(eval_grid_c, eval_grid_c)
eg = []
for i in range(len(eg_x)):
    for j in range(len(eg_y)):
        eg.append([eg_x[i][j], eg_y[i][j]])


scaler = StandardScaler()
out_reshape = scaler.fit_transform(out.reshape((-1, 1)))
model.fit(inp, out_reshape)
mean, std = model.predict(eg, return_std=True)



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


fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')


acquisition_func = mean + beta*std
acq_plt = np.reshape(acquisition_func, (n_grid, n_grid))

ax1.plot_surface(x, y, mean_plt, cmap="coolwarm")
ax1.view_init(45, -45, 0)
ax1.set_title("Function 2 GP means")
ax1.set_xlabel("x")

ax2.plot_surface(x, y, std_plt, cmap="coolwarm")
ax2.set_title("Function 2 GP STDs")
ax2.set_xlabel("x")

ax3.plot_surface(x, y, acq_plt, cmap="coolwarm")
ax3.set_title("Function 2 GP acq. func")
ax3.set_xlabel("x")

plt.show()



idx = np.argmax(acquisition_func)
next_query = eg[idx]
print(next_query)