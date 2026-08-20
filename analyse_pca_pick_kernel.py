import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from sklearn.decomposition import KernelPCA
import optuna

from bbo_data_dict import new_inputs, new_outputs

ec_dict = {-1: "b",
           0 : "r",
           1 : "g",
           2 : "y",
           3 : "m",
           4 : "c",
           5 : "w"}

eps_dict = {1: 0.2, 2 : 0.2, 3 : 0.3, 4 : 0.35,
            5 : 0.4, 6 : 0.5, 7 : 0.6, 8 : 0.65}

fn = 8

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\\initial_outputs.npy" %fn

analyse_scree = False

inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[fn]])
out = np.concat([out, new_outputs[fn]])
    

# def get_pca_reconstruct_distances(kernel, gamma, degree, alpha):
#     pca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, degree=degree,
#                     fit_inverse_transform=True, alpha=alpha, random_state=42)
#     pca.fit(inp[:-1])
#     inv = pca.inverse_transform(pca.transform(inp))
#     distances = abs(inp - inv)
    
#     distances_avg = np.mean(np.linalg.norm(distances, axis=1))
#     return distances_avg/np.sqrt(fn)
    
# def objective(trial):
#     kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"])
#     gamma = trial.suggest_float("gamma", 0.001, 5)
#     degree = trial.suggest_int("degree", 2, 3)
#     alpha = trial.suggest_float("alpha", .01, 5)
    
#     return get_pca_reconstruct_distances(kernel, gamma, degree, alpha)

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# print(study.best_params)



configs = {1:...,
           2:...,
           3:["poly", 3, .85, .0016],
           4:["poly", 3, .87, .019],
           5:["poly", 2, .39, .13],
           6:["rbf", 3, 1.17, .018],
           7:["rbf", 3, 2.622, .014],
           8:["poly", 3, .499, .041],}

kernel, degree, gamma, alpha = configs[fn]

pca = KernelPCA(n_components=2, kernel=kernel, gamma=gamma, 
                degree=degree, alpha=alpha, random_state=42,
                fit_inverse_transform=True)
pca.fit(inp[:-1])


fig = plt.Figure()
plt.title("PCA plot")

transformed_points = pca.transform(inp[:-1])
points_x = transformed_points[:, 0]
points_y = transformed_points[:, 1]
transformed_points_wn = pca.transform([inp[-1]])
points_x_wn = transformed_points_wn[:, 0]
points_y_wn = transformed_points_wn[:, 1]

norm = mc.Normalize(vmin=min(out[:-1]), vmax=max(out[:-1]))

plt.scatter(points_x, points_y, norm=norm, c=out[:-1], alpha=1)
plt.scatter(points_x_wn, points_y_wn, norm=norm, c=out[-1], 
            edgecolor="black")

plt.show()

points_to_retransform = {3:[-.6, .772],
                         4:[-.359, .151],
                         6:[.337, -.231],
                         7:[.432, .169],
                         8:[-.807, -0.365]}

if fn in points_to_retransform:
    qp = pca.inverse_transform([points_to_retransform[fn]])[0]
    divs = ["-"]*(len(qp)-1)
    divs = np.concat([divs, ["\n"]])
    for (point, divider) in zip(qp, divs):
        print("%.6f" %point, end=divider)