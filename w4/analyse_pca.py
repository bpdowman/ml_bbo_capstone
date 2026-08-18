import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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

if analyse_scree:#
    plt.figure(figsize=(8, 4))

    n_components = 2
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(inp[:-1])
    
    plt.plot(range(1, n_components+1), pca.explained_variance_ratio_, label=sum(pca.explained_variance_ratio_))
    
    plt.ylim(0, 1)
    plt.xticks(range(1, n_components+1))
    plt.legend()
    plt.show()
    
else:
    fig, (ax_pca, ax_da, ) = plt.subplots(1, 2)
    ax_da.set_title("DA plot")
    ax_pca.set_title("PCA plot")
    
    
    dn = []
    for point1 in np.concat([inp, new_inputs[fn]]): ##ref point
        dt = []
        for point2 in np.concat([inp, new_inputs[fn]]):
            dist = np.linalg.norm(point1 - point2)
            dt.append(dist)
        dn.append(np.array(dt))
    dn = np.array(dn)
    mean_distances = dn.mean(axis=1) ## let us plot this as x
    max_distances = dn.max(axis=1) ## let us plot this as y (measure of centrality)
    n_new = len(new_outputs[fn])
    
    clusters = DBSCAN(eps=eps_dict[fn], min_samples=3, metric="precomputed").fit_predict(dn)
    colors = [ec_dict[i] for i in clusters]
    
    norm = mc.Normalize(vmin=min(out), vmax=max(out))
    ax_da.scatter(mean_distances[:-n_new], max_distances[:-n_new], norm=norm, c=out,
                edgecolors=colors[:-n_new], linewidths=2, s=150)
    ax_da.scatter(mean_distances[-n_new:], max_distances[-n_new:], norm=norm, c=new_outputs[fn],
                edgecolor="black", linewidths=2)
    
    
    
    pca = PCA(n_components=2, random_state=42)
    scaler = StandardScaler()
    pca.fit(inp[:-1])
    scaler.fit(inp[:-1])
    
    transformed_points = pca.transform(scaler.transform(inp[:-1]))
    points_x = transformed_points[:, 0]
    points_y = transformed_points[:, 1]
    transformed_points_wn = pca.transform(scaler.transform([inp[-1]]))
    points_x_wn = transformed_points_wn[:, 0]
    points_y_wn = transformed_points_wn[:, 1]
    
    norm = mc.Normalize(vmin=min(out[:-1]), vmax=max(out[:-1]))
    
    ax_pca.scatter(points_x, points_y, norm=norm, c=out[:-1], alpha=1)
    ax_pca.scatter(points_x_wn, points_y_wn, norm=norm, c=out[-1], 
                edgecolor="black")
