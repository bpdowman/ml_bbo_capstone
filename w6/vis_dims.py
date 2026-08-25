from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler


fn = 8

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\initial_outputs.npy" %fn

inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[fn]])
out = np.concat([out, new_outputs[fn]])
dims = inp.shape[-1]

scaler = StandardScaler()
out_scaled = scaler.fit_transform(out.reshape((-1, 1)))

fig, axes = plt.subplots(1, dims, figsize=(20, 4))
for i, axis in enumerate(axes):
    x = inp[:, i]
    
    axis.scatter(x, out_scaled)
    
plt.show()