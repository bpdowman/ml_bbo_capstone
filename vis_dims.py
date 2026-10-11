from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data.bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler, PowerTransformer


fn = 6

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\initial_outputs.npy" %fn

inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[fn]])
out = np.concat([out, new_outputs[fn]])
dims = inp.shape[-1]

scaler = StandardScaler()
out_scaled = scaler.fit_transform(out.reshape((-1, 1)))
transformer = PowerTransformer()
out_transformed = transformer.fit_transform(out.reshape((-1, 1)))

fig, (ax1, ax2) = plt.subplots(2, dims, figsize=(2*dims, 4), layout="tight")

for i, axis in enumerate(ax1):
    x = inp[:, i]
    axis.scatter(x, out_scaled)
    axis.set_title("Scaled dim %s" %i)
    
for i, axis in enumerate(ax2):
    x = inp[:, i]
    axis.scatter(x, out_transformed)
    axis.set_title("Trans. dim %s" %i)

plt.show()