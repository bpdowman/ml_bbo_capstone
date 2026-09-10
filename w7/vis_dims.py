from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from bbo_data_dict import new_inputs, new_outputs
from sklearn.preprocessing import StandardScaler, power_transform


fn = 7

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\initial_outputs.npy" %fn

inp = np.load(ip)
out = np.load(op)

inp = np.concat([inp, new_inputs[fn]])
out = np.concat([out, new_outputs[fn]])
dims = inp.shape[-1]

scaler = StandardScaler()
out_scaled = scaler.fit_transform(out.reshape((-1, 1)))
sign = np.sign(out)
trans_inp = out.reshape((-1, 1))
if min(sign) == 1.:
    method = "box-cox"
elif max(sign) == -1.:
    method = "box-cox"
    trans_inp = -trans_inp
else:
    method = "yeo-johnson"
out_transformed = power_transform(trans_inp, method)

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