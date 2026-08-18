from pathlib import Path
import numpy as np

fn = 2
base = Path("C:\\Users\\HP\\Documents\\y4\\ml_course\\bbo_s2\\")

ip = "initial_data\\function_%s\\initial_inputs.npy" %fn
op = "initial_data\\function_%s\\initial_outputs.npy" %fn


inp = np.load(base / ip)
out = np.load(base / op)

print(out)