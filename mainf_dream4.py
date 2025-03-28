from scipy.io import loadmat
import numpy as np
import time
import InVaXMap

# selected genes
dataset = loadmat('data_path', mat_dtype=True) # replace 'data_path' with actual data directory
data = dataset['data']
st = dataset['st']

for i in range(5):
    maxdelay=5; E=3; tau=1; th=0.7
    start = time.time()
    ic, td, fc, _ = InVaXMap.InVaXMap(G = data[:,:,i], maxdelay = maxdelay, E = E, tau = tau, th = th)
    print(InVaXMap.eva_cri(fc,st[:,:,i])[0])
    end = time.time()
    print("time cost: {:.3f}s\n".format(end-start))
