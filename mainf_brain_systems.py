from scipy.io import loadmat, savemat
import numpy as np
import time
import InVaXMap

start = time.time()

data = loadmat('data_path', mat_dtype=True) # replace 'data_path' with actual data directory

data_alc = data['Alcoholic_dataset']

# sub is serial number of subjects
# tr is trail number of each subject
sub = 1; tr = 1

regions = np.concatenate((np.arange(0,17), np.arange(31,41), np.arange(41,49)))
maxdelay=10; E=4; tau=1; th=0.3; person = [2,4,6,7,8,9,10]

a_ic, a_td, a_fc, a_ds = InVaXMap.InVaXMap(G = data_alc[:,regions,tr,sub], maxdelay = maxdelay, E = E, tau = tau, th = th)

end = time.time()
print("time cost: {:.3f}s\n".format(end-start))
