from scipy.io import loadmat, savemat
import numpy as np
import time
import sys
import InVaXMap

system_type = 'cyc'
for N in range(10,101,10):
    file_path = './cyc_ran/'+system_type+'_data/'+system_type+'_N_'+str(N)+'_trails_100.mat'
    file = loadmat(file_path, mat_dtype=True)

    save_path = './cyc_ran/'+system_type+'_result/'

    data = file[system_type+'_data']
    st = file[system_type+'_st']

    se_len, species, trails = np.shape(data)

    # tr is number of trails, which can be set as a specific number.
    batch_num = sys.argv[1]
    tr = int(batch_num)

    maxdelay=10; E=4; tau=1; th=0.5

    start = time.time()
    try:
        ic, td, fc, ds = InVaXMap.InVaXMap(G = data[:,:,tr], maxdelay = maxdelay, E = E, tau = tau, th = th)
        AUC_ic, tpr_ic, fpr_ic, acc_ic = InVaXMap.eva_cri(ic,st[:,:,tr])
        AUC_fc, tpr_fc, fpr_fc, acc_fc = InVaXMap.eva_cri(fc,st[:,:,tr])
        var_dict = {'ic':ic, 'td':td, 'fc':fc, 'ds':ds, 'AUC_ic':AUC_ic, 'tpr_ic':tpr_ic, 'fpr_ic':fpr_ic, 'acc_ic':acc_ic, 'AUC_fc':AUC_fc, 'tpr_fc':tpr_fc, 'fpr_fc':fpr_fc, 'acc_fc':acc_fc}
        #   savemat(save_path+'result_'+system_type+'_N_'+str(N)+'_trail_' + str(tr) + '.mat', var_dict)
    except:
        print('system_dim: '+str(N)+'; tr='+str(tr)+': Not get result!')

    end = time.time()
    print('system_dim: '+str(N)+'; trail_{a}_cost: {b:.3f}s\n'.format(a=tr, b=end-start))
