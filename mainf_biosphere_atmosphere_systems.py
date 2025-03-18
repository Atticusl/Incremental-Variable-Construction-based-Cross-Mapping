import pandas as pd
import InVaXMap
import numpy as np
from scipy.io import savemat

def cal(data, start,end):
    '''
    Input
    season - character string
    start - start date
    end - end date
    '''
    maxdelay=10; E=3; tau=1; th = 0.35
    G = data[(data['TIMESTAMP']>=start) & (data['TIMESTAMP']<end)].to_numpy()
    ic, td, fc, _ = InVaXMap.InVaXMap(G = G[:,1:], maxdelay = maxdelay, E = E, tau = tau, th = th)
    return ic, td, fc


sites = ['CN-Din', 'CN-Qia', 'CN-Ha2', 'CN-Cha']
labels = ['TIMESTAMP','NEE_CUT_REF', 'TA_ERA','SWC_F_MDS_1','VPD_ERA','P_ERA','SW_IN_ERA','LW_IN_ERA','PA_ERA','WS_ERA','LE_F_MDS','H_F_MDS']
xt = ['NEE','TA','SWC','VPD','P','SW','LW','PA','WS','LE','H']
dic = {
    '2003':[20030101,20040101],
    '2004':[20040101,20050101],
    '2005':[20050101,20060101]
    }

for it in range(len(sites)):
    df = pd.read_csv('Biosphere-atmosphere systems/FLX_'+sites[it]+'_FLUXNET2015_FULLSET_DD_2003-2005_1-4.csv')
    a = df[labels]

    ic = np.empty((len(labels)-1,len(labels)-1,len(dic)))
    td = np.empty((len(labels)-1,len(labels)-1,len(dic)))
    fc = np.empty((len(labels)-1,len(labels)-1,len(dic)))
    k = 0
    for i in dic.keys():
        ic[:,:,k], td[:,:,k], fc[:,:,k] = cal(data=a, start=dic[i][0], end=dic[i][1])
        k += 1

    ## save as .mat
    # var_dict = {'ic':ic, 'td':td, 'fc':fc}
    # savemat('./result_'+sites[it]+',mat', var_dict)
