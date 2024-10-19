import numpy as np
from scipy import stats

def InVaXMap(G, maxdelay, E, tau, th='mean'):
    '''
    InVaXMap method

    Parameters
    ----------
    G: time series matrix with length L * processes n
    max_lag: max time lag
    tau: time lag, usually setting tau=1 
    E: embedding dimension
    th: offspring threshold
    
    Return
    ----------
    Ini_causal: initial causal matrix
    time_delay: time delay matrix
    fc: Causal matrix with causal_effect(i,j) meaning causal effect from i to j
    '''
    L = np.size(G,0); n = np.size(G,1)
    lag_interval = [-maxdelay,0]
    ori_data = G[range(lag_interval[1]+(E-1),L-(-lag_interval[0])),:]
    [Ini_causal, time_delay, esti_data, pvalue] = ccm_esti(G, maxdelay, tau, E)
    fc = np.zeros((n,n))
    descendant_set = []
    for i in range(n):
        if th == 'mean':
            th = np.mean(Ini_causal[i,:])
        sub = np.linspace(0,n-1,n).astype(int); sub = np.delete(sub,i)
        offspring = []
        k = 1
        while sub.size != 0 or k >= n:
            matrix = np.vstack((sub, time_delay[i,sub], Ini_causal[i,sub]))
            if matrix.size == 0: break
            tar = int(select_var(matrix))
            if Ini_causal[i,tar] > th:
                    if offspring == []:
                        offspring.append(tar)
                    else:
                        r = partial_corr(ori_data[:,i], esti_data[:,tar,i], np.transpose(esti_data[:,offspring,i]))
                        if np.abs(r) > th:
                            offspring.append(tar)
            sub = np.delete(sub, sub==tar)
            k += 1

        if offspring == []:
            fc[i,:] = Ini_causal[i,:]
        else:
            for j in range(n):
                if j != i:
                    if j in offspring:
                        offspring.remove(j)
                        fc[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,offspring,i]))
                        offspring.append(j)
                    else:
                        fc[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,offspring,i]))
        descendant_set.append(offspring)
    return abs(Ini_causal), time_delay, abs(fc), descendant_set

def ccm_esti(G, max_lag, tau, E):
    '''
    CCM causal inference

    Parameters
    ----------
    G: time series matrix with length L * processes n
    max_lag: max time lag
    tau: time lag, usually setting tau=1 
    E: embedding dimension

    Return
    ----------
    causal_effect: Causal matrix with causal_effect(i,j) meaning causal effect from i to j
    time_delay: time delay matrix with time_delay(i,j) meaning time delay corespongding to causal_effect(i,j)
    est_series: estimated series with est_series(:,j,i) meaning estimated i using j
    '''
    L = np.size(G,0); n = np.size(G,1)
    lag_interval = [-max_lag,0]
    L_seri = L - lag_interval[1] + lag_interval[0] - (E-1)
    causal_effect = np.zeros((n,n)); time_delay = np.zeros((n,n))
    est_series = np.zeros((L_seri,n,n))
    for i in range(n):
        for j in range(i+1,n):
            [causal_effect[i,j], time_delay[i,j], est_series[:,j,i]] = lag_causal(G[:,i], G[:,j], lag_interval, tau, E)
            [causal_effect[j,i], time_delay[j,i], est_series[:,i,j]] = lag_causal(G[:,j], G[:,i], lag_interval, tau, E)
    return causal_effect,time_delay,est_series

def lag_causal(X, Y, lag_interval, tau, E):
    num = lag_interval[1]-lag_interval[0]+1
    seri = len(X) - lag_interval[1] + lag_interval[0] - (E-1)

    series = np.empty((seri, num))
    r = np.zeros((num))
    for i in  range(lag_interval[0],lag_interval[1]+1,1): # max_lag
        [x,y] = lag_series(X,Y,i,lag_interval)
        k = i+(-lag_interval[0])
        [series[:,k], r[k]] = ccm(y, x, tau, E); # 用y估计x，得到X->Y的因果作用

    causal_effect = np.max(np.abs(r))
    pos = np.argmax(np.abs(r))

    ti_de = pos - (-lag_interval[0])
    est_series = series[:,pos]

    return causal_effect, ti_de, est_series

def lag_series(X, Y, lag, lag_interval):
    '''
    Series represent x(t) and y(t-lag).
    If lag = -1, series represent x(t) and y(t+1). If lag = 3, series represent x(t) and y(t-3).
    '''
    L = len(X)
    x = X[lag_interval[1]:L-(-lag_interval[0])]
    y = Y[lag_interval[1]-lag:L-(-lag_interval[0])-lag]
    return x,y

def ccm(x, y, tau, E):
    '''use x to estimate y and get causal relationship from y to x'''
    L = len(x)
    L_lagged = L - (E-1) * tau
    lagged_x = get_lagged_vec(x, tau, E, L_lagged)
    M_x = get_distance_matrix(lagged_x, L_lagged)
    x_index = get_nearest_index(M_x, L_lagged, E)
    nearest_lagged_vec = get_nearest_vec(lagged_x, x_index, L_lagged, E)
    weight = get_weight(lagged_x, nearest_lagged_vec, L_lagged, E)
    [y_esti, Correlation_coefficient] = estimate(y, x_index, weight, L_lagged, E, tau)
    return y_esti, Correlation_coefficient

def get_lagged_vec(x, tau, E, L_lagged):
    '''Get shadow manifold reconstructed using time lags of x'''
    lagged_vec = np.zeros((L_lagged,E))
    for i in range(L_lagged):
        lagged_vec[i,:] = x[np.arange(i+(E-1)*tau, i-1, -tau)]
    return lagged_vec

def get_distance_matrix(lagged_x, L_lagged):
    '''Calculate distence between each point in shadow manifold'''
    M = np.zeros((L_lagged,L_lagged))
    for i in range(L_lagged):
        for j in range(i+1,L_lagged):
            M[i,j] = np.linalg.norm(lagged_x[i,:] - lagged_x[j,:],ord=2)
    M = M + np.transpose(M)
    return M

def get_nearest_index(M, L_lagged, E):
    '''Get time index of nearest neighbors'''
    nearest_index = np.zeros((L_lagged,E+1))
    for i in range(L_lagged):
        nearest_index[i,:] = np.argsort(M[i,:])[np.arange(1,E+2)]
    return nearest_index.astype(int)

def get_nearest_vec(lagged_x, nearest_index, L_lagged, E):
    '''Get nearest neighbors'''
    nearest_vec = np.zeros((E+1, E, L_lagged))
    for i in range(L_lagged):
        nearest_vec[:,:,i] = lagged_x[nearest_index[i,:],:];
    return nearest_vec

def get_weight(lagged_x, nearest_lagged_vec, L_lagged, E):
    '''Calculate weights'''
    weight = np.zeros((L_lagged, E+1))
    for i in range(L_lagged):
        denominator=0; num=1
        while denominator == 0:
            denominator = np.linalg.norm(lagged_x[i,:] - nearest_lagged_vec[num,:,i], ord=2)
            num=num+1
            if num==E+1:
                denominator=1
        for j in range(E+1):
            weight[i,j] = np.exp(-np.linalg.norm(lagged_x[i,:]- nearest_lagged_vec[j,:,i], ord=2) / denominator)
        weight[i,:] = weight[i,:] / np.sum(weight[i,:])
    return weight

def estimate(y, nearest_index, weight, L_lagged, E, tau):
    '''Estimate X by Y.'''
    y_esti = np.zeros((L_lagged))
    for i in range(L_lagged):
        y_esti[i] = weight[i,:] @ y[E-1+nearest_index[i,:]]

    r = stats.pearsonr(y[(E-1)*tau:].reshape(-1), y_esti.reshape(-1))
    return [y_esti,r]

def partial_corr_matrix(X):
    '''
    Calculate partial correlation matrix
    
    Parameters
    ----------
    X: series matrix with processes n * length L

    Return
    ----------
    r: partial correlation matrix with n*n size

    '''
    n = len(X)
    r = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            r[i,j] = partial_corr(X[i],X[j],np.vstack((X[0:i],X[i+1:j],X[j+1:])))
    return r

def partial_corr(X,Y,Z):
    '''
    Calculate partial correlation between X and Y condition on Z
    
    Parameters
    ----------
    X, Y, Z: series with n*1

    Return
    ----------
    r: partial correlation

    '''
    x = np.vstack((X,Y,Z))
    if len(X) >= 2:
        r = par_co(x)
    else:
        print('Parameter Error！')
    return r

def par_co(x):
    if len(x) == 2:
        r = stats.pearsonr(x[0],x[1])[0]
    elif len(x) > 2:
        r1 = par_co(x[:-1])
        r2 = par_co(np.vstack((x[0],x[-1],x[2:-1])))
        r3 = par_co(np.vstack((x[1],x[-1],x[2:-1])))
        r = (r1-r2*r3)/(np.sqrt(1-r2**2)*np.sqrt(1-r3**2))
    return r

def select_var(matrix):
    min_column_numbers = [i for i, value in enumerate(matrix[1]) if value == min(matrix[1])]
    selection = matrix[0,min_column_numbers[np.argmax(matrix[2,min_column_numbers])]]
    return selection

def sericapture(data,maxdelay,E,tau=1):
    '''
    Series Capture
    Due to delay vectors and maxdelay, 
    the series estimated by CCM will be shorter than the original series.
    
    Parameters
    ----------
    data: original series
    maxdelay, E, tau

    Return
    ----------
    cdata: captured data

    '''
    delay_interval = [-maxdelay,0]
    L = np.size(data,0)
    k = np.size(data,1)
    lists = range(delay_interval[1]+(E-1)*tau,L-(-delay_interval[0]))
    cL = len(lists)
    if len(data.shape)==2:
        cdata = data[lists,:]
    elif len(data.shape)==3:
        m = np.size(data,2)
        cdata = np.empty((cL,k,m))
        for i in range(m):
            cdata[:,:,i] = data[lists,:,i]
    return cdata

def eva_cri(P,R):
    '''
    Calculate several evaluation criteria
    
    Parameters
    ----------
    P: Object matrix
    R: Groundtruth
    
    Return
    ----------
    AUC, tpr, fpr, acc(accuracy)
    
    '''
    TH = np.flip(np.arange(0,1.01,0.01))
    
    tpr = np.zeros((len(TH),1))
    fpr = np.zeros((len(TH),1))
    acc = np.zeros((len(TH),1))
    
    for t in range(len(TH)):
        [tpr[t], fpr[t], acc[t]] = eva(np.abs(P), R, TH[t]);
    
    AUC = get_AUC(tpr,fpr)
    
    return AUC, tpr, fpr, acc

def eva(P, R, th):
    '''
    Calculate several evaluation criteria under the given threshold
    
    Parameters
    ----------
    P: Object matrix
    th: threshold
    R: Groundtruth
    
    Return
    ----------
    tpr, fpr, acc, (preci, f1score)
    
    '''
    n = np.size(P,0); M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j!=i and np.abs(P[i,j]) >= th: M[i,j] = 1
            
    [row, col] = np.where(M == 1)
    t = len(row);
    
    Po = 0 # TP + FN
    for i in range(n):
        for j in range(n):
            if j!=i and np.abs(R[i,j]) > 0: Po += 1
    Ne = n*(n-1) - Po # FP + TN
    
    if t == 0:
        tp = 0; fp = 0
    else:
        tp = 0; fp = 0;
        for i in range(t):
            if R[row[i],col[i]] == 0: fp = fp + 1
            else: tp = tp + 1
    
    tn = Ne - fp
    
    tpr = tp/Po; fpr = fp/Ne
    acc = (tp+tn)/(n*(n-1))
    # preci = tp/(tp+fp)
    # f1score = (2*preci*tpr)/(preci+tpr)
    
    return tpr, fpr, acc

def get_AUC(tpr,fpr):
    '''
    Calculate AUC
    
    Parameters
    ----------
    tpr, fpr
    
    Return
    ----------
    AUC
    
    '''
    a,s= np.unique(fpr, return_index=True)
    
    width_screening = np.diff(a)
    height_screening = tpr[s]
    AUC = width_screening @ height_screening[:-1]
    
    return AUC
