import numpy as np
from numpy import array, abs, max, hstack, vstack, ones, zeros, cov, mat, where
from numpy.random import uniform, normal as rnorm
from numpy.linalg import det

import copy
import dcor
from math import gamma, log, pi

from scipy import stats
from scipy.stats import rankdata as rank
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg

from sklearn import gaussian_process


def InVaXMap(G, maxdelay, E, tau, th='mean', test_index='parcorr'):
    '''
    InVaXMap method

    Parameters
    ----------
    G:          time series matrix with length L * processes n
    max_lag:    max time lag
    tau:        time lag, usually setting tau=1 
    E:          embedding dimension
    th:         offspring threshold
    test_index  index for causal discovery
    
    Return
    ----------
    initial:        initial causal matrix
    time_delay:     time delay matrix
    fc:             Causal matrix with causal_effect[i,j] meaning causal effect from i to j
    descendant_set  descendants of each variable
    '''

    L = np.size(G,0); n = np.size(G,1)
    lag_interval = [-maxdelay,0]
    ori_data = G[range(lag_interval[1]+(E-1),L-(-lag_interval[0])),:]
    initial, time_delay, esti_data = ccm_esti(G, maxdelay, tau, E)

    # pcm = np.zeros((n,n))
    # for i in range(n):
    #     for j in range(n):
    #         if j != i:
    #             condition = np.delete(np.linspace(0, n-1, n).astype(int), (i,j))
    #             pcm[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,condition,i]))

    fc = np.zeros((n,n))
    fc2 = np.zeros((n,n))
    descendant_set = []
    for i in range(n):
        if th == 'mean':
            th = np.mean(initial[i,:])
        sub = np.linspace(0,n-1,n).astype(int); sub = np.delete(sub,i)
        offspring = []
        k = 1
        while sub.size != 0 or k >= n:
            matrix = np.vstack((sub, -time_delay[i,sub], initial[i,sub]))
            if matrix.size == 0: break
            tar = int(select_var(matrix))
            if initial[i,tar] > th:
                    if offspring == []:
                        offspring.append(tar)
                    else:
                        r = partial_corr(ori_data[:,i], esti_data[:,tar,i], np.transpose(esti_data[:,offspring,i]))
                        if np.abs(r) > th:
                            offspring.append(tar)
            sub = np.delete(sub, sub==tar)
            k += 1
        if len(offspring) == 0:
            fc[i,:] = initial[i,:]
        else:
            for j in range(n):
                if j != i:
                    if j in offspring:
                        oc = copy.deepcopy(offspring)
                        oc.remove(j)
                    else:
                        oc = copy.deepcopy(offspring)

                    if test_index == 'parcorr':
                        fc[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,oc,i]))\
                        
                    elif test_index == 'gpdc':
                        if len(oc) == 0:
                            fc[i,j] = GPDC(X=ori_data[:,i], Y=esti_data[:,j,i], Z=[])
                        else:
                            fc[i,j] = GPDC(X=ori_data[:,i], Y=esti_data[:,j,i], Z=np.transpose(esti_data[:,oc,i]))

                    elif test_index == 'cmiknn':
                        fc[i,j] = CMIknn(X=ori_data[:,i].reshape((-1,1)), Y=esti_data[:,j,i].reshape((-1,1)), Z=esti_data[:,oc,i], k=20)

                    elif test_index == 'parcorr_and_gpdc':
                        fc[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,oc,i]))
                        if len(oc) == 0:
                            fc2[i,j] = GPDC(X=ori_data[:,i], Y=esti_data[:,j,i], Z=[])
                        else:
                            fc2[i,j] = GPDC(X=ori_data[:,i], Y=esti_data[:,j,i], Z=np.transpose(esti_data[:,oc,i]))
                            
        descendant_set.append(offspring)
    if test_index == 'parcorr_and_gpdc':
        return abs(initial), abs(fc), abs(fc2)
    else:
        return abs(initial), time_delay, abs(fc), descendant_set

def PCM(G, maxdelay, E, tau):
    '''
    Partial Cross Mapping method

    Parameters
    ----------
    G:          time series matrix with length L * processes n
    max_lag:    max time lag
    tau:        time lag, usually setting tau=1 
    E:          embedding dimension
    
    Return
    ----------
    pcm:             Causal matrix with causal_effect[i,j] meaning causal effect from i to j
    '''

    L = np.size(G,0); n = np.size(G,1)
    lag_interval = [-maxdelay,0]
    ori_data = G[range(lag_interval[1]+(E-1),L-(-lag_interval[0])),:]
    _, _, esti_data = ccm_esti(G, maxdelay, tau, E)
    pcm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j != i:
                condition = np.delete(np.linspace(0, n-1, n).astype(int), (i,j))
                pcm[i,j] = partial_corr(ori_data[:,i], esti_data[:,j,i], np.transpose(esti_data[:,condition,i]))
    return abs(pcm)

def InVaXMap_first_order(G, maxdelay, E, tau=1, th='mean'):
    n = np.size(G,1)
    lag_interval = [-maxdelay, 0]
    
    oridata = sericapture(data=G, maxdelay=maxdelay, E=E, tau=tau)
    coridata = sericapture(data=oridata, maxdelay=maxdelay, E=E, tau=tau)
    
    ic, td, est1 = ccm_esti(G, maxdelay, tau, E)
    cest1 = sericapture(est1, maxdelay, E, tau)
    
    fc = np.zeros((n,n))
    for i in range(n):
        if th == 'mean':
            th = np.mean(ic[i,:])
        sub = np.linspace(0,n-1,n).astype(int); sub = np.delete(sub,i)

        offspring = []
        k = 1
        while sub.size != 0 or k >= n:
            matrix = np.vstack((sub, -td[i,sub], ic[i,sub]))
            if matrix.size == 0: break
            tar = int(select_var(matrix))
            if ic[i,tar] > th:
                if offspring == []:
                    offspring.append(tar)
                else:
                    s = len(offspring)
                    est2 = np.empty((s,np.size(coridata,0)))
                    for k in range(s):
                        est2[k,:] = lag_causal(oridata[:,i], est1[:,tar,offspring[k]], lag_interval, tau, E)[2]
                    r = partial_corr(coridata[:,i], cest1[:,tar,i], est2)
                    if np.abs(r) > th:
                        offspring.append(tar)       
            sub = np.delete(sub, sub==tar)
            k += 1
        if len(offspring) == 0:
            fc[i,:] = ic[i,:]
        else:
            for j in range(n):
                if j != i:
                    if j in offspring:
                        oc = copy.deepcopy(offspring)
                        oc.remove(j)
                    else:
                        oc = copy.deepcopy(offspring)

                    s = len(oc)
                    est3 = np.empty((s,np.size(coridata,0)))
                    for k in range(s):
                        est3[k,:] = lag_causal(oridata[:,i], est1[:,j,oc[k]], lag_interval, tau, E)[2]
                    fc[i,j] = partial_corr(coridata[:,i], cest1[:,j,i], est3)
    return abs(ic), td, abs(fc)

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
            causal_effect[i,j], time_delay[i,j], est_series[:,j,i] = lag_causal(G[:,i], G[:,j], lag_interval, tau, E)
            causal_effect[j,i], time_delay[j,i], est_series[:,i,j] = lag_causal(G[:,j], G[:,i], lag_interval, tau, E)
    return causal_effect, time_delay, est_series

def lag_causal(X, Y, lag_interval, tau, E):
    num = lag_interval[1]-lag_interval[0]+1
    seri = len(X) - lag_interval[1] + lag_interval[0] - (E-1)

    series = np.empty((seri, num))
    r = np.zeros((num))
    for i in  range(lag_interval[0],lag_interval[1]+1,1): # max_lag
        x, y = lag_series(X,Y,i,lag_interval)
        k = i+(-lag_interval[0])
        series[:,k], r[k] = ccm(y, x, tau, E); # Estimate x by y, which is the causality from x to y (X->Y).

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
    return x, y

def ccm(x, y, tau, E):
    '''use x to estimate y and get causal relationship from y to x'''
    L = len(x)
    L_lagged = L - (E-1) * tau
    lagged_x = get_lagged_vec(x, tau, E, L_lagged)
    M_x = get_distance_matrix(lagged_x, L_lagged)
    x_index = get_nearest_index(M_x, L_lagged, E)
    nearest_lagged_vec = get_nearest_vec(lagged_x, x_index, L_lagged, E)
    weight = get_weight(lagged_x, nearest_lagged_vec, L_lagged, E)
    y_esti, corr = estimate(y, x_index, weight, L_lagged, E, tau)
    return y_esti, corr

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

    r, _ = stats.pearsonr(y[(E-1)*tau:].reshape(-1), y_esti.reshape(-1))
    return y_esti, r

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
            r[i,j] = partial_corr(X[i], X[j], np.vstack((X[0:i],X[i+1:j],X[j+1:])))
    return r

def partial_corr(X, Y, Z):
    '''
    Calculate partial correlation between X and Y condition on Z
    
    Parameters
    ----------
    X, Y: series with 1 * length L
    Z:  series with processes n * length L

    Return
    ----------
    r: partial correlation
    '''

    r = partial_corr_calculation(np.vstack((X,Y,Z)))

    return r

def partial_corr_calculation(x):
    if len(x) == 2:
        r = stats.pearsonr(x[0], x[1])[0]
    elif len(x) > 2:
        r1 = partial_corr_calculation(x[:-1])
        r2 = partial_corr_calculation(np.vstack((x[0], x[-1], x[2:-1])))
        r3 = partial_corr_calculation(np.vstack((x[1], x[-1], x[2:-1])))
        r = (r1-r2*r3) / (np.sqrt(1-r2**2)*np.sqrt(1-r3**2))
    return r

def GPDC(X, Y, Z=[], standardize=True):
    """GPDC method

    Args:
        X (_type_):            target variable with the size series length (L)
        Y (_type_):            target variable with the size series length (L)
        Z (_type_):            condition variable with the size dim (N) * series length (L)
        standardize (bool, optional):   standardize the input data. Defaults to True.

    Returns:
        _type_: GPDC index between X condition on Z
    """

    if Z == []:
        residus1 = GaussianProcess(X=X, Z=Y, standardize=standardize)
        residus2 = GaussianProcess(X=Y, Z=X, standardize=standardize)
    else:
        residus1 = GaussianProcess(X=X.reshape(1,-1), Z=Z, standardize=standardize)
        residus2 = GaussianProcess(X=Y.reshape(1,-1), Z=Z, standardize=standardize)

    hat_x, hat_y = uniformized(np.vstack([residus1,residus2]))

    val = dcor.distance_correlation(hat_x, hat_y, method='AVL')
    
    return val

def GaussianProcess(X, Z, standardize=True):
    """Gaussian Process

    Args:
        X (_type_):         target variable with the size dim (1) * series length (L)
        Z (_type_):         condition variable with the size dim (N) * series length (L)
        standardize (bool, optional):   standardize the input data. Defaults to True.

    Returns:
        _type_: residus of Gaussian Process Regression
    """

    array = np.vstack((X.reshape(1,-1), Z))
    N = array.shape[0]

    if standardize:
        array -= array.mean(axis=1).reshape(N, 1)
        std = array.std(axis=1)
        for i in range(N):
            if std[i] != 0.:
                array[i] /= std[i]

    target_series = array[0].reshape(-1, 1)
    condition_series = array[1:].T
    if np.ndim(condition_series) == 1:
        condition_series = condition_series.reshape(1, -1)

    kernel = gaussian_process.kernels.RBF() + gaussian_process.kernels.WhiteKernel()
    alpha = 0

    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    gp.fit(condition_series, target_series)
    resid = target_series.T - gp.predict(condition_series)

    return resid

def uniformized(x):
    """uniformized residuals

    Args:
        x (_type_): residuals with the size dim (2) * series length (L)

    Returns:
        _type_: uniformized residuals with the size dim (2) * series length (L)
    """

    if np.ndim(x) == 1:
        u = unifo(x)
    else:
        u = np.empty(x.shape)
        for i in range(x.shape[0]):
            u[i] = unifo(x[i])
    return u

def unifo(x):
    x_sorted = np.sort(x)
    y = np.linspace(1. / len(x), 1, len(x))
    return np.interp(x, x_sorted, y)

def CMIknn(X, Y, Z=[], k=20):
    """calculate conditional mutual information (CMI) with knn method

    Args:
        X (_type_):             target variable with the size series length (L) * 1
        Y (_type_):             target variable with the size series length (L) * 1
        Z (list, optional):     condition variable with the size series length (L) * dim (N). Defaults to [].
        k (int, optional):      number of nearest neighbors. Defaults to 20.

    Returns:
        _type_: _description_
    """

    Np = len(X)
    if len(Z) == 0:
        nxpNN = np.zeros(Np)
        nypNN = np.zeros(Np)
        for i in range(Np):
            distance, _ = NNSearch(points=np.hstack((X, Y)), target=np.hstack((X[i], Y[i])), k=k)
            halfepsilon = distance[-1]

            nxpNN[i] = np.sum(cdist(X, X[i].reshape(1,-1), metric='chebychev') < halfepsilon)
            nypNN[i] = np.sum(cdist(Y, Y[i].reshape(1,-1), metric='chebychev') < halfepsilon)

        idN = (nxpNN != 0) & (nypNN != 0)
        val = abs(digamma(k) - np.mean(digamma(nxpNN[idN]+1)) - np.mean(digamma(nypNN[idN]+1)) + digamma(Np))

    else:
        nzpNN = np.zeros(Np)
        nxzpNN = np.zeros(Np)
        nyzpNN = np.zeros(Np)
        for i in range(Np):
            distance, _ = NNSearch(points=np.hstack((X, Y, Z)), target=np.hstack((X[i], Y[i], Z[i])), k=k)
            halfepsilon = distance[-1]

            nzpNN[i] = np.sum(cdist(Z, Z[i].reshape(1,-1), metric='chebychev') < halfepsilon)
            nxzpNN[i] = np.sum(cdist(np.hstack((X, Z)), np.hstack((X[i], Z[i])).reshape(1,-1), metric='chebychev') < halfepsilon)
            nyzpNN[i] = np.sum(cdist(np.hstack((Y, Z)), np.hstack((Y[i], Z[i])).reshape(1,-1), metric='chebychev') < halfepsilon)

        idN = (nzpNN != 0) & (nxzpNN != 0) & (nyzpNN != 0)
        val = abs(digamma(k) + np.mean(digamma(nzpNN[idN]+1)) - np.mean(digamma(nxzpNN[idN]+1)) - np.mean(digamma(nyzpNN[idN]+1)))

    return val

def NNSearch(points, target, k):
    """get k nearest neighbors of target from points

    Args:
        points (_type_):    search space with the size length (L) * dim (N)
        target (_type_):    target point with the size 1 * dim (N)
        k (_type_):         number of nearest neighbors

    Returns:
        index (_type_):     
    """

    tree = cKDTree(points)
    distance, index = tree.query(target, k+1, p=np.inf)
    if distance[0] != 0:
        return distance[0:k], index[0:k]
    else:
        return distance[1:k+1], index[1:k+1]

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

def GrangerCausality(data, lag, method_type):
	"""
	Granger Causality

	Args:
		data: 			time series matrix with length maxL * processes N.
		lag:    		time lag used to vector autoregression.
		method_type: 	method type, which is 'original' (Granger Causality), 'conditional' (conditional Granger Causality) or 'partial'(partial Granger Causality).

	Returns:
		Causal matrix with gc[i,j] meaning causal effect from i to j
	"""

	maxL,N = np.shape(data)
	gc = np.zeros((N,N))
	if method_type == 'original': # original granger causality
		# Autoregressive model
		ar_error = np.zeros((maxL-lag,N))
		for i in range(N):
			ar_model = AutoReg(data[:,i], lags=lag)
			ar_results = ar_model.fit()
			ar_error[:,i] = ar_results.resid
		# vector autoregressive model
		var_error = np.zeros((maxL-lag,N,N))
		for i in range(N):
			for j in range(i+1,N):
				var_model = VAR(data[:,[i,j]])
				var_results = var_model.fit(maxlags=lag)
				var_error[:,j,i] = var_results.resid[:,0] # the error of regression to i using i and j
				var_error[:,i,j] = var_results.resid[:,1]
				gc[i,j] = np.log(np.var(ar_error[:,j])/np.var(var_error[:,j,i]))
				gc[j,i] = np.log(np.var(ar_error[:,i])/np.var(var_error[:,i,j]))
	else:
		# vector autoregressive model
		var_model = VAR(data)
		var_results = var_model.fit(maxlags=lag)
		var_error = var_results.resid

		var_sub_error = np.zeros((maxL-lag,N,N))
		for i in range(N):
			subdata = data[:, np.concatenate((np.arange(i), np.arange(i+1, N)))]
			var_sub_model = VAR(subdata)
			var_sub_results = var_sub_model.fit(maxlags=lag)
			var_sub_error[:,0:i,i] = var_sub_results.resid[:,0:i]
			var_sub_error[:,i+1:,i] = var_sub_results.resid[:,i:]
		
		# Calculate the result variable of variable i
		if method_type == 'conditional':
			for i in range(N):
				for j in range(N):
					if i == j:
						continue
					gc[i,j] = np.log(np.var(var_sub_error[:,j,i])/np.var(var_error[:,j]))
		elif method_type == 'partial':
			for i in range(N):
				for j in range(N):
					if i == j:
						continue
					e1 = var_sub_error[:,j,i][:, np.newaxis]
					e2 = np.delete(var_sub_error[:, :, i], [i, j], axis=1)
					e3 = var_error[:,j][:, np.newaxis]
					e4 = np.delete(var_error, [i, j], axis=1)
					var1 = np.cov(np.hstack((e1,e2)),rowvar=False)
					var2 = np.cov(np.hstack((e3,e4)),rowvar=False)
					s1 = var1[0,0] - var1[0,1:] @ np.linalg.inv(var1[1:,1:]) @ var1[1:,0]
					s2 = var2[0,0] - var2[0,1:] @ np.linalg.inv(var2[1:,1:]) @ var2[1:,0]
					gc[i,j] = np.log(s1/s2)
	return gc


def TransferEntropy(data, lag=5):
	"""
	Transfer Entropy

	Args:
		data (_type_): _description_
		lag (int, optional): _description_. Defaults to 5.

	Returns:
		_type_: _description_
	"""

	maxL,N = np.shape(data)
	te = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			if i == j:
				continue
			te[i,j] = transent(x=data[:,j], y=data[:,i], lag=lag, k=3) # causality from y to x
	return te

##################################################################################
###  Estimating Copula Entropy and Transfer Entropy 
###  2024-02-12
###  by Ma Jian (Email: majian03@gmail.com)
###
###  Parameters
###	x    	: N * d data, N samples, d dimensions
###	k    	: kth nearest neighbour, parameter for kNN entropy estimation. default = 3
###	dtype	: distance type ['euclidean', 'chebychev' (i.e Maximum distance)]
###	lag	: time lag. default = 1
###	s0,s1	: two samples with same dimension
###	n	: repeat time of estimation. default = 12
###	thd	: threshold for the statistic of two-sample test
###	maxp	: maximal number of change points
###	minseglen : minimal length of binary segmentation
###
###  References
###  [1] Ma Jian, Sun Zengqi. Mutual information is copula entropy. 
###      arXiv:0808.0845, 2008.
###  [2] Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. 
###      Physical review E, 2004, 69(6): 066138.
###  [3] Ma, Jian. Estimating Transfer Entropy via Copula Entropy. 
###      arXiv preprint arXiv:1910.04375, 2019.
###  [4] Ma, Jian. Multivariate Normality Test with Copula Entropy.
###      arXiv preprint arXiv:2206.05956, 2022.
###  [5] Ma, Jian. Two-Sample Test with Copula Entropy.
###      arXiv preprint arXiv:2307.07247, 2023.
###  [6] Ma, Jian. Change Point Detection with Copula Entropy based Two-Sample Test
###      DOI:10.13140/RG.2.2.16378.26562, 2024.
##################################################################################

##### constructing empirical copula density [1]
def construct_empirical_copula(x):
	(N,d) = x.shape	
	xc = zeros([N,d]) 
	for i in range(0,d):
		xc[:,i] = rank(x[:,i]) / N
	
	return xc

##### Estimating entropy with kNN method [2]
def entknn(x, k = 3, dtype = 'chebychev'):
	(N,d) = x.shape
	
	g1 = digamma(N) - digamma(k)
	
	if dtype == 'euclidean':
		cd = pi**(d/2) / 2**d / gamma(1+d/2)
	else:	# (chebychev) maximum distance
		cd = 1

	logd = 0
	dists = cdist(x, x, dtype)
	dists.sort()
	for i in range(0,N):
		logd = logd + log( 2 * dists[i,k] ) * d / N

	return (g1 + log(cd) + logd)

##### 2-step Nonparametric estimation of copula entropy [1]
def copent(x, k = 3, dtype = 'chebychev', log0 = False):
	xarray = array(x)

	if log0:
		(N,d) = xarray.shape
		max1 = max(abs(xarray), axis = 0)
		for i in range(0,d):
			if max1[i] == 0:
				xarray[:,i] = rnorm(0,1,N)
			else:
				xarray[:,i] = xarray[:,i] + rnorm(0,1,N) * max1[i] * 0.000005

	xc = construct_empirical_copula(xarray)

	try:
		return -entknn(xc, k, dtype)
	except ValueError: # log0 error
		return copent(x, k, dtype, log0 = True)


##### conditional independence test [3]
##### to test independence of (x,y) conditioned on z
def ci(x, y, z, k = 3, dtype = 'chebychev'):
	xyz = vstack((x,y,z)).T
	yz = vstack((y,z)).T
	xz = vstack((x,z)).T
	return copent(xyz,k,dtype) - copent(yz,k,dtype) - copent(xz,k,dtype)

##### estimating transfer entropy from y to x with lag [3]
def transent(x, y, lag = 1, k = 3, dtype = 'chebychev'):
	xlen = len(x)
	ylen = len(y)
	if (xlen > ylen):
		l = ylen
	else:
		l = xlen
	if (l < (lag + k + 1)):
		return 0
	x1 = x[0:(l-lag)]
	x2 = x[lag:l]
	y = y[0:(l-lag)]
	return ci(x2,y,x1,k,dtype)

##### multivariate normality test [4]
def mvnt(x, k = 3, dtype = 'chebychev'):
	return -0.5 * log(det(cov(x.T))) - copent(x,k,dtype)

##### two-sample test [5]
def tst(s0,s1,n=12, k = 3, dtype = 'chebychev'):
	(N0,d0) = s0.shape
	(N1,d1) = s1.shape
	x = vstack((s0,s1))
	stat1 = 0
	for i in range(0,n):
		y1 = vstack((ones([N0,1]),ones([N1,1])*2)) + uniform(0, 0.0000001,[N0+N1,1])
		y0 = ones([N0+N1,1]) + uniform(0,0.0000001,[N0+N1,1])
		stat1 = stat1 + copent(hstack((x,y1)),k,dtype) - copent(hstack((x,y0)),k,dtype)
	return stat1/n

##### single change point detection [6]
def cpd(x, thd = 0.13, n = 30):
	x = mat(x)
	len1 = x.shape[0]
	if len1 == 1:
		len1 = x.shape[1]
		x = x.T
	pos = -1
	maxstat = 0
	stat1 = zeros(len1)
	for i in range(1,len1-1):
		s0 = x[0:i,:]
		s1 = x[(i+1):,:]
		stat1[i] = tst(s0,s1,n)
	if(max(stat1) > thd):
		maxstat = max(stat1)
		pos = where(stat1 == maxstat)[0][0]+1
	return pos, maxstat, stat1

##### multiple change point detection [6]
def mcpd(x, maxp = 5, thd = 0.13, minseglen = 10, n = 30):
	x = mat(x)
	len1 = x.shape[0]
	if len1 == 1:
		len1 = x.shape[1]
		x = x.T
	maxstat = []
	pos = []
	bisegs = mat([0,len1-1])
	for i in range(0,maxp):
		if i >= bisegs.shape[0]:
			break
		rpos, rmaxstat, _ = cpd(x[bisegs[i,0]:bisegs[i,1],:],thd,n)
		if rpos > -1 :
			rpos = rpos + bisegs[i,0]
			maxstat.append(rmaxstat)
			pos.append(rpos)
			if (rpos - bisegs[i,0]) > minseglen :
				bisegs = vstack((bisegs,[bisegs[i,0],rpos-1]))
			if (bisegs[i,1] - rpos +1) > minseglen :
				bisegs = vstack((bisegs,[rpos,bisegs[i,1]]))
	return pos,maxstat
