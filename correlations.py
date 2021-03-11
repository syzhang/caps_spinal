"""
calculate correlation given time series
"""

import os
import sys
import numpy as np
import pandas as pd
from DCC_GARCH.GARCH import GARCH, garch_loss_gen
from DCC_GARCH.DCC import DCC, dcc_loss_gen, R_gen

def dynamic_corr_parallel(func_file, max_itr=2, flatten=False):
    """calculate garch-dcc"""
    # load txt file
    sj_mat = pd.read_csv(func_file, sep='\t', header=None)
    epsilon_mat = np.empty(sj_mat.shape)
    # calculate epsilon for each roi ts
    for i, col in sj_mat.T.iterrows():
        ts_epsilon = garch_epsilon(col)
        epsilon_mat[:,i] = ts_epsilon
    # calculate dcc output
    dcc_sj = dcc_calc(epsilon_mat, max_itr=max_itr, flatten=flatten)
    # save correlation
    tmp = func_file.split('/')
    path_name = tmp[:-1]
    file_name = tmp[-1].split('.')[0]
    save_dir = os.path.join('../output', 'dynamic_corr')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = file_name+'_dcc.npy'
    output_file = os.path.join(save_dir, file_name)
    np.save(output_file, dcc_sj)
    print(f'dcc output saved to {output_file}')
    return dcc_sj

def dcc_calc(epsilon_mat, max_itr=10, flatten=False):
    """estimate DCC model given epsilon matrix of shape (t, n)"""
    epsilon = epsilon_mat.T # (n,t)
    dcc_model = DCC(max_itr=max_itr)
    dcc_model.set_loss(dcc_loss_gen())
    dcc_model.fit(epsilon)
    # get DCC R (conditional correlation matrix)
    ab = dcc_model.get_ab()
    tr = epsilon
    R_ls = R_gen(tr,ab)
    R = np.array(R_ls)
    # flatten Rt or not (flatten will prep data for classification)
    if flatten:
        K = R.shape[1]
        Rt_triu = R[:,np.triu(np.ones((K,K)),1)>0].T
        return Rt_triu
    else:
        return R

def garch_epsilon(t1):
    """calculate GARCH(1,1) epsilon of given ts vector (t,)"""
    t1_model = GARCH(1,1)
    t1_model.set_loss(garch_loss_gen(1,1))
    t1_model.set_max_itr(1)
    t1_model.fit(t1)
    t1_sigma = t1_model.sigma(t1)
    t1_epsilon = t1/t1_sigma
    return t1_epsilon


# running
if __name__=="__main__":
    # dynamic correlation
    func_file = sys.argv[1]
    dynamic_corr_parallel(func_file, max_itr=2, flatten=False)