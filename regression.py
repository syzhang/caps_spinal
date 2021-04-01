"""
ramp ratings regression to dcc matrices
"""

import os
import numpy as np
import pandas as pd

def load_dccmat(cond='capsaicin', exclude=True):
    """load dcc matrices (ramp)"""
    # get dcc file list
    output_dir = '../output'
    if cond == 'capsaicin':
        cond_dir = 'allsubj_creamA_wc_ramp'
    elif cond == 'placebo':
        cond_dir = 'allsubj_creamB_wc_ramp'
    dcc_dir = os.path.join(output_dir, cond_dir, 'dynamic_corr_whiten')
    dcc_list = [f for f in sorted(os.listdir(dcc_dir))]
    # exclude subjects
    if exclude:
        dcc_list_excluded = []
        for f in dcc_list:
            tmp = np.load(os.path.join(dcc_dir,f))
            if tmp.shape[0] == 900:
                dcc_list_excluded.append(f)
            # else:
            #     print(f)
    else:
        dcc_list_excluded = dcc_list
    # load subjects
    mat_ls = [np.load(os.path.join(dcc_dir
    ,f)) for f in dcc_list_excluded]
    # stack matrices
    mat_concat = np.stack(mat_ls)
    print(mat_concat.shape)
    return mat_concat

def load_ratings(cond='capsaicin', exclude=True):
    """load ramp pain ratings"""
    df = pd.read_csv('ramp_intensity.csv')
    # convert to long form
    dfl = pd.wide_to_long(df, stubnames='OGP-', i=['subject', 'cream'], j='timepoint').reset_index()
    dfl.rename(columns={'OGP-':'rating'}, inplace=True)
    # exclude subjects
    if exclude:
        mask = dfl['subject'].isin(['s01', 's03', 's08'])
        dfl_exclude = dfl[~mask]
    else:
        dfl_exclue = dfl
    n_sj = len(np.unique(dfl_exclude['subject']))
    # which condition
    if (cond == 'capsaicin') or (cond == 'placebo'):
        dfl_cond = dfl_exclude[dfl_exclude['cream']==cond]
        # reshape to matrix
        n_ratings = int(dfl_cond.shape[0]/n_sj)
        ratings = dfl_cond['rating'].to_numpy().reshape(n_sj, n_ratings)
        # ratings = ratings[~np.any(np.isnan(ratings), axis=1)]
        print(ratings.shape)
        return ratings
    else:
        raise ValueError('Condition does not exist.')

def reshape_dcc(dcc_mat):
    """reshape dcc matrix and ratings"""
    from visualise import bin_mat
    dcc_bin = []
    for i in range(dcc_mat.shape[0]):
        dcc_bin.append(bin_mat(dcc_mat[i], time_bins=6))
    dcc_reshape = np.concatenate(dcc_bin)
    print(dcc_reshape.shape)


# running
if __name__=="__main__":
    caps_ratings = load_ratings(cond='capsaicin')
    caps_dcc = load_dccmat(cond='capsaicin')
    # reshape
    reshape_dcc(caps_dcc)
    # dcc = data1['dcc'].reshape((caps_dcc.shape[1], timebin_num*sj_num))
    # rate = data1['rate'].reshape(sj_num*timebin_num)