"""
visualising dcc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

def plot_dcc(dcc_file, save_name, time_bins=5, save_plot=True):
    """plot connectivity matrix from DCC file"""
    # load dcc file
    if type(dcc_file) is str:
        dcc_mat = np.load(dcc_file)
    else: # when plotting array directly
        dcc_mat = dcc_file
    # reduce dcc matrix
    if (time_bins is not None) and (time_bins > 0):
        dcc_proc = bin_mat(dcc_mat, time_bins=time_bins)
    else:
        dcc_proc = dcc_mat
    # plot matrix
    plot_mat(dcc_proc, save_name=save_name, save_plot=save_plot)

def plot_mat(dcc_proc, save_name, save_plot=True):
    """plot matrix"""
    fig_num = dcc_proc.shape[0]
    fig, axes = plt.subplots(1,fig_num, figsize=(6*fig_num, 5), facecolor='w')
    for n in range(fig_num):
        plotting.plot_matrix(dcc_proc[n,:,:], axes=axes[n], tri='lower')
        axes[n].set_title(f'Time bin={n}')
    # save plot
    if save_plot:
        save_dir = './figs'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, save_name+'.png')
        plt.savefig(save_path, bbox_inches='tight')

def bin_mat(dcc_mat, time_bins=5):
    """average into time bins to reduce number of plots"""
    bin_int = dcc_mat.shape[0]/time_bins # t/bins=points in bin
    dcc_reduced = np.empty((int(time_bins), dcc_mat.shape[1], dcc_mat.shape[2]))
    if (time_bins is not None) and (time_bins > 0):
        bc = 0
        for b in range(time_bins):
            start_idx = int(bc + b*bin_int)
            end_idx = int(bc + (b+1)*bin_int)
            tmp = np.mean(dcc_mat[start_idx:end_idx,:,:], axis=0)
            dcc_reduced[b,:,:] = tmp
            bc += 1
    else:
        raise ValueError('Time bins not specified or invalid.')
    return dcc_reduced

def mean_subjects(dcc_list, exclude=True):
    """produce mean acros all subject matrices"""
    mat_ls = [np.load(f) for f in dcc_list]
    # check matrices if they have the same size
    time_lengths = []
    for mat in mat_ls:
        time_lengths.append(mat.shape[0])
    time_lengths = np.array(time_lengths)
    subject_check = time_lengths<max(time_lengths)
    if sum(subject_check) > 0:
        mat_ls_excluded = []
        if exclude:
            for i, mat in enumerate(mat_ls):
                if mat.shape[0] < max(time_lengths):
                    print(f'Excluding {dcc_list[i]}')
                else:
                    mat_ls_excluded.append(mat)
            print(f'{len(mat_ls_excluded)} subject matrices remained after exclusion.')
        else:
            raise ValueError('Matrices size different, consider exclude subjects.')
    else: # not excluding
        mat_ls_excluded = mat_ls

    # stacking matrices and calculate mean
    mat_concat = np.stack(mat_ls_excluded)
    mean_mat = np.mean(mat_concat, axis=0)
    # print(mean_mat.shape)
    return mean_mat
    

# running
if __name__=="__main__":
    # testing a random dcc output file (possible to mean across subjects)
    dcc_dir = '../output/'
    # walk through conditions
    for condition in os.listdir(dcc_dir):
        print(f'plotting mean image of condition {condition}')
        condition_dir = os.path.join(dcc_dir, condition, 'dynamic_corr_whiten')
        f_ls = []
        for f in os.listdir(condition_dir):
            dcc_file = os.path.join(condition_dir, f)
            f_ls.append(dcc_file)
            # # uncomment below to plot individuals
            # # derive fig name
            # save_name = f.split('.')[0]
            # # plot time bins
            # plot_dcc(dcc_file, save_name=save_name, time_bins=10)
    
        # plot mean across subjects
        mean_mat = mean_subjects(f_ls)
        plot_dcc(mean_mat, save_name=condition+'_mean', time_bins=6)
