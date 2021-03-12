"""
visualising dcc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

def plot_dcc(dcc_file, plot_name, time_bins=5, save_plot=True):
    """plot connectivity matrix from DCC file"""
    # load dcc file 
    dcc_mat = np.load(dcc_file)
    # reduce dcc matrix
    if (time_bins is not None) and (time_bins > 0):
        dcc_proc = bin_mat(dcc_mat, time_bins=time_bins)
    else:
        dcc_proc = dcc_mat
    # plot matrix
    plot_mat(dcc_proc, plot_name=plot_name, save_plot=save_plot)

def plot_mat(dcc_proc, plot_name, save_plot=True):
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
        save_path = os.path.join(save_dir, plot_name+'.png')
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

# running
if __name__=="__main__":
    # testing a random dcc output file (possible to mean across subjects)
    for i in range(1,19):
        dcc_file = f'../output/dynamic_corr/s{i:02d}_creamA_dcc.npy'
        # plot 4 time bins
        plot_name = f's{i:02d}_dcc'
        plot_dcc(dcc_file, plot_name=plot_name, time_bins=12)
