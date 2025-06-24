import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from pathlib import Path
import sys
sys.path.append('..')
from mvpa.classify_models import isExpUnexp
from utils import Options
from mne.stats import permutation_cluster_1samp_test
import plotting.PtitPrince as pt
import os
import ipdb


def plot_by_nvoxels(data, measure='distance', tfce_pvals=None, right_part=False, n_perms=10000, fixed_ylim=True):
    """
    - data: pandas dataframe containing the data
    - tfce_pvals are provided if they have been precomputed,
        else they're computed here
    """
    fpath = Path("./fonts/HelveticaWorld-Regular.ttf")
    fontprop = FontProperties(fname=fpath)
    
    if right_part:
        assert 'hemi' in data.columns
    assert data.roi.nunique()==1
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    if tfce_pvals is None:
        _, _, tfce_pvals, _ = get_tfce_stats(avgdata.groupby(['subject','nvoxels','expected']).mean().reset_index(),
                                             measure=measure, n_perms=n_perms)
        
    # sort n. voxels and make categorical (for plotting)
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(avgdata.loc[:, 'nvoxels'], 
                                               categories=avgdata.nvoxels.unique(), ordered=True)
            
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 4, 4, 1])
    with sns.axes_style('white'):
        if right_part:
            ax0 = fig.add_subplot(gs[1:3, 1:])
        else:
            ax0 = fig.add_subplot(gs[1:3, :])
        sns.lineplot(data=avgdata.groupby(['subject', 'nvoxels', 'expected']).mean().reset_index(), 
                     x='nvoxels', y=measure,
                     hue='expected', hue_order=[True, False], #linewidth=1,
                     palette='Dark2', ci=68, marker='o', mec='none', markersize=10) #plot_kws=dict(edgecolor="none")) #markersize=10
        if measure == 'distance':
            ylabel_left = 'Classifier Information (a.u.)'
            ylimits = (0.0, 0.45)
            yticks = list(np.arange(0., 0.45, 0.1))
            marker_bottom = 0.02
            marker_top = 0.04
        elif measure == 'correct':
            ylabel_left = 'Decoding Accuracy (a.u.)'
            ylimits = (0.5, 0.75)
            yticks = list(np.arange(0.5, 0.75, 0.1))
            marker_bottom = 0.52
            marker_top = 0.54
        plt.yticks(font=fpath, fontsize=28, ticks=yticks)
        ax0.set(ylim=ylimits, xticks=['100', '500']+[str(x) for x in np.arange(1000, maxvoxels+1000, 1000)])
        ax0.set_xlabel('Number of Voxels', font=fpath, fontsize=32)
        ax0.set_ylabel(ylabel_left, font=fpath, fontsize=32)
        plt.xticks(font=fpath, fontsize=28)
        plt.margins(0.02)
        ax0.legend_.set_title(None)
        fontprop.set_size(28)
        ax0.legend(['Congruent', 'Incongruent'], prop=fontprop, frameon=False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_linewidth(2)
        ax0.spines['bottom'].set_linewidth(2)
        for x in np.arange(0, len(tfce_pvals)):
            if tfce_pvals[x] < 0.01:
                ax0.scatter(x, marker_bottom, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
                ax0.scatter(x, marker_top, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
            elif tfce_pvals[x] < 0.05:
                ax0.scatter(x, marker_bottom, marker=(6, 2, 0), s=180, color='k', linewidths=2.)
    if right_part:
        avgdiffs = accs_to_diffs(avgdata).groupby(['subject', 'hemi']).mean().reset_index()
        with sns.axes_style('white'):
            ax1 = fig.add_subplot(gs[:, 0])
            _, suppL, densL = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='L'], color='.8', 
                                                width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04)
            _, suppR, densR = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='R'], color='.8', 
                                                width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04)
            
            densities_left = []
            for d in avgdiffs[avgdiffs['hemi']=='L']['difference']:
                ix, _ = find_nearest(suppL[0], d)
                densities_left.append(densL[0][ix])
            densities_left = np.array(densities_left).reshape(nsubjs, 1)
            scatter_left = -0.04-np.random.uniform(size=(nsubjs,1))*densities_left*0.15
            plt.scatter(scatter_left, avgdiffs[avgdiffs['hemi']=='L']['difference'], color='black', alpha=.3)
            densities_right = []
            for d in avgdiffs[avgdiffs['hemi']=='R']['difference']:
                ix, _ = find_nearest(suppR[0], d)
                densities_right.append(densR[0][ix])
            densities_right = np.array(densities_right).reshape(nsubjs,1)
            scatter_right = 0.04+np.random.uniform(size=(nsubjs,1))*densities_right*0.15
            plt.scatter(scatter_right, avgdiffs[avgdiffs['hemi']=='R']['difference'], color='black', alpha=.3)
            
            # Get mean and 95% CI:
            meandiff = avgdiffs['difference'].mean()
            tstats = pg.ttest(avgdiffs.groupby(['subject']).mean().reset_index()['difference'], 0.0)
            ci95 = tstats['CI95%'][0]
            #ax1.axis("equal")
            #ax1.set_aspect('equal')
            for tick in ax1.get_xticks():
                #ax1.plot([tick-0.1, tick+0.1], [meandiff, meandiff],
                #            lw=4, color='k')
                ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
                ax1.plot([tick-0.01, tick+0.01], [ci95[0], ci95[0]], lw=3, color='k')
                ax1.plot([tick-0.01, tick+0.01], [ci95[1], ci95[1]], lw=3, color='k')
                #circlemarker = plt.Circle((tick, meandiff), 0.015, color='k')
                #ax1.add_patch(circlemarker)
                ax1.plot(tick,meandiff, 'o', markersize=15, color='black')
            ax1.axhline(0.0, linestyle='--', color='black', linewidth=2)
            plt.yticks(font=fpath, fontsize=32) 
            ax1.set_xlabel('Average', font=fpath, fontsize=32)
            if measure == 'distance':
                ylabel_right = 'Δ Classifier Information (a.u.)'
            elif measure == 'correct':
                ylabel_right = 'Δ Decoding Accuracy (a.u.)'
            ax1.set_ylabel(ylabel_right, font=fpath, fontsize=32)
            if fixed_ylim:
                ax1.set(ylim=(-0.7, 0.7))
            ax1.axes_style = 'white'
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_linewidth(2)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    # if saveimg:
    #     plt.savefig('results_plots/EVC_nvox_distance.pdf')
    #plt.show()
    
def plot_univar_by_nvoxels(data):
    fpath = Path("./fonts/HelveticaWorld-Regular.ttf")
    fontprop = FontProperties(fname=fpath)
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    # sort n. voxels and make categorical (for plotting)
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(avgdata.loc[:, 'nvoxels'], 
                                               categories=avgdata.nvoxels.unique(), ordered=True)
    
    fig = fig = plt.figure(figsize=(20,10))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 4, 4, 1])
    ax0 = fig.add_subplot(gs[1:3, 1:])
    with sns.axes_style('white'):
        sns.lineplot(data=avgdata.groupby(['subject', 'nvoxels', 'condition']).mean().reset_index(), 
                     x='nvoxels', y='mean_beta',
                     hue='condition', hue_order=['expected', 'unexpected'], #linewidth=1,
                     palette='Dark2', ci=68, marker='o', mec='none', markersize=10)
    plt.yticks(font=fpath, fontsize=28, ticks=list(np.arange(-6.0, 1.0, 1.0)))
    ax0.set(ylim=(-6.0, 1.0), xticks=['100', '500']+[str(x) for x in np.arange(1000, maxvoxels+1000, 1000)])
    ax0.set_xlabel('Number of Voxels', font=fpath, fontsize=32)
    ax0.set_ylabel('Mean Beta Value (a.u.)', font=fpath, fontsize=32)
    plt.xticks(font=fpath, fontsize=28)
    plt.margins(0.02)
    ax0.legend_.set_title(None)
    fontprop.set_size(28)
    ax0.legend(['Congruent', 'Incongruent'], prop=fontprop, frameon=False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.axhline(0.0, linestyle='--', color='black', linewidth=2)
    
    # average violin plot
    avgdiffs = meanbetas_to_diffs(avgdata).groupby(['subject', 'hemi']).mean().reset_index()
    with sns.axes_style('white'):
        ax1 = fig.add_subplot(gs[:, 0])
        _, suppL, densL = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='L'], color='.8', 
                                                width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04)
        _, suppR, densR = pt.half_violinplot(y='difference', data=avgdiffs[avgdiffs['hemi']=='R'], color='.8', 
                                            width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04)
        
        densities_left = []
        for d in avgdiffs[avgdiffs['hemi']=='L']['difference']:
            ix, _ = find_nearest(suppL[0], d)
            densities_left.append(densL[0][ix])
        densities_left = np.array(densities_left).reshape(nsubjs, 1)
        scatter_left = -0.04-np.random.uniform(size=(nsubjs, 1)) * densities_left * 0.15
        plt.scatter(scatter_left, avgdiffs[avgdiffs['hemi']=='L']['difference'], color='black',alpha=.3)
        densities_right = []
        for d in avgdiffs[avgdiffs['hemi']=='R']['difference']:
            ix, _ = find_nearest(suppR[0], d)
            densities_right.append(densR[0][ix])
        densities_right = np.array(densities_right).reshape(nsubjs,1)
        scatter_right = 0.04+np.random.uniform(size=(nsubjs,1))*densities_right*0.15
        plt.scatter(scatter_right, avgdiffs[avgdiffs['hemi']=='R']['difference'], color='black', alpha=.3)
        
        # Get mean and 95% CI:
        meandiff = avgdiffs['difference'].mean()
        tstats = pg.ttest(avgdiffs.groupby(['subject']).mean().reset_index()['difference'], 0.0)
        ci95 = tstats['CI95%'][0]
        
        for tick in ax1.get_xticks():
            ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[0], ci95[0]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[1], ci95[1]], lw=3, color='k')
            ax1.plot(tick, meandiff, 'o', markersize=15, color='black')
        
        ax1.axhline(0.0, linestyle='--', color='black', linewidth=2)
        plt.yticks(font=fpath, fontsize=32) 
        ax1.set_xlabel('Average', font=fpath, fontsize=32)
        ax1.set_ylabel('Δ Mean Beta (a.u.)', font=fpath, fontsize=32)
        ax1.set(ylim=(-2.5, 2.5))
        ax1.axes_style = 'white'
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
        

def get_tfce_stats(data, measure='distance', n_perms=10000):
    subxvoxels = df_to_array_tfce(data.groupby(['subject','nvoxels','expected']).mean().reset_index(),
                                        measure=measure)
    threshold_tfce = dict(start=0, step=0.01)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        subxvoxels, n_jobs=1, threshold=threshold_tfce, adjacency=None,
        n_permutations=n_perms, out_type='mask')
    return t_obs, clusters, cluster_pv, H0

def meanbetas_to_diffs(df):
    diffs = []
    for nv in df.nvoxels.unique():
        for hemi in df.hemi.unique():
            for sub in df[(df['nvoxels']==nv)&(df['hemi']==hemi)].subject.unique():
                thissub = df[(df['nvoxels']==nv)&(df['hemi']==hemi)&(df['subject']==sub)]
                thisdiff = thissub[thissub['condition']=='expected']['mean_beta'].values[0] - \
                    thissub[thissub['condition']=='unexpected']['mean_beta'].values[0]
                diffs.append({'subject': sub, 'nvoxels': nv, 'hemi': hemi, 'difference': thisdiff})
    return pd.DataFrame(diffs)

def accs_to_diffs(df, measure='distance'):
    diffs = []
    for nv in df.nvoxels.unique():
        for hemi in df.hemi.unique():
            for sub in df[(df['nvoxels']==nv)&(df['hemi']==hemi)].subject.unique():
                thissub = df[(df['nvoxels']==nv)&(df['hemi']==hemi)&(df['subject']==sub)]
                #assert(len(thissub)==2) # should only be one expected, and one unexpected value
                thisdiff = thissub[thissub['expected']==True][measure].values[0] - \
                           thissub[thissub['expected']==False][measure].values[0]
                diffs.append({'subject': sub, 'nvoxels': nv, 'hemi': hemi, 'difference': thisdiff})
    return pd.DataFrame(diffs)


    
def df_to_array_tfce(df, measure='correct'):
    """
    """
    subxvoxels = np.zeros((df.subject.nunique(), df.nvoxels.nunique()))
    for i, sub in enumerate(np.sort(df.subject.unique())):
        for j, nv in enumerate(np.sort(df.nvoxels.unique())):
            thisdata = df[(df['subject']==sub)&(df['nvoxels']==nv)]
            subxvoxels[i, j] = thisdata[thisdata['expected']==True][measure].values - \
                thisdata[thisdata['expected']==False][measure].values
    return subxvoxels



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# --------------------------------------------------------------------------------------
# Behavioral plots
# --------------------------------------------------------------------------------------

def pretty_behav_plot(avgdata, measure='Hit', excl=True, fname=None, saveimg=False):
    
    assert(measure in ['Hit', 'DPrime', 'Criterion'])
    
    fpath = Path("./fonts/HelveticaWorld-Regular.ttf")
    
    # Get all differences
    alldiffs = []
    for sub in avgdata.Subject.unique():
        thisdiff = avgdata[(avgdata.Subject==sub)&(avgdata.Consistent==1)][measure].values[0] - \
                   avgdata[(avgdata.Subject==sub)&(avgdata.Consistent==0)][measure].values[0]
        alldiffs.append(thisdiff)
    alldiffs = pd.DataFrame(alldiffs, columns=['difference'])
    
    fig = plt.figure(figsize=(10,10)) # (10, 8)
    
    ax0 = fig.add_subplot(121)
    sns.barplot(x='Consistent', y=measure, data=avgdata, ci=68, order=[1.0, 0.0], 
                palette='Set2', ax=ax0, errcolor='black', edgecolor='black', linewidth=2, capsize=.2)
    if measure=='Hit':
        ax0.set_ylabel('Accuracy', font=fpath, fontsize=34)
    elif measure=='DPrime':
        ax0.set_ylabel('d\'', font=fpath, fontsize=34)
    elif measure=='Criterion':
        ax0.set_ylabel('Criterion', font=fpath, fontsize=34)
    plt.yticks(font=fpath, fontsize=28) 
    ax0.tick_params(axis='y', direction='out', color='black', length=10, width=2)
    ax0.tick_params(axis='x', length=0, pad=15)
    ax0.set_xlabel(None)
    ax0.set_xticklabels(['Cong.', 'Incong.'], font=fpath, fontsize=34)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    if measure=='Hit':
        ax0.set(ylim=(0.5, 0.75))
    elif measure=='DPrime':
        ax0.set(ylim=(0.0, 1.0))
    elif measure=='Criterion':
        ax0.set(ylim=(0.0, 1.0))
    
    ax1 = fig.add_subplot(122)
    sns.violinplot(y='difference', data=alldiffs, color=".8", inner=None)
    sns.stripplot(y='difference', data=alldiffs, jitter=0.07, ax=ax1, color='black', alpha=.5)
    # Get mean and 95% CI:
    meandiff = alldiffs['difference'].mean()
    tstats = pg.ttest(alldiffs['difference'], 0.0)
    ci95 = tstats['CI95%'][0]
    for tick in ax1.get_xticks():
        ax1.plot([tick-0.1, tick+0.1], [meandiff, meandiff],
                    lw=4, color='k')
        ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
        ax1.plot([tick-0.03, tick+0.03], [ci95[0], ci95[0]], lw=3, color='k')
        ax1.plot([tick-0.03, tick+0.03], [ci95[1], ci95[1]], lw=3, color='k')
    ax1.axhline(0.0, linestyle='--', color='black')
    plt.yticks(font=fpath, fontsize=28) 
    if measure=='Hit':
        ax1.set_ylabel('Δ Accuracy', font=fpath, fontsize=34)
        if excl:
            ax1.set(ylim=(-0.2, 0.4))
        else:
            ax1.set(ylim=(-0.3, 0.4))
    elif measure=='DPrime':
        ax1.set_ylabel('Δ d\'', font=fpath, fontsize=34)
        ax1.set(ylim=(-2., 2.))
    elif measure=='Criterion':
        ax1.set_ylabel('Δ Criterion', font=fpath, fontsize=34)
        ax1.set(ylim=(-1.0, 1.25))
    ax1.axes_style = 'white'
    ax1.tick_params(axis='y', direction='out', color='black', length=10, width=2)
    ax1.tick_params(axis='x', length=0)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    if not fname:
        fname = f'behavior_{measure}.svg'
        if not excl:
            fname.replace('.pdf', '_noexcl.svg')
    if saveimg:
        if not os.path.isdir('results_plots'):
            os.mkdir('results_plots')
        plt.savefig(os.path.join('results_plots', fname))
