#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 00:49:03 2020

@author: soukhind
"""

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter('ignore')

# Import neuroimaging, analysis and general libraries
import numpy as np
from time import time
import pandas as pd

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import cross_val_score, cross_validate, PredefinedSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#from sklearn.feature_selection import SelectKBest, RFECV, SelectKBestf_classif
from sklearn.pipeline import Pipeline

#%matplotlib inline
#%autosave 5
sns.set(style = 'white', context='poster', rc={'lines.linewidth': 2.5})
sns.set(palette="colorblind")

#%%
# load some helper functions
from utils import load_labels, load_data, blockwise_sampling, label2TR, shift_timing, reshape_data
from utils import normalize, decode
# load some constants
from utils import vdc_data_dir, vdc_all_ROIs, vdc_label_dict, vdc_n_runs, vdc_hrf_lag, vdc_TR, vdc_TRs_run


print('Here\'re some constants, which is specific for VDC data:')

print('data dir = %s' % (vdc_data_dir))
print('ROIs = %s' % (vdc_all_ROIs))
print('Labels = %s' % (vdc_label_dict))
print('number of runs = %s' % (vdc_n_runs))
print('1 TR = %.2f sec' % (vdc_TR))
print('HRF lag = %.2f sec' % (vdc_hrf_lag))
print('num TRs per run = %d' % (vdc_TRs_run))

#%%
sub_id = 2
mask_name = 'FFA' # This is set in order to reduce memory demands in order to run within 4Gb, however, if you want to make this run on whole brain, then set this to ''

# Specify the subject name
sub = 'sub-%.2d' % (sub_id)
# Convert the shift into TRs
shift_size = int(vdc_hrf_lag / vdc_TR)  

# Load subject labels
stim_label_allruns = load_labels(vdc_data_dir, sub)

# Load run_ids
run_ids_raw = stim_label_allruns[5,:] - 1 

# Load the fMRI data using a mask
epi_mask_data_all = load_data(vdc_data_dir, sub, mask_name=mask_name)[0]

# This can differ per participant
print(sub, '= TRs: ', epi_mask_data_all.shape[1], '; Voxels: ', epi_mask_data_all.shape[0])
TRs_run = int(epi_mask_data_all.shape[1] / vdc_n_runs)

# Convert the timing into TR indexes
stim_label_TR = label2TR(stim_label_allruns, vdc_n_runs, vdc_TR, TRs_run)

# Shift the data some amount
stim_label_TR_shifted = shift_timing(stim_label_TR, shift_size)

# Perform the reshaping of the data
bold_data_raw, labels_raw = reshape_data(stim_label_TR_shifted, epi_mask_data_all)

# Normalize raw data within each run
bold_normalized_raw = normalize(bold_data_raw, run_ids_raw)

# Down sample the data to be blockwise rather than trialwise. 
#We'll use the blockwise data for all the 
bold_data, labels, run_ids = blockwise_sampling(bold_data_raw, labels_raw, run_ids_raw)

# Normalize blockwise data within each run
bold_normalized = normalize(bold_data, run_ids)

#%%
bold_cov = np.cov(bold_data[:,0:500].T)
bold_corr = np.corrcoef(bold_data[:,0:500].T)
bold_corr_norm = np.cov(bold_normalized[:,1000:1500].T)

#plt.imshow(bold_cov)
#plt.imshow(bold_corr)
plt.imshow(bold_corr_norm,cmap = 'jet')
plt.colorbar()
plt.xlabel("Voxel #",size = 20)
plt.ylabel("Voxel #",size = 20)
plt.title("Correlation Matrix of 500 voxels",size = 20)
#%%
# We now use the PCA function in scikit-learn to reduce the dimensionality of the data
# The number of components was chosen arbitrarily.
n = 40
pca = PCA(n_components=n)
bold_pca = pca.fit_transform(bold_normalized)

print('Original data shape:', bold_normalized.shape)
print('PCA data shape:', bold_pca.shape)
#%%
# Setting plotting parameter
n_bins=75

# Plot
n_plots = 4
components_to_plot = [0,1,2,19]
f, axes = plt.subplots(1, n_plots, figsize=(14, 14/n_plots))
#st=f.suptitle("Figure 3.1. Histogram of values for each PC dimension ", fontsize="x-large")

for i in range(n_plots): 
    axes[i].hist(bold_pca[:, components_to_plot[i]], 
                 bins=n_bins)
    # mark the plots 
    axes[i].set_title('PC Dimension %d'%(components_to_plot[i]+1),size = 20)
    axes[i].set_ylabel('Frequency',size = 20)
    axes[i].set_xlabel('Value',size = 20)    
    axes[i].set_xticks([])
    axes[i].set_yticks([])    

f.tight_layout()
st.set_y(0.95)
f.subplots_adjust(top=0.75)
#%%
# Setting plotting parameters
alpha_val = .8
cur_pals = sns.color_palette('colorblind', n_colors=vdc_n_runs)
 
# Plot
n_plots = 3 
f, axes = plt.subplots(1, n_plots, figsize=(14,5))
#st=f.suptitle("Figure 3.2. Scatter plots comparing PCA dimensions ", fontsize="x-large")

# plot data
axes[0].scatter(bold_pca[:, 0], bold_pca[:, 1], 
                alpha=alpha_val, marker='.', color = 'k')
axes[1].scatter(bold_pca[:, 2], bold_pca[:, 3], 
                alpha=alpha_val, marker='.', color = 'k')
axes[2].scatter(bold_pca[:, 18], bold_pca[:, 19], 
                alpha=alpha_val, marker='.', color = 'k')

axes[0].set_title('PCA Dimensions\n1 x 2',size = 20)
axes[1].set_title('PCA Dimensions\n3 x 4',size = 20)
axes[2].set_title('PCA Dimensions\n18 x 19',size = 20)

# modifications that are common to all plots 
for i in range(n_plots): 
    axes[i].axis('equal')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

#%% Scree Plot

fig,ax = plt.subplots()

sing_vals = np.arange(n) + 1
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(sing_vals, cumsum, 'ro-', linewidth=2)
#plt.title('Scree Plot',size = 20)
plt.xlabel('Principal Component',size = 20)
plt.ylabel('Cumulative Variance',size = 20)
plt.xticks(np.arange(1,n+1,2),size=15)
#plt.ylim(0,1)
plt.xlim(0,n+1)

#leg = plt.legend(['Cumulative Explained Variance'], loc='best',shadow=False,
               #  markerscale=0.4)
leg.get_frame().set_alpha(0.4)

#ax.vlines(x=19,ymax = 0.6, ymin = 0, color='grey')
#ax.hlines(y=0.6,xmax = 19, xmin = 0, color='grey')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
plt.show()

#%%
# Run a basic n-fold classification

# Get baseline, whole-brain decoding accuracy without PCA
print('Baseline classification')
print('Original size: ', bold_normalized.shape)
svc = SVC(kernel="linear", C=1)

start = time()
models, scores = decode(bold_normalized, labels, run_ids, svc)
end = time()
print('Accuracy: ', scores)
print('Run time: %0.4fs' %(end - start))

#%% Regular PCA
pca = PCA(n_components= 19)
bold_pca_normalized = pca.fit_transform(bold_normalized)
print('PCA (c=%d) classification' % bold_pca_normalized.shape[1])
print('New size after PCA: ', bold_pca_normalized.shape)

start = time()
models_pca, scores_pca = decode(bold_pca_normalized, labels, run_ids, svc)
end = time()
print('Accuracy: ', scores_pca)
print('Run time: %0.4fs' %(end - start))
#%% PCA loop

pcatime = np.zeros(45)
# Run the classifier on data in component space 
for i in range(1,45):
    pca = PCA(n_components=i)
    bold_pca_normalized = pca.fit_transform(bold_normalized)
    #print('PCA (c=%d) classification' % bold_pca_normalized.shape[1])
    #print('New size after PCA: ', bold_pca_normalized.shape)
    
    start = time()
    models_pca, scores_pca = decode(bold_pca_normalized, labels, run_ids, svc)
    end = time()
    pcatime[i-1] = end - start
    #sys.stdout.flush()

print('Accuracy: ', scores_pca)
print('Run time: %0.4fs' %(end - start))
#%% Baseline data cross validation
ps = PredefinedSplit(run_ids) # Provides train/test indices to split data in train/test sets
clf_pipe = cross_validate(SVC(kernel="linear", C=1),bold_normalized,labels,cv=ps,
        return_train_score=True)
print(clf_pipe)
print ("Average Testing Accuracy: %0.2f" % (np.mean(clf_pipe['test_score'])))

#%% PCA cross validation
pipe = Pipeline([
                ('reduce_dim', PCA(n_components=19)),
                ('classify', SVC(kernel="linear", C=1)),])

ps = PredefinedSplit(run_ids) # Provides train/test indices to split data in train/test sets
clf_pipe = cross_validate(pipe,bold_normalized,labels,cv=ps,
                              return_train_score=True)

print(clf_pipe)
print ("Average Testing Accuracy: %0.2f" % (np.mean(clf_pipe['test_score'])))
#%% Pipeline
for ncomp in range(1,30):
    # Set up the pipeline
    pipe = Pipeline([
        ('reduce_dim', PCA(n_components = ncomp)),
        ('classify', SVC(kernel="linear", C=1)),
    ])
    
    # Run the pipeline with cross-validation
    ps = PredefinedSplit(run_ids) # Provides train/test indices to split data in train/test sets
    clf_pipe = cross_validate(
        pipe,bold_normalized,labels,cv=ps,
        return_train_score=True
    )
    if ncomp == 1:
        pca_score = clf_pipe['test_score']
        fit_time_all = clf_pipe['fit_time']
    else:
        pca_score = np.vstack((pca_score,clf_pipe['test_score']))
        fit_time_all = np.vstack((fit_time_all,clf_pipe['fit_time']))

    print(ncomp)
    
all_score = np.mean(pca_score,1).reshape(pca_score.shape[0],1)
all_time = np.cumsum(fit_time_all,1)
pca_score = np.hstack((pca_score,all_score))
# Print results from this dimensionality reduction technique
print(clf_pipe)
print ("Average Testing Accuracy: %0.2f" % (np.mean(clf_pipe['test_score'])))