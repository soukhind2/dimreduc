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
from sklearn.feature_selection import SelectKBest, RFECV, f_classif
from sklearn.pipeline import Pipeline

#%matplotlib inline
#%autosave 5
sns.set(style = 'white', context='poster', rc={'lines.linewidth': 2.5})
sns.set(palette="colorblind")

#%%
# load some helper functions
from brainiak.utils.utils import load_labels, load_data, blockwise_sampling, label2TR, shift_timing, reshape_data
from brainiak.utils import normalize, decode
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
