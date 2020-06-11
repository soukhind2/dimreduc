#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 07:48:42 2020

@author: soukhind
"""

'''
Works in conjunction with main.py ONLY
Soukhin Das
06th June 2020
'''

#%%
logtime = np.log(pcatime)
fig = plt.figure()

plt.plot(logtime)
plt.xlabel("log(time)")
plt.ylabel
plt.xlim(0,40)
plt.xticks(np.arange(41))

#%%
# Plot accuracy vs pca components
plt.style.use('seaborn-whitegrid')
fig= plt.figure(figsize = (12,8))
ax = fig.add_subplot(111)
plt.plot(pca_score[:,0],'-D',label = 'Face',c = 'lime' , alpha = 0.4)
plt.plot(pca_score[:,1],'-*',label = 'Scene',c = 'blue', alpha = 0.4)
plt.plot(pca_score[:,2],'-x',label = 'Object',c = 'orange', alpha = 0.4)
plt.plot(pca_score[:,3],'-o',label = 'Overall',c = 'red',linewidth = 2)   
plt.title("Classifier Performance on Training Data",size = 25)
l = plt.legend(loc = 4,title = 'Category')
plt.setp(l.get_title(), multialignment='center')
#ax.set_xlabel('Upper Bound of SOA')
ax.set_ylabel('Classification Accuracy',size = 25)
ax.set_xlabel('Principal Components',size = 25)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)
#%%
#plot pca cross validation time
fig = plt.figure()
ax = fig.add_subplot(111)

plt.style.use('seaborn-whitegrid')
plt.plot(all_time[:,2])
plt.xlabel("PCA Components",size = 20)
plt.ylabel("Time (s)",size = 20)
plt.title("TIme taken for SVM with 3 fold Cross Validation",size = 20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)