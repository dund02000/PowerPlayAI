# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:08:27 2020

@author: Patrick
"""
import os
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math

#Player clustering - Forwards 
#Import full player list who played at least 20 games in 2019-2020

Fullplayerlistf = pd.read_csv('FPLfor.csv')
Fullplayerlistf = Fullplayerlistf[Fullplayerlistf['GP']>= 10 ]   

del Fullplayerlistf['Clause']
del Fullplayerlistf['Contract expiry']
del Fullplayerlistf['Free Agent Type']
del Fullplayerlistf['ATOI']
del Fullplayerlistf['TOI/60']
del Fullplayerlistf['TOI(EV)']

#Fill NA values with 0
Fullplayerlistf = Fullplayerlistf.fillna(0)

#Multiply cap hit % by 100
Fullplayerlistf['Current cap %'] = Fullplayerlistf['Current cap %']*100

#Create new columns
Fullplayerlistf['ATOI'] = Fullplayerlistf['TOI']/Fullplayerlistf['GP']
Fullplayerlistf['PIM/G'] = Fullplayerlistf['PIM']/Fullplayerlistf['GP']
Fullplayerlistf['G/G'] = Fullplayerlistf['G']/Fullplayerlistf['GP']
Fullplayerlistf['A/G'] = Fullplayerlistf['A']/Fullplayerlistf['GP']
Fullplayerlistf['PTS/G'] = Fullplayerlistf['PTS']/Fullplayerlistf['GP']
Fullplayerlistf['S/G'] = Fullplayerlistf['S']/Fullplayerlistf['GP']
Fullplayerlistf['SHP/G'] = (Fullplayerlistf['SH']+Fullplayerlistf['SH1'])/Fullplayerlistf['GP']
Fullplayerlistf['Giveaways/G'] = Fullplayerlistf['Giveaways']/Fullplayerlistf['GP']
Fullplayerlistf['Takeaways/G'] = Fullplayerlistf['Takeaways']/Fullplayerlistf['GP']
Fullplayerlistf['Hits/G'] = Fullplayerlistf['Hit']/Fullplayerlistf['GP']
Fullplayerlistf['BLK/G'] = Fullplayerlistf['BLK']/Fullplayerlistf['GP']
Fullplayerlistf['Rush Attempts/G'] = Fullplayerlistf['Rush Attempts']/Fullplayerlistf['GP']
Fullplayerlistf['Rebounds Created/G'] = Fullplayerlistf['Rebounds Created']/Fullplayerlistf['GP']
Fullplayerlistf['Penalties Drawn/G'] = Fullplayerlistf['Penalties Drawn']/Fullplayerlistf['GP']

#modify columns
Fullplayerlistf['Height'] = Fullplayerlistf['Height'].str[:2]
Fullplayerlistf['Weight'] = Fullplayerlistf['Weight'].str[:3]
Fullplayerlistf['Position'] = Fullplayerlistf['Position'].str[:1]

#Feauture selection: Drop columns directly correlated to player success
Fullplayerlistf = Fullplayerlistf.drop(['Cap hit', 'Current cap %', 'Height', 'Weight', 'GP', 'G', 'A', 'PTS', '+/-',
                         'S', 'PIM', 'EV', 'PP', 'SH', 'GW', 'EV1', 'SH1', 'PP1', 'ixG', 'iCF', 'iSCF',
                         'iHDCF', 'Rush Attempts', 'Rebounds Created', 'Penalties Drawn', 'Giveaways', 
                         'Takeaways', 'Hit', 'Hits Taken', 'BLK', 'FOW', 'FOL', 'TOI', 'CF', 'CA', 'FF',
                         'FA', 'G/G', 'A/G', 'CF% rel', 'FF% rel'], axis = 1)

import seaborn as sns
# load the R package ISLR
corr = Fullplayerlistf.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#####K MEANS CLUSTER
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.cluster import KMeans
from sklearn import metrics

X = Fullplayerlistf._get_numeric_data().dropna(axis=1)

del X['UFA']
del X['Age']


df = pd.DataFrame(X)
X = scale(X)

Player = Fullplayerlistf['Player']

#DETERMINE # OF VARIABLES TO USE
pca = PCA(n_components = 28)
pca.fit(X)
var = pca.explained_variance_ratio_
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
print(var1)
plt.plot(var1)

pca = PCA(n_components = 21)
pca.fit(X)
X1 = pca.fit_transform(X)
loadings_df = pd.DataFrame(pca.components_, columns = df.columns)

#Determine # of clusters are best
new_data = PCA(n_components = 21, whiten=True).fit_transform(X)
k_range = range(2,15)
k_means_var = [KMeans(n_clusters=k).fit(new_data) for k in k_range]
labels = [i.labels_ for i in k_means_var]
sil_score = [metrics.silhouette_score(new_data,i,metric='euclidean') for i in labels]
centroids = [i.cluster_centers_ for i in k_means_var]
k_euclid = [cdist(new_data,cent,'euclidean') for cent in centroids]
dist = [np.min(ke,axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(new_data)**2/new_data.shape[0])
bss = tss-wcss

plt.clf()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(k_range, bss/tss*100, 'o-')
plt.axis([0, np.max(k_range), 0, 100])
plt.xlabel('Number of Clusters')
plt.ylabel('% of variance explaned');

plt.subplot(1,2,2)
plt.plot(k_range, np.transpose(sil_score)*100, 'o-')
plt.axis([0, np.max(k_range), 0, 40])
plt.xlabel('# of Clusters')
plt.ylabel('AVG Silhouette Score*100');

#We choose 5 clusters

fit_clusters = KMeans(n_clusters=5, random_state=1).fit(new_data)
df['kmeans_label'] = fit_clusters.labels_
df['Player'] = Player

#Check how many players in each cluster

len(df[df['kmeans_label']==0])
len(df[df['kmeans_label']==1])
len(df[df['kmeans_label']==2])
len(df[df['kmeans_label']==3])
len(df[df['kmeans_label']==4])

#Naming the clusters

Cluster1for = df[df['kmeans_label']==0]['Player']
Cluster2for = df[df['kmeans_label']==1]['Player']
Cluster3for = df[df['kmeans_label']==2]['Player']
Cluster4for = df[df['kmeans_label']==3]['Player']
Cluster5for = df[df['kmeans_label']==4]['Player']


### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
new_data = pd.DataFrame(new_data)
new_data['kmeans_label'] = fit_clusters.labels_


reduced_data = PCA(n_components=2).fit_transform(new_data)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

results['kmeans_label'] = fit_clusters.labels_
results['Player'] = Player

sns.scatterplot(x="pca1", y="pca2", hue=new_data['kmeans_label'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


Cluster1for.to_csv('Cluster1forwards.csv')
Cluster2for.to_csv('Cluster2forwards.csv')
Cluster3for.to_csv('Cluster3forwards.csv')
Cluster4for.to_csv('Cluster4forwards.csv')
Cluster5for.to_csv('Cluster5forwards.csv')
results.to_csv('resultsforwards.csv')

