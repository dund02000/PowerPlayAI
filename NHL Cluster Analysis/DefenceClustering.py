# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:42:28 2020

@author: Patrick
"""

#################################################################################Now Defense

# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:08:27 2020

@author: John
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

Fullplayerlistd = pd.read_csv('FPLdef.csv')
Fullplayerlistd = Fullplayerlistd[Fullplayerlistd['GP']>= 10 ]   

del Fullplayerlistd['Clause']
del Fullplayerlistd['Contract expiry']
del Fullplayerlistd['Free Agent Type']
del Fullplayerlistd['ATOI']
del Fullplayerlistd['TOI/60']
del Fullplayerlistd['TOI(EV)']

#Fill NA values with 0
Fullplayerlistd = Fullplayerlistd.fillna(0)

#Multiply cap hit % by 100
Fullplayerlistd['Current cap %'] = Fullplayerlistd['Current cap %']*100

#Create new columns
Fullplayerlistd['ATOI'] = Fullplayerlistd['TOI']/Fullplayerlistd['GP']
Fullplayerlistd['PIM/G'] = Fullplayerlistd['PIM']/Fullplayerlistd['GP']
Fullplayerlistd['G/G'] = Fullplayerlistd['G']/Fullplayerlistd['GP']
Fullplayerlistd['A/G'] = Fullplayerlistd['A']/Fullplayerlistd['GP']
Fullplayerlistd['PTS/G'] = Fullplayerlistd['PTS']/Fullplayerlistd['GP']
Fullplayerlistd['S/G'] = Fullplayerlistd['S']/Fullplayerlistd['GP']
Fullplayerlistd['SHP/G'] = (Fullplayerlistd['SH']+Fullplayerlistd['SH1'])/Fullplayerlistd['GP']
Fullplayerlistd['Giveaways/G'] = Fullplayerlistd['Giveaways']/Fullplayerlistd['GP']
Fullplayerlistd['Takeaways/G'] = Fullplayerlistd['Takeaways']/Fullplayerlistd['GP']
Fullplayerlistd['Hits/G'] = Fullplayerlistd['Hit']/Fullplayerlistd['GP']
Fullplayerlistd['BLK/G'] = Fullplayerlistd['BLK']/Fullplayerlistd['GP']
Fullplayerlistd['Rush Attempts/G'] = Fullplayerlistd['Rush Attempts']/Fullplayerlistd['GP']
Fullplayerlistd['Rebounds Created/G'] = Fullplayerlistd['Rebounds Created']/Fullplayerlistd['GP']
Fullplayerlistd['Penalties Drawn/G'] = Fullplayerlistd['Penalties Drawn']/Fullplayerlistd['GP']

#modify columns
Fullplayerlistd['Height'] = Fullplayerlistd['Height'].str[:2]
Fullplayerlistd['Weight'] = Fullplayerlistd['Weight'].str[:3]

#Feauture selection: Drop columns directly correlated to player success
Fullplayerlistd = Fullplayerlistd.drop(['Cap hit', 'Current cap %', 'Height', 'Weight', 'GP', 'G', 'A', 'PTS', '+/-',
                         'S', 'PIM', 'EV', 'PP', 'SH', 'GW', 'EV1', 'SH1', 'PP1', 'ixG', 'iCF', 'iSCF',
                         'iHDCF', 'Rush Attempts', 'Rebounds Created', 'Penalties Drawn', 'Giveaways', 
                         'Takeaways', 'Hit', 'Hits Taken', 'BLK', 'FOW', 'FOL', 'TOI', 'CF', 'CA', 'FF',
                         'FA', 'G/G', 'A/G', 'CF% rel', 'FF% rel'], axis = 1)

#####K MEANS CLUSTER
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist, pdist, euclidean
from sklearn.cluster import KMeans
from sklearn import metrics

X2 = Fullplayerlistd._get_numeric_data().dropna(axis=1)

del X2['UFA']
del X2['Age']


df2 = pd.DataFrame(X2)
X2 = scale(X2)

Player2 = Fullplayerlistd['Player']

#DETERMINE # OF VARIABLES TO USE
pca2 = PCA(n_components = 28)
pca2.fit(X2)
var2 = pca2.explained_variance_ratio_
var3 = np.cumsum(np.round(pca2.explained_variance_ratio_, decimals = 4)*100)
print(var3)
plt.plot(var3)

pca2 = PCA(n_components = 21)
pca2.fit(X2)
X3 = pca2.fit_transform(X2)
loadings_df2 = pd.DataFrame(pca2.components_, columns = df2.columns)

#Determine # of clusters are best
new_data2 = PCA(n_components = 21, whiten=True).fit_transform(X2)
k_range2 = range(2,15)
k_means_var2 = [KMeans(n_clusters=k).fit(new_data2) for k in k_range2]
labels2 = [i.labels_ for i in k_means_var2]
sil_score2 = [metrics.silhouette_score(new_data2,i,metric='euclidean') for i in labels2]
centroids2 = [i.cluster_centers_ for i in k_means_var2]
k_euclid2 = [cdist(new_data2,cent,'euclidean') for cent in centroids2]
dist2 = [np.min(ke,axis=1) for ke in k_euclid2]
wcss2 = [sum(d**2) for d in dist2]
tss2 = sum(pdist(new_data2)**2/new_data2.shape[0])
bss2 = tss2-wcss2

plt.clf()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(k_range2, bss2/tss2*100, 'o-')
plt.axis([0, np.max(k_range2), 0, 100])
plt.xlabel('Number of Clusters')
plt.ylabel('% of variance explaned');

plt.subplot(1,2,2)
plt.plot(k_range2, np.transpose(sil_score2)*100, 'o-')
plt.axis([0, np.max(k_range2), 0, 40])
plt.xlabel('# of Clusters')
plt.ylabel('AVG Silhouette Score*100');

#We choose 4 clusters

fit_clusters2 = KMeans(n_clusters=4, random_state=1).fit(new_data2)
df2['kmeans_label'] = fit_clusters2.labels_
df2['Player'] = Player2

#Check how many players in each cluster

len(df2[df2['kmeans_label']==0])
len(df2[df2['kmeans_label']==1])
len(df2[df2['kmeans_label']==2])
len(df2[df2['kmeans_label']==3])

#Naming the clusters

Cluster1def = df2[df2['kmeans_label']==0]['Player']
Cluster2def = df2[df2['kmeans_label']==1]['Player']
Cluster3def = df2[df2['kmeans_label']==2]['Player']
Cluster4def = df2[df2['kmeans_label']==3]['Player']

### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
new_data2 = pd.DataFrame(new_data2)
new_data2['kmeans_label'] = fit_clusters2.labels_


reduced_data2 = PCA(n_components=2).fit_transform(new_data2)
results2 = pd.DataFrame(reduced_data2,columns=['pca1','pca2'])

results2['kmeans_label'] = fit_clusters2.labels_
results2['Player'] = Player2

sns.scatterplot(x="pca1", y="pca2", hue=new_data2['kmeans_label'], data=results2)
plt.title('K-means Clustering with 2 dimensions')
plt.show()


Cluster1def.to_csv('Cluster1defence.csv')
Cluster2def.to_csv('Cluster2defence.csv')
Cluster3def.to_csv('Cluster3defence.csv')
Cluster4def.to_csv('Cluster4defence.csv')
results2.to_csv('resultsdefence.csv')
