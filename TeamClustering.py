# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:12:37 2020

@author: Patrick
"""

import os
import pandas as pd

#Team Clustering

playercluster = pd.read_csv('NHLPlayers1920Clustered.csv')
teamstats = pd.read_csv('TeamStats.csv')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math

playercluster = playercluster[['Player','Team','Cluster','UFA']]
playercluster = playercluster.fillna(0)

#Create roster composition counts for each team

team_cluster_dummies = pd.get_dummies(playercluster['Cluster'])

team_cluster_dummies['Team'] = playercluster['Team']

team_roster_composition = team_cluster_dummies.groupby(['Team']).sum()

team_roster_composition.head()

#add some team stats to roster composition
team_roster_composition = pd.DataFrame(team_roster_composition)
team_roster_composition['Team'] = team_roster_composition.index

team_stats = teamstats[['Team', 'PP%','PK%', 'S%', 'SV%']]
ros_comp_team_stats = team_roster_composition.merge(team_stats, left_on='Team', right_on='Team')

team_roster_composition['Team'] = team_roster_composition.index

team_roster_composition.to_csv('rostercomposition.csv')
team_stats.to_csv('team_statsSHORT.csv')

team_roster_composition2 = pd.read_csv('Readyforteamcluster.csv')



