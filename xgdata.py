#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas
pandas.options.mode.chained_assignment = None
from numpy import loadtxt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
import whoscored_custom_events as wsce
import whoscored_data_engineering as wsde
from mplsoccer import Standardizer
import random
import joblib
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import ast

#modeldict = dict()
#for i in range(10):
#    name = 'model' + str(i)
#    modeldict[name] = keras.models.load_model('xgmodels/' + str(name) + '.h5')
    
import xgboost as xgb

model = xgb.Booster()
model.load_model('xg.bin')
# In[ ]:


def get_xg(data, nmodels=5):
    nmodels = nmodels
    alldata = data
    matches = alldata['matchId'].unique()
    finshots = pandas.DataFrame()
    for n in matches:
        dataset = alldata.loc[alldata['matchId'] == n]
        shots = dataset.loc[dataset['isShot']==True]
        shots = shots.loc[shots['goalOwn'] == False].reset_index()

        related = shots['relatedEventId']
        shots['relatedevent'] = False
        for i in range(len(shots)):

            related = shots.iloc[i]['relatedEventId']
            if related != np.nan:
                ev_r = dataset.loc[dataset['eventId'] == float(related)]
                ev_r = ev_r.loc[ev_r['teamId'] == shots.iloc[i]['teamId']]
                ev_r = ev_r.loc[ev_r['matchId'] == shots.iloc[i]['matchId']]
                ev_r = ev_r['satisfiedEventsTypes']
                shots['relatedevent'].iloc[i] = ev_r
        
        shots_training = shots[['x','y','isGoal','shotBodyType','situation', 'shotCounter',
                                'bigChanceMissed','bigChanceScored', 'Women']].reset_index(drop=True)
        
        print('data engineering')
        
        stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
        x, y  = stand.transform(shots_training['x'],shots_training['y'])

        shots_training['x'] = x
        shots_training['y'] = y
        lf = shots_training['shotBodyType'] == 'LeftFoot'
        rf = shots_training['shotBodyType'] == 'RightFoot'
        foot = lf + rf
        h = shots_training['shotBodyType'] == 'Head'


        shots_training['Foot'] = foot

        shots_training['Head'] = h

        shots_training['OpenPlay'] = shots_training['situation']=='OpenPlay'
        shots_training['SetPiece'] = shots_training['situation']=='SetPiece'
        shots_training['FromCorner'] = shots_training['situation']=='FromCorner'
        shots_training['DirectFreekick'] = shots_training['situation']=='DirectFreekick'

        m1 = shots_training['bigChanceMissed'] == True
        m2 = shots_training['bigChanceScored'] == True
        mask = m1 + m2
        shots_training['BigChance'] = mask
        
        shots_training = shots_training*1
        
        matchids = dataset['matchId'].unique()
        
        
        recipient = []
        for n in matchids:
            df = dataset.loc[dataset['matchId'] == n]
            passdf = wsde.get_recipient(df)
            rec = list(passdf['pass_recipient'].values)
            recipient = recipient + rec


        dataset['pass_recipient'] = recipient
        dataset['cumulative_mins'] = dataset['minute'] + (dataset['second']/60)

        dataset = dataset
        
        shots['Assisted'] = 0
        for i in range(len(shots)):
            print(str(i) + ' / ' + str(len(shots)), end = '\r')
            dat = shots.iloc[i]['qualifiers']
            if "type': 'Assisted'" in dat:
                shots['Assisted'].iloc[i] = 1
                
        shots['takeOn'] = 0
        shots['a_passCrossAccurate'] = 0
        shots['a_passThroughBallAccurate'] = 0
        shots['a_passChipped'] = 0
        shots['last_action'] = 0
        shots['a_def'] = 0
        shots['a_pass'] = 0

        for i in range(len(shots)):

            related = shots['relatedevent'].iloc[i]
            if isinstance(related, list) and len(related) != 0:
                related = list(related.iloc[0])
                if 'passCrossAccurate' in related:
                    shots['a_passCrossAccurate'].iloc[i] = 1
                elif 'passThroughBallAccurate' in related:
                    shots['a_passThroughBallAccurate'].iloc[i] = 1
                elif 'passChipped' in related:
                    shots['a_passChipped'].iloc[i] = 1
                elif 'passAccurate' in related:
                    shots['a_pass'].iloc[i] = 1
            
            
            if shots['Assisted'].iloc[i] == 0:
                print(str(i) + ' / ' + str(len(shots)), end = '\r')
                last_action = shots['eventId'].iloc[i]-1
                match = shots['matchId'].iloc[i]
                team = shots['teamId'].iloc[i]
                la = dataset.loc[(dataset['matchId']==match) & (dataset['teamId'] == team) &
                                 (dataset['eventId'] == last_action)]
                if len(la) != 0:
#                    if la['outcomeType'].iloc[0] in ['Successful', 'Unsuccessful']:
                    if la['type'].iloc[0] in ['Interception', 'Tackle', 'BallRecovery']:
                        shots['a_def'].iloc[i] = 1
                    if la['outcomeType'].iloc[0]=='Successful':
                        if la['type'].iloc[0] == 'TakeOn':
                            shots['takeOn'].iloc[i] = 1
                shots['last_action'].iloc[i] = la['type']
        
        before_shot = shots[['Assisted', 'a_passCrossAccurate', 'a_passChipped', 'a_passThroughBallAccurate',
                       'a_def', 'takeOn', 'a_pass']]
        before_shot = before_shot.reset_index(drop=True) * 1
        
        before_shot = before_shot.fillna(0)
        
        df1 = before_shot[['Assisted', 'takeOn', 'a_passCrossAccurate', 'a_passThroughBallAccurate', 'a_def', 'a_passChipped', 'a_pass']]
        df1['last_action'] = shots['last_action']
        df2 = shots_training[['x', 'y', 'isGoal', 'shotCounter',
               'BigChance', 'Foot', 'Head',
               'OpenPlay', 'SetPiece', 'FromCorner', 'DirectFreekick', 'Women']]
        df = pandas.concat([df2, df1], axis = 1)
        
        # do r and theta

        df['r'] = np.sqrt((120 - df['x'])**2 + (df['y'] - 40)**2)
        ind = df['r'] <=1
        df['r'].iloc[ind] = 1
        # Calculate the distance to the left post (100, 45)
        distance_to_left_post = np.sqrt((120 - df['x'])**2 + (df['y'] - 36)**2)

        # Calculate the distance to the right post (100, 55)
        distance_to_right_post = np.sqrt((120 - df['x'])**2 + (df['y'] - 44)**2)

        # Create a new column to store the angle to the nearest post
        df['angle_to_nearest_post'] = np.where(distance_to_left_post <= distance_to_right_post,
                                               np.arctan2(df['y'] - 36, 120 - df['x']),
                                               np.arctan2(df['y'] - 44, 120 - df['x']))

        # Create a new column to store the scaled angle
        df['theta'] = np.where((df['y'] >= 36) & (df['y'] <= 44), 1, 1 - np.abs(df['angle_to_nearest_post']) / (np.pi / 2))
        df['1/r'] = 1/df['r']

        scale_x = joblib.load('scale_x.gz')
        loc = df[['r', '1/r']]
        x = scale_x.fit_transform(loc)
        xtrain = df[['r', '1/r', 'theta', 'isGoal', 'shotCounter',
                   'BigChance', 'Foot', 'Head','Women',
                   'OpenPlay', 'SetPiece', 'FromCorner', 'DirectFreekick', 'Assisted', 'takeOn', 'a_passCrossAccurate',
                'a_passThroughBallAccurate', 'a_def', 'a_passChipped', 'last_action', 'a_pass']]

        xtrain['r'] = x[:,0]
        xtrain['1/r'] = x[:,1]

        y = df['isGoal']

        xtrain = xtrain[['r', '1/r', 'theta', 'shotCounter',
               'BigChance', 'Foot', 'Head','Women',
               'OpenPlay', 'SetPiece', 'FromCorner', 'DirectFreekick',
                'Assisted', 'takeOn', 'a_passCrossAccurate',
                'a_passThroughBallAccurate', 'a_def', 'a_passChipped', 'last_action', 'a_pass']]
                    

#        xtrain['BigChance'] = (xtrain['BigChance'] * 0.8)+0.2
#        xtrain['BigChance'] = (xtrain['BigChance'] * (1/(1+xtrain['r'])))
    #    xtrain['Assisted'] = (xtrain['Assisted'] * 0.6) + 0.4
        xtrain['SetPiece'] = (xtrain['SetPiece'] + xtrain['FromCorner'])
        
#        print(xtrain[['r', '1/r']])
        x_pred = np.asarray(xtrain[['r', 'theta','shotCounter',
               'BigChance', 'Foot', 'Head','Women',
               'OpenPlay', 'SetPiece', 'DirectFreekick',
                'Assisted', 'takeOn', 'a_passCrossAccurate',
                'a_passThroughBallAccurate', 'a_def', 'a_passChipped']]).astype('float32')
        


        x_train = xtrain[['r','theta','shotCounter',
           'BigChance', 'Foot', 'Head', 'Women',
           'OpenPlay', 'SetPiece','DirectFreekick',
            'Assisted', 'takeOn', 'a_passCrossAccurate',
            'a_passThroughBallAccurate', 'a_def', 'a_passChipped']]
            
        counter = np.array(x_train['shotCounter'], dtype='float32').flatten()
        x_train['shotCounter'] = counter
        
                
        X_new_dmatrix = xgb.DMatrix(x_train)
        xg = model.predict(X_new_dmatrix)

        print(xg)
        shots['xG'] = xg
        shots['assist_cross'] = xtrain['a_passCrossAccurate']
        shots['assist_throughball'] = xtrain['a_passThroughBallAccurate']
        shots['assist_pass'] = xtrain['a_pass']+xtrain['a_passChipped']
        shots['assist_def'] = xtrain['a_def']
        shots['takeOn'] = xtrain['takeOn']

        for i in range(len(shots)):
            if shots['penaltyMissed'].iloc[i] == True:
                shots['xG'].iloc[i] = 0.78
            elif shots['penaltyScored'].iloc[i] == True:
                shots['xG'].iloc[i] = 0.78
        finshots = pandas.concat((finshots,shots))
    
    return finshots
