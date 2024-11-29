#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import visuals
import whoscored_custom_events as wsce
import whoscored_data_engineering as wsde
pd.options.mode.chained_assignment = None
from mplsoccer.pitch import Pitch, VerticalPitch
import matplotlib as mpl

import pickle
from mplsoccer import Standardizer
import streamlit as st
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ast

def shotmaps(shots, match_file, teamid, teamname, text1, text2, text3, size):
    
    df = shots
#    .loc[shots['teamId'] == teamid]
    penscored = df.loc[df['penaltyScored'] == True]
    penmissed = df.loc[df['penaltyMissed'] == True]
    df = df.loc[df['penaltyScored'] == False]
    df = df.loc[df['penaltyMissed'] == False]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x, y  = stand.transform(df['x'],df['y'])
    df['x'] = x
    df['y'] = y
    
    markers = []
    ec = []
    lw = []
    zorder = np.zeros(len(df['x'])) + 2
    passes = 0
    op_crosses = 0
    sp_crosses = 0
    throughball = 0
    defactions = 0
    directfk = 0
    other =0
    for i in range(len(df['x'])):
        if df['isGoal'].iloc[i] == True:
            ec.append('red')
            lw.append(3)
            zorder[i] = 3
        elif df['shotBlocked'].iloc[i] == True:
            ec.append('white')
            lw.append(3)
        elif df['type'].iloc[i] == 'SavedShot':
            ec.append('cyan')
            lw.append(3)
        elif df['shotOffTarget'].iloc[i] == True:
            ec.append('orange')
            lw.append(2)
        if df['assist_cross'].iloc[i] == True:
            if df['situation'].iloc[i] in ['SetPiece','FromCorner']:
                markers.append('D')
                sp_crosses+=1
            else:
                markers.append('s')
                op_crosses+=1
        elif df['assist_throughball'].iloc[i] == True:
            markers.append('^')
            throughball+=1
        elif df['assist_pass'].iloc[i] == True:
            markers.append('o')
            passes+=1
        elif df['assist_def'].iloc[i] == True:
            markers.append('P')
            defactions+=1
        elif df['situation'].iloc[i] == 'DirectFreekick':
            markers.append('*')
            directfk+=1
        else:
            markers.append('p')
            other+=1


    norm = plt.Normalize(0,1)
    colmap= mpl.colors.LinearSegmentedColormap.from_list("",['white','orange','red','darkred'])
    rescale = lambda y: (y - 0) / (1)
    colour = colmap(rescale(df['xG']))
    
    pitch = VerticalPitch(half = True,pitch_type = 'statsbomb', goal_type='box',
                    pitch_color = 'navy', line_color = 'white', pad_top = 17)

    fig, ax = pitch.draw(figsize = (16,16))
    
    inarea = df.loc[(df['x'] >= 102) & (df['y'] >= 18) & (df['y']<=62)]
    for i in range(len(df['x'])):
        plt.scatter(df['y'].iloc[i],df['x'].iloc[i],
                    fc = colour[i], ec = ec[i],
                    linewidth = lw[i], marker=markers[i], alpha = 0.9,
                    s = size, zorder=zorder[i])

    legend_labels = list(['Goal', 'Saved', 'Blocked', 'Off Target'])
    colors = ['red', 'cyan','white','orange']
    edge = (3,3,3,1)

    for i in range(4):
        plt.scatter(0, -10, marker = 'o', s= 300, c = 'white', ec = colors[i], lw = edge[i],
                    label = legend_labels[i])

    legend = plt.legend(loc =2, fontsize = 20)

    plt.gca().add_artist(legend)
    
    legend_labels = list(['Pass: '+str(passes),'Open Play Cross: '+str(op_crosses), 'Set Piece Cross: '+str(sp_crosses), 'Def Action: '+str(defactions),'Through Ball: '+str(throughball),'Direct Freekick: ' + str(directfk),'Other/Unassisted: '+str(other)])
    colors = ['k', 'k','k','k','k','k','k']
    edge = (1,1,1,1,1,1,1)
    mark = ['o','s','D', 'P', '^', '*','p']

    scatter = [ax.scatter(0, -10, marker = mark[i], s= 200, c = 'white', ec = colors[i], lw = edge[i],
                           label = legend_labels[i])for i in range(7)]

    plt.legend(handles = scatter,loc = 1, fontsize = 16, title = 'Assist', title_fontsize=20)

    axins = inset_axes(ax,
                        width="50%",
                        height='5%',
                        loc='upper center',
                        )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colmap),
                 ax=ax, cax = axins, orientation = 'horizontal')

    cbar.set_label('xG', size=24, color = 'white')

    cbar.ax.tick_params(labelsize=21, color = 'white', labelcolor = 'white')

    ax.scatter(12, np.mean(df['x']), s=300, c='white', ec='white')


    ax.plot((12,12), (120,np.mean(df['x'])), c='white', lw=2)


    ax.text(s='Average \nShot Distance\n' +str(np.round(np.mean(120-df['x']),1)) + 'm',
           x =9.2, y= np.mean(df['x'])+1.5, c='white', rotation = 90, size = 16)
           
#    if venue == 'H':
#        venue = 'At Home'
#    elif venue == 'A':
#        venue = 'Away'

    plt.title('Shot Map - ' + str(text1) + '\n' + str(text2) + '\n' + str(text3) + '\nEFL Championship 2024/25', c='white', size = 32, y=-15)

    plt.scatter(40,108,c='white',s=100)

    plt.text(0.95,-17.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta\nxG via Potterlytics',fontsize = 18,
             c='white')
             

             
    plt.text(-0.3,-17.5, s= str(int(sum(df['isGoal']))) + ' goal(s) from ' + str(np.round(sum(df['xG']),2)) + ' non-penalty xG\n'
             + str(np.round(sum(df['xG'])/len(df), 2)) + ' xG per Shot\n'
             + str(len(df)) + ' shots, '+ str(len(df.loc[df['shotOnTarget'] == True])) + ' on target\n'
             + str(len(df.loc[df['xG'] > 0.2])) + ' shots > 0.2xG'  + ' + ' + str(len(penscored)) + '/' + str(len(penscored) + len(penmissed)) + ' penalties',
             fontsize = 18, c='white')

    return fig, ax
    
def fullpitch_pass(events_df, text1, text2, text3):

    df = events_df
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'navy', line_color = 'white',
                          pad_top = 20, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (12,8))
    for i in range(len(df)):
        if df['assist'].iloc[i] == True:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'orange',lw = 2), zorder = 3)
        elif df['passKey'].iloc[i] == True:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'magenta',lw = 2), zorder = 2)
        elif df['passCrossAccurate'].iloc[i] == True:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'cyan', lw = 1.5), zorder = 2)
        elif df['passCrossInaccurate'].iloc[i] == True:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'cyan', alpha=0.8, lw = 1), zorder = 1)
        elif df['passThroughBallAccurate'].iloc[i] == True:
                plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'chartreuse', lw = 1.5), zorder = 2)
        elif df['passThroughBallInaccurate'].iloc[i] == True:
                plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'chartreuse', alpha=0.5,lw = 1), zorder = 2)
      
        elif df['passAccurate'].iloc[i] == True:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'white',alpha=0.6, lw = 1), zorder = 2)
                             
        else:
            plt.annotate("", xytext=(x[i],y[i]),
                         xy=(endx[i],endy[i]),
                             arrowprops=dict(arrowstyle="->", color = 'red',alpha=0.6, lw = 0.8), zorder = 2)

        plt.title('Passing Map - ' + str(text1) + '\n' + str(text2) + '\n' + str(text3) + '\nEFL Championship 2024/25', c='white', size = 18, y=0.82)
    plt.text(0.5,79, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')
    legend_labels = list(['Successful Pass - ' + str(sum(df['passAccurate'])), 'Unsuccessful Pass - ' + str(sum(df['passInaccurate'])), 'Assist - ' +str(sum(df['assist'])), 'Key Pass - ' + str(sum(df['passKey'])), 'Cross - ' + str(sum(df['passCrossAccurate'])) + ' / ' + str(sum(df['passCrossAccurate']) + sum(df['passCrossInaccurate'])), 'Through Ball - '  + str(sum(df['passThroughBallAccurate'])) + ' / ' + str(sum(df['passThroughBallAccurate']) + sum(df['passThroughBallInaccurate']))])
    colors = ['white', 'red', 'orange', 'magenta', 'cyan', 'chartreuse']

    scatter = [plt.scatter(0, -30, marker = r'$\rightarrow$', s= 100, c = colors[i],
                           label = legend_labels[i])for i in range(6)]

    ax.legend(handles = scatter,loc = (0.032,0.82), fontsize = 10)
    ax.annotate("Attacking Direction", xytext=(65,82),
                                             xy=(75,82), color = 'white',
                                             arrowprops=dict(arrowstyle="->", color = 'white', lw = 1), zorder = 4, va='center',
                                ha='right',fontsize = 12)
                             
    return fig, ax
    
def fullpitch_pass_hmap(events_df, average_values, end, scatter, bins, text1, text2, text3, matchlen, binx, biny):

    df = events_df
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    
    if bins == True:
        xedges = [0,18,39,60,81,102,120]
        yedges = [0,18,30,40,50,62,80]
    else:
        xedges = np.linspace(0,120,binx+1, endpoint=True)
        yedges = np.linspace(0,80,biny+1, endpoint=True)
    
    hrange = [[0,120],[0, 80]]
    
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'white', line_color = 'k',
                 linewidth = 2, goal_type='box', pad_right = 20, spot_scale=0.004)
    fig, ax = pitch.draw(figsize = (12,8))
    
    if end == True:
        heatmap, xedges, yedges = np.histogram2d(df['endX'],df['endY'],
                                             bins=[xedges,yedges])
        if scatter ==True:
            ax.scatter(df['endX'],df['endY'], s=1, c='k', zorder=0.2, alpha=0.5)
    else:
        heatmap, xedges, yedges = np.histogram2d(df['x'], df['y'],
                                             bins=[xedges,yedges])
        if scatter ==True:
            ax.scatter(df['x'],df['y'], s=1, c='k', zorder=0.2, alpha=0.5)
                                             
    if bins == True:
        heatmap = (heatmap/matchlen) - average_values['Average Value'].values.reshape(len(xedges)-1,len(yedges)-1)
    max = np.max((heatmap.max(),-1 * heatmap.min()))
    if bins == False:
        im = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='RdBu_r', shading='auto',edgecolors='k',
                   linewidth=0.5, zorder=0.1)
    else:
        im = ax.pcolormesh(xedges, yedges, heatmap.T, cmap='RdBu_r', shading='auto',edgecolors='k',
                   linewidth=0.5, zorder=0.1, vmin = -1 * max, vmax = max)

    colorbar = fig.colorbar(im, ax=ax, shrink = 0.8, pad = -0.1, aspect = 15)
    if bins == True:
        colorbar.set_ticks([-max/2, max/2])  # Set positions for ticks
        colorbar.set_ticklabels(['< Below average', 'Above average >'], rotation = 90, va = 'center',
                               fontsize=14, c = 'k')
        colorbar.ax.tick_params(size=0)
    
    plt.title(str(text1) + '\n' + str(text2) + '\n' + str(text3), weight = 'bold',
         fontsize = 18, y = 0.97)
    plt.text(1,78, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 10,
             c='k')
             
    ax.annotate("Attacking Direction", xytext=(65,82),
                                             xy=(75,82), color = 'k',
                                             arrowprops=dict(arrowstyle="->", color = 'k', lw = 1), zorder = 4, va='center',
                                ha='right',fontsize = 12)
    
    return fig, ax
    
def pass_in_fin3rd(events_df, matchdf, match_file, teamid, venue):

    df = events_df.loc[events_df['teamId'] == teamid]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    df = df.loc[(df['x']>=80) ]
    df = df.loc[(df['passFreekick']!=True) & (df['throwIn']!=True) & (df['passCorner']!=True)]
    
    pitch = VerticalPitch(half=True, pitch_type = 'statsbomb', pitch_color = 'navy', line_color = 'white',
                          pad_top = 15, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (12,8))
    for i in range(len(df)):
        if df['passAccurate'].iloc[i] == True:
            plt.scatter(df['y'].iloc[i],df['x'].iloc[i], s=80,c='green', ec = 'lightgreen',
                                zorder =5)
            plt.annotate("", xytext=(df['y'].iloc[i],df['x'].iloc[i]),
                             xy=(df['endY'].iloc[i],df['endX'].iloc[i]),
                             arrowprops=dict(arrowstyle="->", color = 'white', lw = 2.5), zorder = 4)
        elif df['passInaccurate'].iloc[i] == True:
            plt.annotate("", xytext=(df['y'].iloc[i],df['x'].iloc[i]),
                             xy=(df['endY'].iloc[i],df['endX'].iloc[i]),
                             arrowprops=dict(arrowstyle="->", color = 'red', lw = 1), zorder = 2)
        elif df['type'].iloc[i] == 'Carry':
            plt.scatter(df['y'].iloc[i],df['x'].iloc[i], s=80,c='blue', ec = 'lightblue',
                            zorder =5)
            plt.annotate("", xytext=(df['y'].iloc[i],df['x'].iloc[i]),
                         xy=(df['endY'].iloc[i],df['endX'].iloc[i]),
                         arrowprops=dict(arrowstyle="->", color = 'deepskyblue', lw = 2.5), zorder = 4)
    if venue == 'home':
        oppvenue = 'away'
    else:
        oppvenue = 'home'
    plt.title(str(match_file[venue].iloc[0]) + '\nvs. ' +str(match_file[oppvenue].iloc[0]) +  ' (' + str(venue).title() + ')\nOpen-Play Final 3rd Pass/Carry',
              c='white', fontsize=18, y=0.87)
    plt.text(0.5,61, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')
    legend_labels = list(['Pass', 'Carry'])
    colors = ['green', 'blue']
    edge = ('lightgreen','lightblue')

    scatter = [plt.scatter(0, -20, marker = 'o', s= 60, c = colors[i], ec = edge[i],
                           label = legend_labels[i])for i in range(2)]

    ax.legend(handles = scatter,loc = 1, fontsize = 14)
                             
    return fig, ax


def shot_assists(events_df, matchdf, match_file, teamid, venue):

    df = events_df.loc[events_df['teamId'] == teamid]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    
    df = df.loc[(df['passKey']==True) & (df['passCorner']!=True)]
    
    playerids = ast.literal_eval(matchdf[venue].iloc[0])[0]["formations"][0]["playerIds"]
    shirtnumbers =ast.literal_eval(matchdf[venue].iloc[0])[0]["formations"][0]["jerseyNumbers"]
    idnumber = dict(zip(playerids, shirtnumbers))
    
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'navy', line_color = 'white',
                          pad_top = 15, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (12,8))
    for i in range(len(df)):
        plt.scatter(df['x'].iloc[i],df['y'].iloc[i], s=250,c='green', ec = 'lightgreen',
                                zorder =5)
        plt.annotate('', xytext=(df['x'].iloc[i],df['y'].iloc[i]),
                             xy=(df['endX'].iloc[i],df['endY'].iloc[i]),
                             arrowprops=dict(arrowstyle="->", color = 'white', lw = 2.5), zorder = 4)
        plt.text(s=str(idnumber[df['playerId'].iloc[i]]), x=(df['x'].iloc[i]), y=(df['y'].iloc[i]), c='white', va='center', ha='center', zorder=5, weight = 'bold')
    if venue == 'home':
        oppvenue = 'away'
    else:
        oppvenue = 'home'
    plt.title(str(match_file[venue].iloc[0]) + '\nvs. ' +str(match_file[oppvenue].iloc[0]) +  ' (' + str(venue).title() + ')\nAll Non-Corner Shot Assists',
              c='white', fontsize=18, y=0.87)
    plt.text(0.5,79, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')

                             
    return fig, ax
    
def defensive_actions(events_df, matchdf, match_file, teamid, venue):

    df = events_df.loc[events_df['teamId'] == teamid]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    
    
