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
import xgdata as xgdata
import pickle
from mplsoccer import Standardizer
import streamlit as st
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ast


# In[13]:

def momentum(events_df, match_file):
    p_home = events_df.loc[events_df['teamId'] == match_file['hometeam.id'].iloc[0]]
    p_home = p_home.dropna(subset=['EPV'])
    
    maxtime =int(events_df['maxMinute'].iloc[0])
    epvhome = np.zeros(maxtime+1)
    for i in range(maxtime):
        dat = p_home.loc[p_home['expandedMinute'] <= i]
        dat = dat.loc[dat['expandedMinute'] >= (i-4)]['EPV']
        if len(dat) !=0:
            epvhome[i+1] = sum(dat.values)/5
            
    p_away = events_df.loc[events_df['teamId'] == match_file['awayteam.id'].iloc[0]]
    p_away = p_away.dropna(subset=['EPV'])
    epvaway = np.zeros(maxtime+1)
    for i in range(maxtime):
        dat = p_away.loc[p_away['expandedMinute'] <= i]
        dat = dat.loc[dat['expandedMinute'] >= (i-4)]['EPV']
        if len(dat) !=0:
            epvaway[i+1] = sum(dat.values)/5
        
    goals = events_df.loc[events_df['isGoal'] == 1]
    colours = dict({str(int(match_file['hometeam.id'].iloc[0])) : 'r', str(int(match_file['awayteam.id'].iloc[0])) : 'b'})
    teamids = events_df['teamId'].unique()
    epvsplit = epvhome - epvaway
    epvsplit[0] = 0
    epvsplit = epvsplit/(6*max(np.abs(epvsplit)))
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(np.array(range(maxtime+1)), epvsplit, c='grey')
    plt.fill_between(np.array(range(maxtime+1)), epvsplit, np.zeros(maxtime+1), where = epvsplit>= 0,
                     facecolor='red', interpolate=True, alpha=0.3)
    plt.fill_between(np.array(range(maxtime+1)), epvsplit, np.zeros(maxtime+1), where = epvsplit<= 0,
                     facecolor='blue', interpolate=True, alpha=0.3)
    plt.axhline(0, c='k', ls='--')
    hg = 0
    ag = 0
    if len(goals)>0:
        for n in range(len(goals)):
            if goals['goalOwn'].iloc[n] == True:
                team = goals['teamId'].iloc[n]
                if team == match_file['hometeam.id'].iloc[0]:
                    ag+=1
                elif team == match_file['awayteam.id'].iloc[0]:
                    hg+=1
                for i in teamids:
                    if i != team:
                        teamnumber = i
                cid = teamnumber
                plt.axvline(goals['expandedMinute'].iloc[n]+1, ymax=0.85, ls='--', c=colours[str(cid)])
                plt.scatter(goals['expandedMinute'].iloc[n]+1, y=0.197, s=200, ec=colours[str(cid)],
                        c='white', lw=3)
                plt.text(x=goals['expandedMinute'].iloc[n]+1, y=0.235, s=str(hg) + ' - ' + str(ag), ha='center', va='center', fontsize=10, rotation = 90)
            else:
                team = goals['teamId'].iloc[n]
                if team == match_file['hometeam.id'].iloc[0]:
                    hg+=1
                elif team == match_file['awayteam.id'].iloc[0]:
                    ag+=1
                plt.axvline(goals['expandedMinute'].iloc[n]+1, ymax=0.85, ls='--', c=colours[str(int(goals['teamId'].iloc[n]))])
                plt.scatter(goals['expandedMinute'].iloc[n]+1, y=0.197, s=200, ec=colours[str(int(goals['teamId'].iloc[n]))],
                            c='white', lw=3)
                plt.text(x=goals['expandedMinute'].iloc[n]+1, y=0.235, s=str(hg) + ' - ' + str(ag), ha='center', va='center', fontsize=10, rotation = 90)
                        
    ht = events_df.loc[events_df['period']=='FirstHalf']['minute'].max()
    plt.axvline(ht, c='grey', lw = 2, ls='--')
    plt.yticks([])
    plt.xticks(list(np.linspace(0,100,21)))
    plt.xlabel('Minute')
    plt.ylim(-0.25,0.26)
    plt.xlim(-5, maxtime+3)
    plt.title('Scaled Momentum via xT')
    plt.text(s=str(match_file['home'].iloc[0]) + r'$\rightarrow$',  x=-4, y=0.125, rotation = 90, fontsize=12, va='center')
    plt.text(s=r'$\leftarrow$' + str(match_file['away'].iloc[0]),  x=-4, y=-0.125, rotation = 90, fontsize=12, va='center')

    plt.text(70,-0.23, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
                 c='k', alpha=0.7)
    return fig, ax

def shotmaps(shots, match_file, teamid, teamname, opposition, venue):
    
    df = shots.loc[shots['teamId'] == teamid]
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
        if df['a_passCrossAccurate'].iloc[i] == True:
            if df['situation'].iloc[i] in ['SetPiece','FromCorner']:
                markers.append('D')
                sp_crosses+=1
            else:
                markers.append('s')
                op_crosses+=1
        elif df['a_passThroughBallAccurate'].iloc[i] == True:
            markers.append('^')
            throughball+=1
        elif df['a_pass'].iloc[i] == True:
            markers.append('o')
            passes+=1
        elif df['a_def'].iloc[i] == True:
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
                    s = 600, zorder=zorder[i])

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
           
    if venue == 'H':
        venue = 'At Home'
    elif venue == 'A':
        venue = 'Away'

    plt.title('Shot Map - ' + str(teamname) + ' - ' + str(venue) + '\nvs ' + str(opposition) + ' - EFL Championship 2024/25', c='white', size = 32, y=-15)

    plt.scatter(40,108,c='white',s=100)

    plt.text(0.95,-17.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta\nxG via Potterlytics',fontsize = 18,
             c='white')
             

             
    plt.text(-0.3,-17.5, s= str(int(sum(df['isGoal']))) + ' goal(s) from ' + str(np.round(sum(df['xG']),2)) + ' non-penalty xG\n'
             + str(np.round(sum(df['xG'])/len(df), 2)) + ' xG per Shot\n'
             + str(len(df)) + ' shots, '+ str(len(df.loc[df['shotOnTarget'] == True])) + ' on target\n'
             + str(len(df.loc[df['xG'] > 0.2])) + ' shots > 0.2xG'  + ' + ' + str(len(penscored)) + '/' + str(len(penscored) + len(penmissed)) + ' penalties',
             fontsize = 18, c='white')

    return fig, ax

def averagepassmaps(match_data, events_df, match_file, colour, textcolour, teamven):

    sub = events_df.loc[events_df['type']=='SubstitutionOn']
    if len(sub.loc[sub['teamId']== match_data[teamven]['teamId']]['minute']) > 0:
        subtime = sub.loc[sub['teamId']== match_data[teamven]['teamId']]['minute'].iloc[0]
    else:
        subtime = events_df['maxTime']
    
    if subtime < 45:
        subtime = 45

    fig,ax = plt.subplots(figsize=(20,33))
    plt.subplots_adjust(hspace=-0.05)
    visuals.createPassNetworks(match_data, events_df, matchId=match_file['matchId'], team=match_data[teamven]['name'], max_line_width=5, marker_size=90, edgewidth=3, dh_arrow_width=25, marker_color=colour, marker_edge_color='white', shrink=30, ax=ax, kit_no_size=18, min_time = 0, max_time = subtime, textcol = textcolour, pad_top=10)
    ax.set_title(str(match_data['home']['name']) +' '+ str(match_data['score']) +' '+ str(match_data['away']['name'])+
                '\nEFL Championship 2024/25', c='white', y=0.92, size = 18, weight='bold')
    ax.text(2,3, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 16,
                 c='white')
    ax.text(90,2, s='Plot inspired by \n@rockingAli5', fontsize=10, c='white')
    ax.text(83,102, s='Marker size indicates\nno# of pass receipts', fontsize=12, c='white', weight='bold')
    return fig, ax
    
def pass_sonars(events_df, matchdf, match_file, teamid, formation, venue):
    pitch = VerticalPitch(half = False,pitch_type = 'statsbomb', pitch_color = '#000080', line_color = 'white',pad_top=20, linewidth=1)
    positions = ast.literal_eval(matchdf[venue].iloc[0])[0]["formations"][0]["formationSlots"]
    playerids = ast.literal_eval(matchdf[venue].iloc[0])[0]["formations"][0]["playerIds"]
    shirtnumbers =ast.literal_eval(matchdf[venue].iloc[0])[0]["formations"][0]["jerseyNumbers"]
    ftt = pitch.formations_dataframe.loc[pitch.formations_dataframe['formation'] == formation]
    sbpos = []
    names = []
    for n in range(11):
        pos = positions[n]
        sbpos.append(ftt.loc[ftt['opta'] == pos]['statsbomb'].iloc[0][0])
        name = events_df.loc[events_df['playerId'] == playerids[n]]['playerName'].dropna().iloc[0]
        names.append(str(name).split(' ')[-1].replace('-', '-\n'))
        
    fig, ax = pitch.draw(figsize=(8,12))
    player_text = pitch.formation(formation, positions=sbpos, xoffset = [-6, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
                                  text=names, va='top', ha='center',
                                  fontsize=14, color='white', kind='text', ax=ax)

    axs = pitch.formation(formation, positions=sbpos,xoffset=[4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4],
                          height=15, polar=True, kind='axes',
                          ax=ax)
                          
    df_pass = events_df.loc[events_df['type'] == 'Pass']
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df_pass['x'], df_pass['y'])
    endx, endy = stand.transform(df_pass['endX'], df_pass['endY'])
    bx, by = stand.transform(df_pass['blockedX'], df_pass['blockedY'])
    df_pass['x'] = x
    df_pass['y'] = y
    df_pass['endX'] = endx
    df_pass['endY'] = endy
    df_pass['blockedX'] = bx
    df_pass['blockedY'] = by
    angle, distance = pitch.calculate_angle_and_distance(df_pass.x, df_pass.y, df_pass.endX,
                                                     df_pass.endY)
                                                     
    mask_success = df_pass['passAccurate']==True
    a=0
    for name in playerids[:11]:
        key = sbpos[a]
        playername = events_df.loc[events_df['playerId'] == name]['playerName'].iloc[0]

        mask = df_pass.playerId == name
        
        bs_count_all = pitch.bin_statistic_sonar(df_pass[mask].x, df_pass[mask].y, angle[mask],
                                                 bins=(1,1,9), center=True)
        bs_count_success = pitch.bin_statistic_sonar(df_pass[mask & mask_success].x,
                                                     df_pass[mask & mask_success].y,
                                                     angle[mask & mask_success],
                                                     bins=(1,1,9), center=True)
        bs_distance = pitch.bin_statistic_sonar(df_pass[mask].x, df_pass[mask].y, angle[mask],
                                                values=distance[mask], statistic='mean',
                                                bins=(1,1,9), center=True)
        pitch.sonar(bs_count_success, stats_color=bs_distance, vmin=0, vmax=30,
                    cmap='Blues', ec='#202020', zorder=3, ax=axs[key])

        pitch.sonar(bs_count_all, color='#f2f0f0', zorder=2, ec='#202020', ax=axs[key])
        a+=1
        
    if venue == 'home':
        oppvenue = 'away'
    else:
        oppvenue = 'home'
    plt.title(str(match_file[venue].iloc[0]) + '\nvs. ' +str(match_file[oppvenue].iloc[0]) +  ' (' + str(venue).title() + ')\nStarting XI Passing Sonars',
              c='white', fontsize=18, y=0.91)
    plt.text(0.5,0.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')
                          
    return fig, ax

def xt_map(events_df, matchdf, match_file, teamid, venue):
    
    df = events_df.loc[events_df['teamId'] == teamid]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    bx, by = stand.transform(df['blockedX'], df['blockedY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    df['blockedX'] = bx
    df['blockedY'] = by
    df = df.dropna(subset='EPV')
    df = df.loc[df['EPV']>0]
    df = df.loc[(df['throwIn'] != True) & (df['passCorner']!= True) & (df['passFreekick']!=True)]
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = '#000080', line_color = 'white', pad_top = 15, linewidth = 1,goal_type='box')
    fig, ax = pitch.draw(figsize = (12,8))

    hrange = [[0,120],[0, 80]]
    heatmap, xedges, yedges = np.histogram2d(df['x'],df['y'],weights = np.sqrt(df['EPV']), range = hrange,
                                             bins=(12,9))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # Plotting the heatmap
    ax.imshow(heatmap.T, origin='lower',cmap='jet',extent=extent, interpolation = 'bicubic')
    

    for i in range(len(df)):
        if df['type'].iloc[i] == 'Pass':
            if df['passAccurate'].iloc[i] == True:
                al = df['EPV'].iloc[i]/0.05
                if al > 1:
                    al = 1
                elif al <0:
                    al = 0
                ax.annotate("", xytext=(df['x'].iloc[i],df['y'].iloc[i]),
                         xy=(df['endX'].iloc[i],df['endY'].iloc[i]),
                         arrowprops=dict(arrowstyle="->", color = 'white', lw = df['EPV'].iloc[i] * 20, alpha = al, zorder = 4))
                         
    if venue == 'home':
        oppvenue = 'away'
    else:
        oppvenue = 'home'
    plt.title(str(match_file[venue].iloc[0]) + '\nvs. ' +str(match_file[oppvenue].iloc[0]) +  ' (' + str(venue).title() + ')\nOpen-Play xT Starting Zones',
              c='white', fontsize=18, y=0.87)
              
    plt.text(s='Arrows Show Highest xT Passes', x = 0, y=-1, c='white', weight='bold')
    plt.text(0.5,79, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')
    return fig, ax

def pass_to_fin3rd(events_df, matchdf, match_file, teamid, venue):

    df = events_df.loc[events_df['teamId'] == teamid]
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x,y = stand.transform(df['x'], df['y'])
    endx, endy = stand.transform(df['endX'], df['endY'])
    df['x'] = x
    df['y'] = y
    df['endX'] = endx
    df['endY'] = endy
    df = df.loc[(df['x']<80) & (df['endX']>=80)]
    df = df.loc[(df['passFreekick']!=True) & (df['throwIn']!=True) & (df['passCorner']!=True)]
    
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'navy', line_color = 'white',
                          pad_top = 15, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (12,8))
    for i in range(len(df)):
        if df['passAccurate'].iloc[i] == True:
            plt.scatter(df['x'].iloc[i],df['y'].iloc[i], s=80,c='green', ec = 'lightgreen',
                                zorder =5)
            plt.annotate("", xytext=(df['x'].iloc[i],df['y'].iloc[i]),
                             xy=(df['endX'].iloc[i],df['endY'].iloc[i]),
                             arrowprops=dict(arrowstyle="->", color = 'white', lw = 2.5), zorder = 4)
        elif df['type'].iloc[i] == 'Carry':
            plt.scatter(df['x'].iloc[i],df['y'].iloc[i], s=80,c='blue', ec = 'lightblue',
                            zorder =5)
            plt.annotate("", xytext=(df['x'].iloc[i],df['y'].iloc[i]),
                         xy=(df['endX'].iloc[i],df['endY'].iloc[i]),
                         arrowprops=dict(arrowstyle="->", color = 'deepskyblue', lw = 2.5), zorder = 4)
    if venue == 'home':
        oppvenue = 'away'
    else:
        oppvenue = 'home'
    plt.title(str(match_file[venue].iloc[0]) + '\nvs. ' +str(match_file[oppvenue].iloc[0]) +  ' (' + str(venue).title() + ')\nOpen-Play Final 3rd Entries',
              c='white', fontsize=18, y=0.87)
    plt.text(0.5,79, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 12,
             c='white', va='bottom')
    legend_labels = list(['Pass', 'Carry'])
    colors = ['green', 'blue']
    edge = ('lightgreen','lightblue')

    scatter = [plt.scatter(0, -20, marker = 'o', s= 60, c = colors[i], ec = edge[i],
                           label = legend_labels[i])for i in range(2)]

    ax.legend(handles = scatter,loc = 1, fontsize = 14)
                             
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
    
    
