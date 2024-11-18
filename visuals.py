# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:38:46 2020

@author: aliha
@twitter: rockingAli5 
"""

import pandas as pd
import numpy as np
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import whoscored_custom_events as wsce
import whoscored_data_engineering as wsde
pd.options.mode.chained_assignment = None
from mplsoccer.pitch import Pitch, VerticalPitch
import matplotlib as mpl
import matplotlib
import matplotlib.colors as col
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def createShotmap(match_data, events_df, team, pitchcolor, shotcolor, goalcolor, 
                  titlecolor, legendcolor, marker_size, fig, ax):
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'
        
    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']
        
    total_shots = events_df.loc[events_df['isShot']==True].reset_index(drop=True)
    team_shots = total_shots.loc[total_shots['teamId'] == teamId].reset_index(drop=True)
    mask_goal = team_shots.isGoal == True

    # Setup the pitch
    # orientation='vertical'
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=pitchcolor, line_color='#c7d5cc',
                          half=True, pad_top=2)
    pitch.draw(ax=ax, tight_layout=True, constrained_layout=True)


    # Plot the goals
    pitch.scatter(team_shots[mask_goal].x/100*120, 80-team_shots[mask_goal].y/100*80, s=marker_size,
                  edgecolors='black', c=goalcolor, zorder=2,
                  label='goal', ax=ax)
    pitch.scatter(team_shots[~mask_goal].x/100*120, 80-team_shots[~mask_goal].y/100*80,
                  edgecolors='white', c=shotcolor, s=marker_size, zorder=2,
                  label='shot', ax=ax)
    # Set the title
    ax.set_title(f'{team} shotmap \n vs {opponent}', fontsize=30, color=titlecolor)

    # set legend
    leg = ax.legend(facecolor=pitchcolor, edgecolor='None', fontsize=20, loc='lower center', handlelength=4)
    leg_texts = leg.get_texts() # list of matplotlib Text instances.
    leg_texts[0].set_color(legendcolor)
    leg_texts[1].set_color(legendcolor)
    
    # Set the figure facecolor
    fig.set_facecolor(pitchcolor)
    
    
    
    


def createPassNetworks(match_data, events_df, matchId, team, max_line_width, 
                       marker_size, edgewidth, dh_arrow_width, marker_color,
                       marker_edge_color, shrink, ax,textcol = 'white', kit_no_size=20, min_time = 0, max_time = 0, pad_top=0):
    
    if max_time == 0:
        max_time = 150
        mt_text = np.max(events_df['expandedMinute'])+0.0
    else:
        max_time = max_time
        mt_text = max_time + 0.0
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'
    
    
    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']
    
    
    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']
    
    
    # getting minute of first substitution
    for i in events_df.index:
        if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId:
            sub_minute = str(events_df.loc[i, 'minute'])
            print(sub_minute)
            break
    
    
    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number
    
    
    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
    passes_df.dropna(subset=["playerId"], inplace=True)
    passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[passes_df['type'] == 'Pass', :].reset_index(drop=True)
    passes_df = passes_df.loc[passes_df['outcomeType'] == 'Successful', :].reset_index(drop=True)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])
#    passes_df = passes_df[passes_df['playerPos'] != 'Sub']
    passes_df = passes_df.loc[passes_df['expandedMinute'] >= min_time]
    passes_df = passes_df.loc[passes_df['expandedMinute'] <= max_time]
    
    # getting team formation
    formation = match_data[venue]['formations'][0]['formationName']
    formation = '-'.join(formation)
    
    
    # getting player average locations
    location_formation = passes_df[['playerKitNumber', 'x', 'y']]
    average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count']})
    average_locs_and_count.columns = ['x', 'y', 'count']

    
    # getting separate dataframe for selected columns 
    passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()
    passes_formation['EPV'] = passes_df['EPV']

    
    # getting dataframe for passes between players
    passes_between = passes_formation.groupby(['playerKitNumber', 'playerKitNumberReceipt']).agg({ 'id' : 'count', 'EPV' : 'sum'}).reset_index()
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumberReceipt', right_index=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumber', right_index=True,
                                          suffixes=['', '_end'])
    
    
    # filtering passes
    pass_filter = int(passes_between['pass_count'].mean())
    passes_between = passes_between.loc[passes_between['pass_count'] > pass_filter]
    
    
    # calculating the line width 
    passes_between['width'] = passes_between.pass_count / passes_between.pass_count.max() * max_line_width
    passes_between = passes_between.reset_index(drop=True)
    
    
    # setting color to make the lines more transparent when fewer passes are made
    min_transparency = 0.3
    color = np.array(to_rgba('white'))
    color = np.tile(color, (len(passes_between), 1))
    c_transparency = passes_between.pass_count / passes_between.pass_count.max()
    c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
    color[:, 3] = c_transparency
    passes_between['alpha'] = color.tolist()

    
    # separating paired passes from normal passes
    passes_between_threshold = 15
    filtered_pair_df = []
    pair_list = [comb for comb in combinations(passes_between['playerKitNumber'].unique(), 2)]
    for pair in pair_list:
        df = passes_between[((passes_between['playerKitNumber']==pair[0]) & (passes_between['playerKitNumberReceipt']==pair[1])) | 
                            ((passes_between['playerKitNumber']==pair[1]) & (passes_between['playerKitNumberReceipt']==pair[0]))]
        if df.shape[0] == 2:
            if (np.array(df.pass_count)[0] >= passes_between_threshold) and (np.array(df.pass_count)[1] >= passes_between_threshold):
                filtered_pair_df.append(df)
                passes_between.drop(df.index, inplace=True)
    if len(filtered_pair_df) > 0:
        filtered_pair_df = pd.concat(filtered_pair_df).reset_index(drop=True)
        passes_between = passes_between.reset_index(drop=True)
    
    
    # plotting
    pitch = Pitch(pitch_type='opta', pitch_color='navy', line_color='white', linewidth = 1, goal_type='box', pad_top = pad_top)
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    average_locs_and_count['zorder'] = list(np.linspace(1,5,len(average_locs_and_count)))
    for i in average_locs_and_count.index:
        pitch.scatter(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y'], s=marker_size * average_locs_and_count.loc[i, 'count'],
                      color=marker_color, edgecolors=marker_edge_color, linewidth=edgewidth, 
                      alpha=1, zorder=average_locs_and_count.loc[i, 'zorder'], ax=ax)
    
    for i in passes_between.index:
        x = passes_between.loc[i, 'x']
        y = passes_between.loc[i, 'y']
        endX = passes_between.loc[i, 'x_end']
        endY = passes_between.loc[i, 'y_end']
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch([endX, endY], [x, y],
                              coordsA, coordsB,
                              arrowstyle="simple", shrinkA=shrink, shrinkB=shrink,
                              mutation_scale=passes_between.loc[i, 'width']*max_line_width, color=passes_between.loc[i, 'alpha'])
        ax.add_artist(con)
    
    if len(filtered_pair_df) > 0:
        for i in filtered_pair_df.index:
            x = filtered_pair_df.loc[i, 'x']
            y = filtered_pair_df.loc[i, 'y']
            endX = filtered_pair_df.loc[i, 'x_end']
            endY = filtered_pair_df.loc[i, 'y_end']
            coordsA = "data"
            coordsB = "data"
            con = ConnectionPatch([endX, endY], [x, y],
                                  coordsA, coordsB,
                                  arrowstyle="<|-|>", shrinkA=shrink, shrinkB=shrink,
                                  mutation_scale=dh_arrow_width, lw=filtered_pair_df.loc[i, 'width']*max_line_width/5, 
                                  color=filtered_pair_df.loc[i, 'alpha'])
            ax.add_artist(con)
    
    for i in average_locs_and_count.index:
        if marker_size * average_locs_and_count.loc[i, 'count'] >= 900:
                    pitch.annotate(i, xy=(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y']),
                       family='DejaVu Sans', c=textcol,
                       va='center', ha='center', zorder=average_locs_and_count.loc[i, 'zorder'], size=kit_no_size, weight='bold', ax=ax)
        else:
            pitch.annotate(i, xy=(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y']+0.2*average_locs_and_count.loc[i,'count'] + 2),
                       family='DejaVu Sans', c='white',
                       va='center', ha='center', zorder=average_locs_and_count.loc[i, 'zorder'], size=kit_no_size, weight='bold', ax=ax)
    ax.text(50, 98, "{} (Mins {}-{})".format(team, min_time, str(np.round(mt_text,0))[:-2]).upper(), size=10, fontweight='bold', ha='center', c='white',
           va='center', bbox = dict(fc='white', ec='k', alpha=0.4))

#    ax.text(2, 3, '{}'.format(formation), size=9, c='grey')

    
    
    
    
    
def createAttPassNetworks(match_data, events_df, matchId, team, max_line_width, 
                      marker_size, edgewidth, dh_arrow_width, marker_color, 
                      marker_edge_color, shrink, ax, kit_no_size = 20):
    
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'
    
    
    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']
    
    
    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']
    
    
    # getting minute of first substitution
    for i in events_df.index:
        if events_df.loc[i, 'type'] == 'SubstitutionOn' and events_df.loc[i, 'teamId'] == teamId:
            sub_minute = str(events_df.loc[i, 'minute'])
            break
    
    
    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []


    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number
    
    
    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
    passes_df.dropna(subset=["playerId"], inplace=True)
    passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[events_df['type'] == 'Pass', :].reset_index(drop=True)
    passes_df = passes_df.loc[events_df['outcomeType'] == 'Successful', :].reset_index(drop=True)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])
    passes_df = passes_df[passes_df['playerPos'] != 'Sub']
    
    
    # getting team formation
    formation = match_data[venue]['formations'][0]['formationName']
    formation = '-'.join(formation)
    
    
    # getting player average locations
    location_formation = passes_df[['playerKitNumber', 'x', 'y']]
    average_locs_and_count = location_formation.groupby('playerKitNumber').agg({'x': ['mean'], 'y': ['mean', 'count']})
    average_locs_and_count.columns = ['x', 'y', 'count']
    
    
    # filtering progressive passes 
    passes_df = passes_df.loc[passes_df['EPV'] > 0]

    
    # getting separate dataframe for selected columns 
    passes_formation = passes_df[['id', 'playerKitNumber', 'playerKitNumberReceipt']].copy()
    passes_formation['EPV'] = passes_df['EPV']


    # getting dataframe for passes between players
    passes_between = passes_formation.groupby(['playerKitNumber', 'playerKitNumberReceipt']).agg({ 'id' : 'count', 'EPV' : 'sum'}).reset_index()
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumberReceipt', right_index=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='playerKitNumber', right_index=True,
                                          suffixes=['', '_end'])
    
    
    # filtering passes
    pass_filter = int(passes_between['pass_count'].mean())
    passes_between = passes_between.loc[passes_between['pass_count'] > pass_filter*2]
    
    
    # calculating the line width and marker sizes relative to the largest counts
    passes_between['width'] = passes_between.pass_count / passes_between.pass_count.max() * max_line_width
    passes_between = passes_between.reset_index(drop=True)
    
    
    # setting color to make the lines more transparent when fewer passes are made
    min_transparency = 0.3
    color = np.array(to_rgba('white'))
    color = np.tile(color, (len(passes_between), 1))
    c_transparency = passes_between.EPV / passes_between.EPV.max()
    c_transparency = (c_transparency * (1 - min_transparency)) + min_transparency
    color[:, 3] = c_transparency
    passes_between['alpha'] = color.tolist()
    
    
    # separating paired passes from normal passes
    passes_between_threshold = 20
    filtered_pair_df = []
    pair_list = [comb for comb in combinations(passes_between['playerKitNumber'].unique(), 2)]
    for pair in pair_list:
        df = passes_between[((passes_between['playerKitNumber']==pair[0]) & (passes_between['playerKitNumberReceipt']==pair[1])) | 
                            ((passes_between['playerKitNumber']==pair[1]) & (passes_between['playerKitNumberReceipt']==pair[0]))]
        if df.shape[0] == 2:
            if np.array(df.pass_count)[0]+np.array(df.pass_count)[1] >= passes_between_threshold:
                filtered_pair_df.append(df)
                passes_between.drop(df.index, inplace=True)
    if len(filtered_pair_df) > 0:
        filtered_pair_df = pd.concat(filtered_pair_df).reset_index(drop=True)
        passes_between = passes_between.reset_index(drop=True)
    
    
    # plotting
    pitch = Pitch(pitch_type='opta', pitch_color='#171717', line_color='#5c5c5c',
                  goal_type='box')
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    
    average_locs_and_count['zorder'] = list(np.linspace(1,5,11))
    for i in average_locs_and_count.index:
        pitch.scatter(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y'], s=marker_size,
                      color=marker_color, edgecolors=marker_edge_color, linewidth=edgewidth, 
                      alpha=1, zorder=average_locs_and_count.loc[i, 'zorder'], ax=ax)
    
    for i in passes_between.index:
        x = passes_between.loc[i, 'x']
        y = passes_between.loc[i, 'y']
        endX = passes_between.loc[i, 'x_end']
        endY = passes_between.loc[i, 'y_end']
        coordsA = "data"
        coordsB = "data"
        con = ConnectionPatch([endX, endY], [x, y],
                              coordsA, coordsB,
                              arrowstyle="simple", shrinkA=shrink, shrinkB=shrink,
                              mutation_scale=passes_between.loc[i, 'width']*max_line_width, color=passes_between.loc[i, 'alpha'])
        ax.add_artist(con)
    
    if len(filtered_pair_df) > 0:
        for i in filtered_pair_df.index:
            x = filtered_pair_df.loc[i, 'x']
            y = filtered_pair_df.loc[i, 'y']
            endX = filtered_pair_df.loc[i, 'x_end']
            endY = filtered_pair_df.loc[i, 'y_end']
            coordsA = "data"
            coordsB = "data"
            con = ConnectionPatch([endX, endY], [x, y],
                                  coordsA, coordsB,
                                  arrowstyle="<|-|>", shrinkA=shrink, shrinkB=shrink,
                                  mutation_scale=dh_arrow_width, lw=filtered_pair_df.loc[i, 'width']*max_line_width/5, 
                                  color=filtered_pair_df.loc[i, 'alpha'])
            ax.add_artist(con)
    
    for i in average_locs_and_count.index:
        pitch.annotate(i, xy=(average_locs_and_count.loc[i, 'x'], average_locs_and_count.loc[i, 'y']), 
                       family='DejaVu Sans', c='white', 
                       va='center', ha='center', zorder=average_locs_and_count.loc[i, 'zorder'], size=kit_no_size, weight='bold', ax=ax)
    ax.text(50, 104, "{} (Mins 1-{})".format(team, sub_minute).upper(), size=10, fontweight='bold', ha='center',
           va='center')
    ax.text(2, 3, '{}'.format(formation), size=9, c='grey')

    
    






def getTeamSuccessfulBoxPasses(events_df, teamId, team, pitch_color, cmap):
    """
    Parameters
    ----------
    events_df : DataFrame of all events.
    
    teamId : ID of the team, the passes of which are required.
    
    team : Name of the team, the passes of which are required.
    
    pitch_color : color of the pitch.
    
    cmap : color design of the pass lines. 
           You can select more cmaps here: 
               https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    Returns
    -------
    Pitch Plot.

    """
    
    # Get Total Passes
    passes_df = events_df.loc[events_df['type']=='Pass'].reset_index(drop=True)
    
    # Get Team Passes
    team_passes = passes_df.loc[passes_df['teamId'] == teamId]
        
    # Extracting Box Passes from Total Passes
    box_passes = team_passes.copy()
    for i,pas in box_passes.iterrows():
        X = pas["x"]/100*120
        Xend = pas["endX"]/100*120
        Y = pas["y"]/100*80
        Yend = pas["endY"]/100*80
        if Xend >= 102 and Yend >= 18 and Yend <= 62:
            if X >=102 and Y >= 18 and Y <= 62:
                box_passes = box_passes.drop([i])
            else:
                pass
        else:
            box_passes = box_passes.drop([i])
            
    
    successful_box_passes = box_passes.loc[box_passes['outcomeType']=='Successful'].reset_index(drop=True)
    
        
    # orientation='vertical'
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color='#c7d5cc',
                          figsize=(16, 11), half=True, pad_top=2)
    fig, ax = pitch.draw(tight_layout=True)
    
    # Plot the completed passes
    pitch.lines(successful_box_passes.x/100*120, 80-successful_box_passes.y/100*80,
                successful_box_passes.endX/100*120, 80-successful_box_passes.endY/100*80,
                lw=5, cmap=cmap, opp_comet=True, opp_transparent=True,
                label='Successful Passes', ax=ax)
    
    pitch.scatter(successful_box_passes.x/100*120, 80-successful_box_passes.y/100*80,
                  edgecolors='white', c='white', s=50, zorder=2,
                  ax=ax)
    
    # Set the title
    fig.suptitle(f'Completed Box Passes - {team}', y=.95, fontsize=15)
    
    # Set the subtitle
    ax.set_title('Data : Whoscored/Opta', fontsize=8, loc='right', fontstyle='italic', fontweight='bold')
    
    # set legend
    #ax.legend(facecolor='#22312b', edgecolor='None', fontsize=8, loc='lower center', handlelength=4)
    
    # Set the figure facecolor
    fig.set_facecolor(pitch_color) 








def getTeamTotalPasses(events_df, teamId, team, opponent, pitch_color):
    """
    

    Parameters
    ----------
    events_df : DataFrame of all events.
    
    teamId : ID of the team, the passes of which are required.
    
    team : Name of the team, the passes of which are required.
    
    opponent : Name of opponent team.
    
    pitch_color : color of the pitch.


    Returns
    -------
    Pitch Plot.
    """
    
    # Get Total Passes
    passes_df = events_df.loc[events_df['type']=='Pass'].reset_index(drop=True)
    
    # Get Team Passes
    team_passes = passes_df.loc[passes_df['teamId'] == teamId]
        
    successful_passes = team_passes.loc[team_passes['outcomeType']=='Successful'].reset_index(drop=True)
    unsuccessful_passes = team_passes.loc[team_passes['outcomeType']=='Unsuccessful'].reset_index(drop=True)
            
    # Setup the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color=pitch_color, line_color='#c7d5cc')
    fig, ax = pitch.draw(constrained_layout=True, tight_layout=False)
    
    # Plot the completed passes
    pitch.arrows(successful_passes.x/100*120, 80-successful_passes.y/100*80,
                 successful_passes.endX/100*120, 80-successful_passes.endY/100*80, width=1,
                 headwidth=10, headlength=10, color='#ad993c', ax=ax, label='Completed')
    
    # Plot the other passes
    pitch.arrows(unsuccessful_passes.x/100*120, 80-unsuccessful_passes.y/100*80,
                 unsuccessful_passes.endX/100*120, 80-unsuccessful_passes.endY/100*80, width=1,
                 headwidth=6, headlength=5, headaxislength=12, color='#ba4f45', ax=ax, label='Blocked')
    
    # setup the legend
    ax.legend(facecolor=pitch_color, handlelength=5, edgecolor='None', fontsize=8, loc='upper left', shadow=True)
    
    # Set the title
    fig.suptitle(f'{team} Passes vs {opponent}', y=1, fontsize=15)
    
    
    # Set the subtitle
    ax.set_title('Data : Whoscored/Opta', fontsize=8, loc='right', fontstyle='italic', fontweight='bold')
    
    
    # Set the figure facecolor
    
    fig.set_facecolor(pitch_color)
    
    
    
    
    

def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] 
            - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]




    
def createPVFormationMap(match_data, events_df, team, color_palette,
                        markerstyle, markersize, markeredgewidth, labelsize, labelcolor, ax):
    
    # getting team id and venue
    if match_data['home']['name'] == team:
        teamId = match_data['home']['teamId']
        venue = 'home'
    else:
        teamId = match_data['away']['teamId']
        venue = 'away'


    # getting opponent   
    if venue == 'home':
        opponent = match_data['away']['name']
    else:
        opponent = match_data['home']['name']


    # getting player dictionary
    team_players_dict = {}
    for player in match_data[venue]['players']:
        team_players_dict[player['playerId']] = player['name']


    # getting minute of first substitution
    for i,row in events_df.iterrows():
        if row['type'] == 'SubstitutionOn' and row['teamId'] == teamId:
            sub_minute = str(row['minute'])
            break


    # getting players dataframe
    match_players_df = pd.DataFrame()
    player_names = []
    player_ids = []
    player_pos = []
    player_kit_number = []

    for player in match_data[venue]['players']:
        player_names.append(player['name'])
        player_ids.append(player['playerId'])
        player_pos.append(player['position'])
        player_kit_number.append(player['shirtNo'])

    match_players_df['playerId'] = player_ids
    match_players_df['playerName'] = player_names
    match_players_df['playerPos'] = player_pos
    match_players_df['playerKitNumber'] = player_kit_number


    # extracting passes
    passes_df = events_df.loc[events_df['teamId'] == teamId].reset_index().drop('index', axis=1)
    passes_df['playerId'] = passes_df['playerId'].astype('float').astype('Int64')
    if 'playerName' in passes_df.columns:
        passes_df = passes_df.drop(columns='playerName')
    passes_df.dropna(subset=["playerId"], inplace=True)
    passes_df.insert(27, column='playerName', value=[team_players_dict[i] for i in list(passes_df['playerId'])])
    if 'passRecipientId' in passes_df.columns:
        passes_df = passes_df.drop(columns='passRecipientId')
        passes_df = passes_df.drop(columns='passRecipientName')
    passes_df.insert(28, column='passRecipientId', value=passes_df['playerId'].shift(-1))  
    passes_df.insert(29, column='passRecipientName', value=passes_df['playerName'].shift(-1))  
    passes_df.dropna(subset=["passRecipientName"], inplace=True)
    passes_df = passes_df.loc[events_df['type'] == 'Pass', :].reset_index(drop=True)
    passes_df = passes_df.loc[events_df['outcomeType'] == 'Successful', :].reset_index(drop=True)
    index_names = passes_df.loc[passes_df['playerName']==passes_df['passRecipientName']].index
    passes_df.drop(index_names, inplace=True)
    passes_df = passes_df.merge(match_players_df, on=['playerId', 'playerName'], how='left', validate='m:1')
    passes_df = passes_df.merge(match_players_df.rename({'playerId': 'passRecipientId', 'playerName':'passRecipientName'},
                                                        axis='columns'), on=['passRecipientId', 'passRecipientName'],
                                                        how='left', validate='m:1', suffixes=['', 'Receipt'])
    # passes_df = passes_df[passes_df['playerPos'] != 'Sub']
    
    
    # Getting net possesion value for passes
    netPVPassed = passes_df.groupby(['playerId', 'playerName'])['EPV'].sum().reset_index()
    netPVReceived = passes_df.groupby(['passRecipientId', 'passRecipientName'])['EPV'].sum().reset_index()
    

    
    # Getting formation and player ids for first 11
    formation = match_data[venue]['formations'][0]['formationName']
    formation_positions = match_data[venue]['formations'][0]['formationPositions']
    playerIds = match_data[venue]['formations'][0]['playerIds'][:11]

    
    # Getting all data in a dataframe
    formation_data = []
    for playerId, pos in zip(playerIds, formation_positions):
        pl_dict = {'playerId': playerId}
        pl_dict.update(pos)
        formation_data.append(pl_dict)
    formation_data = pd.DataFrame(formation_data)
    formation_data['vertical'] = normalize(formation_data['vertical'], 
                                           {'actual': {'lower': 0, 'upper': 10}, 'desired': {'lower': 10, 'upper': 110}})
    formation_data['horizontal'] = normalize(formation_data['horizontal'],
                                             {'actual': {'lower': 0, 'upper': 10}, 'desired': {'lower': 80, 'upper': 0}})
    formation_data = netPVPassed.join(formation_data.set_index('playerId'), on='playerId', how='inner').reset_index(drop=True)
    formation_data = formation_data.rename(columns={"EPV": "PV"})


    # Plotting
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#171717', line_color='#5c5c5c',
                  goal_type='box')
    pitch.draw(ax=ax, constrained_layout=True, tight_layout=True)
    
    sns.scatterplot(x='vertical', y='horizontal', data=formation_data, hue='PV', s=markersize, marker=markerstyle, legend=False, 
                    palette=color_palette, linewidth=markeredgewidth, ax=ax)
    
    ax.text(2, 78, '{}'.format('-'.join(formation)), size=20, c='grey')
    
    for index, row in formation_data.iterrows():
        pitch.annotate(str(round(row.PV*100,2))+'%', xy=(row.vertical, row.horizontal), c=labelcolor, va='center',
                       ha='center', size=labelsize, zorder=2, weight='bold', ax=ax)
        pitch.annotate(row.playerName, xy=(row.vertical, row.horizontal+5), c=labelcolor, va='center',
                       ha='center', size=labelsize, zorder=2, weight='bold', ax=ax)
        
def player_shotmap(data, league, season, team, player, mins, size=400, save= True):

    from mplsoccer import Standardizer
    df = data.loc[data['playerName'] == player]
    penscored = df.loc[df['penaltyScored'] == True]
    penmissed = df.loc[df['penaltyMissed'] == True]
    df = df.loc[df['penaltyScored'] == False]
    df = df.loc[df['penaltyMissed'] == False]
    
    stand = Standardizer(pitch_from = 'opta', pitch_to = 'statsbomb')
    x, y  = stand.transform(df['x'],df['y'])
    df['x'] = x
    df['y'] = y

    playername = str.split(player)[1]
    
    markers = []
    ec = []
    lw = []
    for i in range(len(df['x'])):
        if df['isGoal'].iloc[i] == True:
            ec.append('red')
            lw.append(4)
        elif df['shotBlocked'].iloc[i] == True:
            ec.append('orange')
            lw.append(2)
        elif df['type'].iloc[i] == 'SavedShot':
            ec.append('blue')
            lw.append(3)
        elif df['shotOffTarget'].iloc[i] == True:
            ec.append('cyan')
            lw.append(2)
        if df['situation'].iloc[i] == 'DirectFreekick':
            markers.append('D')
        elif df['shotHead'].iloc[i] == True:
            markers.append('^')
        else:
            markers.append('o')


    norm = plt.Normalize(0,1)
    colmap = col.LinearSegmentedColormap.from_list("",["white",'orange','red',"darkred"])
    rescale = lambda y: (y - 0) / (1)
    colour = colmap(rescale(df['xG']))
    
    pitch = VerticalPitch(half = True,pitch_type = 'statsbomb',
                      pitch_color = '#101010', line_color = 'white', pad_top = 12)

    fig, ax = pitch.draw(figsize = (16,16))

    inarea = df.loc[df['x'] >= 102]
    inarea = inarea.loc[inarea['y'] >= 18]
    inarea = inarea.loc[inarea['y'] <= 62]


    for i in range(len(df['x'])):
        plt.scatter(df['y'].iloc[i],df['x'].iloc[i],
                    fc = colour[i], ec = ec[i],
                    linewidth = lw[i], marker=markers[i], alpha = 0.9,
                    s = size, zorder=2)

    legend_labels = list(['Goal', 'Saved', 'Blocked', 'Off Target'])
    colors = ['red', 'blue','orange','cyan']
    edge = (3,2,1,1)

    for i in range(4):
        plt.scatter(0, -10, marker = 'o', s= 300, c = 'white', ec = colors[i], lw = edge[i],
                    label = legend_labels[i])

    legend = plt.legend(loc =2, fontsize = 20)

    plt.gca().add_artist(legend)

    legend_labels = list(['Foot/Other', 'Head', 'Free Kick'])
    colors = ['k', 'k','k']
    edge = (1,1,1)
    mark = ['o', '^', 'D']

    scatter = [plt.scatter(0, -10, marker = mark[i], s= 300, c = 'white', ec = colors[i], lw = edge[i],
                           label = legend_labels[i])for i in range(3)]

    ax.legend(handles = scatter,loc = 1, fontsize = 20)

    axins = inset_axes(ax,
                        width="50%",
                        height='5%',
                        loc='upper center',
                        )
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colmap),
                 ax=ax, cax = axins, orientation = 'horizontal')

    cbar.set_label('xG', size=24, color = 'white')

    cbar.ax.tick_params(labelsize=21, color = 'white', labelcolor = 'white')

    ax.scatter(12, np.mean(df['x']), s=300, c='white', ec='white')


    ax.plot((12,12), (120,np.mean(df['x'])), c='white', lw=2)


    ax.text(s='Average \nShot Distance\n' +str(np.round(np.mean(120-df['x']),1)) + 'm',
           x =9.2, y= np.mean(df['x'])+1.5, c='white', rotation = 90, size = 16)

    plt.title('Shot Map - ' + str(player) + '\n' + str(team) + ' - ' + str(mins) + ' Mins Played\n' + str(league) + ' - ' + str(season), c='white', size = 32, y=-15)

    ax.scatter(40,108,c='white',s=100)

    plt.text(0.95,-17.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta\nxG via Potterlytics',fontsize = 18,
             c='white')
    plt.text(-0.3,-17.5, s= str(int(sum(df['isGoal']))) + ' goals from ' + str(np.round(sum(df['xG']),2)) + ' non-penalty xG\n'
             + str(np.round(sum(df['xG'])/len(df), 2)) + ' xG per Shot\n'
             + str(len(df)) + ' shots, '+ str(len(df.loc[df['shotOnTarget'] == True])) + ' on target\n'
             + str(len(inarea)) + ' inside area'  + ' + ' + str(len(penscored)) + '/' + str(len(penscored) + len(penmissed)) + ' penalties',
             fontsize = 18, c='white')

    if save == True:
        plt.savefig(str(playername) + 'shotmap.png', bbox_inches = 'tight', dpi = 500)
        


def shotmap(data, league, season, team,teamid=96, size=400,save= True):

    from mplsoccer import Standardizer
    if team == 'Stoke City':
        df = data.loc[data['teamId'] == 96]
    else:
        df = data.loc[data['teamId'] != 96]
    if teamid != 96:
        df = data.loc[data['teamId'] == teamid]
    
    

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
    for i in range(len(df['x'])):
        if df['isGoal'].iloc[i] == True:
            ec.append('red')
            lw.append(4)
        elif df['shotBlocked'].iloc[i] == True:
            ec.append('orange')
            lw.append(2)
        elif df['type'].iloc[i] == 'SavedShot':
            ec.append('blue')
            lw.append(3)
        elif df['shotOffTarget'].iloc[i] == True:
            ec.append('cyan')
            lw.append(2)
        if df['situation'].iloc[i] == 'DirectFreekick':
            markers.append('D')
        elif df['shotHead'].iloc[i] == True:
            markers.append('^')
        else:
            markers.append('o')


    norm = plt.Normalize(0,1)
    colmap = col.LinearSegmentedColormap.from_list("",["white",'orange','red',"darkred"])
    rescale = lambda y: (y - 0) / (1)
    colour = colmap(rescale(df['xG']))
    
    pitch = VerticalPitch(half = True,pitch_type = 'statsbomb',
                      pitch_color = '#101010', line_color = 'white', pad_top = 12)

    fig, ax = pitch.draw(figsize = (16,16))

    inarea = df.loc[df['x'] >= 102]
    inarea = inarea.loc[inarea['y'] >= 18]
    inarea = inarea.loc[inarea['y'] <= 62]


    for i in range(len(df['x'])):
        plt.scatter(df['y'].iloc[i],df['x'].iloc[i],
                    fc = colour[i], ec = ec[i],
                    linewidth = lw[i], marker=markers[i], alpha = 0.9,
                    s = size, zorder=2)

    legend_labels = list(['Goal', 'Saved', 'Blocked', 'Off Target'])
    colors = ['red', 'blue','orange','cyan']
    edge = (3,2,1,1)

    for i in range(4):
        plt.scatter(0, -10, marker = 'o', s= 300, c = 'white', ec = colors[i], lw = edge[i],
                    label = legend_labels[i])

    legend = plt.legend(loc =2, fontsize = 20)

    plt.gca().add_artist(legend)

    legend_labels = list(['Foot/Other', 'Head', 'Free Kick'])
    colors = ['k', 'k','k']
    edge = (1,1,1)
    mark = ['o', '^', 'D']

    scatter = [plt.scatter(0, -10, marker = mark[i], s= 300, c = 'white', ec = colors[i], lw = edge[i],
                           label = legend_labels[i])for i in range(3)]

    ax.legend(handles = scatter,loc = 1, fontsize = 20)

    axins = inset_axes(ax,
                        width="50%",
                        height='5%',
                        loc='upper center',
                        )
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colmap),
                 ax=ax, cax = axins, orientation = 'horizontal')

    cbar.set_label('xG', size=24, color = 'white')

    cbar.ax.tick_params(labelsize=21, color = 'white', labelcolor = 'white')

    ax.scatter(12, np.mean(df['x']), s=300, c='white', ec='white')


    ax.plot((12,12), (120,np.mean(df['x'])), c='white', lw=2)


    ax.text(s='Average \nShot Distance\n' +str(np.round(np.mean(120-df['x']),1)) + 'm',
           x =9.2, y= np.mean(df['x'])+1.5, c='white', rotation = 90, size = 16)

    plt.title('Shot Map - ' + str(team) + '\n' + str(league) + ' - ' + str(season), c='white', size = 32, y=-15)

    plt.scatter(40,108,c='white',s=100)

    plt.text(0.95,-17.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta\nxG via Potterlytics',fontsize = 18,
             c='white')
    plt.text(-0.3,-17.5, s= str(int(sum(df['isGoal']))) + ' goals from ' + str(np.round(sum(df['xG']),2)) + ' non-penalty xG\n'
             + str(np.round(sum(df['xG'])/len(df), 2)) + ' xG per Shot\n'
             + str(len(df)) + ' shots, '+ str(len(df.loc[df['shotOnTarget'] == True])) + ' on target\n'
             + str(len(inarea)) + ' inside area'  + ' + ' + str(len(penscored)) + '/' + str(len(penscored) + len(penmissed)) + ' penalties',
             fontsize = 18, c='white')

    if save == True:
        plt.savefig('data/stoke23_24/shotmap.png', bbox_inches = 'tight', dpi = 500)


def heatmap(data, player, title, league, season, team,bins = (60,40), start=True, save = False):
    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'black', line_color = 'white',
                              pad_top = 8, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (16,12))
    
    if start == True:
        x = 120*data['x']/100
        y = 80*(100-data['y'])/100
        
    else:
        x = 120*data['endX']/100
        y = 80*(100-data['endY'])/100
        
    

    hrange = [[0, 120],[0,80]]
    heatmap, xedges, yedges = np.histogram2d(x,y,
                                             range = hrange,
                                             bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # Plotting the heatmap
    ax.imshow(heatmap.T, origin='lower', cmap='hot',extent=extent,  interpolation='bicubic')
    if np.median(y) <= 40:
        plt.plot((np.median(x),np.median(x)),
                 (np.median(y),0),
                 lw=2,marker = '',c='white')
    else:
        plt.plot((np.median(x),np.median(x)),
                 (np.median(y),80),
                 lw=2,marker = '',c='white')
    plt.scatter(np.median(x),
             np.median(y), c='white',s=100)


    plt.plot((120,115),(-6,-6),
            lw=2,marker = '',c='white')
    plt.scatter(120,-6, c='white',s=100)
    plt.text(115,-1.5, s= 'Average \nposition',fontsize = 16,
                 c='white')
    plt.title(str(player) + ' - ' + str(title) + '\n' + str(league) + ' ' + str(season) + ' - ' + str(team),
              y=0.925,c='white',fontsize=20)
    plt.text(0.95,28, s= '@Potterlytics\npotterlytics.blog\nData via Opta',fontsize = 16,
                 c='white')
    if save == True:
        plt.savefig(str(playername) + 'heatmap.png', bbox_inches = 'tight', dpi = 500)
        
def arrowmap(data, name=None):
    if name == None:
        carry = data
    else:
        name = name
        carry= data.loc[data['playerId'] == player_ids[name]]


    pitch = Pitch(pitch_type = 'statsbomb', pitch_color = 'black', line_color = 'white',
                          pad_top = 10, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (16,12))
    for i in range(len(carry)):
        if carry['outcomeType'].iloc[i] == 'Successful':
            plt.scatter(120 *carry['x'].iloc[i]/100,80* (100-carry['y'].iloc[i])/100, s=40,c='green', ec = 'lightgreen',
                        zorder =5)
            plt.annotate("", xytext=((120 *carry['x'].iloc[i]/100),80* (100-carry['y'].iloc[i])/100),
                     xy=((120 *carry['endX'].iloc[i]/100),80 * (100-carry['endY'].iloc[i])/100),
                     arrowprops=dict(arrowstyle="->", color = 'white', lw = 1), zorder = 5)
        else:
            plt.scatter(120 *carry['x'].iloc[i]/100,80* (100-carry['y'].iloc[i])/100, s=40,c='grey', ec = 'lightgrey',
                        alpha = 0.5, zorder =5)
            plt.annotate("", xytext=((120 *carry['x'].iloc[i]/100),80* (100-carry['y'].iloc[i])/100),
                     xy=((120 *carry['endX'].iloc[i]/100),80 * (100-carry['endY'].iloc[i])/100),
                     arrowprops=dict(arrowstyle="->", color = 'lightgrey', lw = 0.8), zorder = 5)
                     
    legend_labels = list(['Successful', 'Unsuccessful'])
    colors = ['green', 'grey']
    edge = ('lightgreen','lightgrey')

    scatter = [plt.scatter(0, -10, marker = 'o', s= 40, c = colors[i], ec = edge[i],
                           label = legend_labels[i])for i in range(2)]

    ax.legend(handles = scatter,loc = 1, fontsize = 20)

def arrowmapopta(data, name=None):
    if name == None:
        carry = data
    else:
        name = name
        carry= data.loc[data['playerId'] == player_ids[name]]


    pitch = Pitch(pitch_type = 'opta', pitch_color = 'black', line_color = 'white',
                          pad_top = 10, linewidth = 1, goal_type = 'box')
    fig, ax = pitch.draw(figsize = (16,12))
    for i in range(len(carry)):
        if carry['outcomeType'].iloc[i] == 'Successful':
            plt.scatter(carry['x'].iloc[i],carry['y'].iloc[i], s=40,c='green', ec = 'lightgreen',
                        zorder =5)
            plt.annotate("", xytext=(carry['x'].iloc[i], carry['y'].iloc[i]),xy=(carry['endX'].iloc[i],carry['endY'].iloc[i]),
                     arrowprops=dict(arrowstyle="->", color = 'white', lw = 1), zorder = 5)
        else:
            plt.scatter(120 *carry['x'].iloc[i]/100,80* (100-carry['y'].iloc[i])/100, s=40,c='green', ec = 'lightgreen',
                        alpha = 0.5, zorder =5)
            plt.annotate("", xytext=((120 *carry['x'].iloc[i]/100),80* (100-carry['y'].iloc[i])/100),
                     xy=((120 *carry['endX'].iloc[i]/100),80 * (100-carry['endY'].iloc[i])/100),
                     arrowprops=dict(arrowstyle="->", color = 'lightgrey', lw = 0.8), zorder = 5)
                     
def assist_shotmap(data, league, season, team,teamid=96, size=400,save= True):

    from mplsoccer import Standardizer
    if team == 'Stoke City':
        df = data.loc[data['teamId'] == 96]
    else:
        df = data.loc[data['teamId'] != 96]
    if teamid != 96:
        df = data.loc[data['teamId'] == teamid]
    
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
            ec.append('orange')
            lw.append(2)
        elif df['type'].iloc[i] == 'SavedShot':
            ec.append('blue')
            lw.append(3)
        elif df['shotOffTarget'].iloc[i] == True:
            ec.append('cyan')
            lw.append(2)
        if df['Assisted'].iloc[i] == True:
            if df['assist_cross'].iloc[i] == True:
                if df['situation'].iloc[i] in ['SetPiece','FromCorner']:
                    markers.append('D')
                    sp_crosses+=1
                else:
                    markers.append('s')
                    op_crosses+=1
            else:
                if df['assist_throughball'].iloc[i] == True:
                    markers.append('^')
                    throughball+=1
                elif df['assist_pass'].iloc[i] == 1:
                    markers.append('o')
                    passes+=1
        else:
            if df['assist_def'].iloc[i] == True:
                markers.append('P')
                defactions+=1
            elif df['situation'].iloc[i] == 'DirectFreekick':
                markers.append('*')
                directfk+=1
            elif df['Assisted'].iloc[i] != True:
                markers.append('p')
                other+=1
            else:
                markers.append('X')


    norm = plt.Normalize(0,1)
    colmap = col.LinearSegmentedColormap.from_list("",["white",'orange','red',"darkred"])
    rescale = lambda y: (y - 0) / (1)
    colour = colmap(rescale(df['xG']))
    
    pitch = VerticalPitch(half = True,pitch_type = 'statsbomb', goal_type='box',
                    pitch_color = '#101010', line_color = 'white', pad_top = 17)

    fig, ax = pitch.draw(figsize = (16,18))

    inarea = df.loc[df['x'] >= 102]
    inarea = inarea.loc[inarea['y'] >= 18]
    inarea = inarea.loc[inarea['y'] <= 62]


    for i in range(len(df['x'])):
        plt.scatter(df['y'].iloc[i],df['x'].iloc[i],
                    fc = colour[i], ec = ec[i],
                    linewidth = lw[i], marker=markers[i], alpha = 0.9,
                    s = size, zorder=zorder[i])

    legend_labels = list(['Goal', 'Saved', 'Blocked', 'Off Target'])
    colors = ['red', 'blue','orange','cyan']
    edge = (3,2,1,1)

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
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colmap),
                 ax=ax, cax = axins, orientation = 'horizontal')

    cbar.set_label('xG', size=24, color = 'white')

    cbar.ax.tick_params(labelsize=21, color = 'white', labelcolor = 'white')

    ax.scatter(12, np.mean(df['x']), s=300, c='white', ec='white')


    ax.plot((12,12), (120,np.mean(df['x'])), c='white', lw=2)


    ax.text(s='Average \nShot Distance\n' +str(np.round(np.mean(120-df['x']),1)) + 'm',
           x =9.2, y= np.mean(df['x'])+1.5, c='white', rotation = 90, size = 16)

    plt.title('Shot Map - ' + str(team) + '\n' + str(league) + ' - ' + str(season), c='white', size = 32, y=-15)

    plt.scatter(40,108,c='white',s=100)

    plt.text(0.95,-17.5, s= '@Potterlytics\npotterlytics.blog\nData via Opta\nxG via Potterlytics',fontsize = 18,
             c='white')
    plt.text(-0.3,-17.5, s= str(int(sum(df['isGoal']))) + ' goals from ' + str(np.round(sum(df['xG']),2)) + ' non-penalty xG\n'
             + str(np.round(sum(df['xG'])/len(df), 2)) + ' xG per Shot\n'
             + str(len(df)) + ' shots, '+ str(len(df.loc[df['shotOnTarget'] == True])) + ' on target\n'
             + str(len(df.loc[df['xG'] > 0.2])) + ' shots > 0.2xG'  + ' + ' + str(len(penscored)) + '/' + str(len(penscored) + len(penmissed)) + ' penalties',
             fontsize = 18, c='white')

    if save == True:
        plt.savefig('data/stoke23_24/shotmap.png', bbox_inches = 'tight', dpi = 500)
