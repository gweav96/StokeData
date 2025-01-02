import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import visuals
import numpy as np
import main
import pickle
import whoscored_data_engineering as wsde
import summaryplots as sp
import pitchplots as pp
import ast

st.set_page_config(page_title="Stoke Data", layout="wide")


st.html("""
  <style>
    [alt=Logo] {
      height: 10rem;
    }
  </style>
        """)
st.logo('logo.jpg')

    # Large title for the toggle
st.markdown("<h1 style='text-align: center; font-size: 40px;'>EFL Championship Data 2024/2025</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 30px;'>Potterlytics.blog - @potterlytics</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-size: 35px;'>--------------------------------------------------------</h2>", unsafe_allow_html=True)

@st.cache_data
def load_over_data():
    matchfiles = pd.read_csv('2425/matchfiledata.csv')
    matchdf =pd.read_csv('2425/matchesdata.csv')
    shotsdf = pd.read_csv('2425/allshots.csv')
    return matchfiles, matchdf, shotsdf
    
matchfiles, matchdf, shotsdf = load_over_data()

teamdf = pd.DataFrame()
teamdf['teamname'] = matchfiles['home'].unique()
for teamname in matchfiles['home'].unique():
    teamdf.loc[teamdf['teamname'] == teamname, 'teamid'] = matchfiles.loc[matchfiles['home'] == teamname]['hometeam.id'].iloc[0]

st.markdown(
    """
    <style>
    .custom-header {
        margin-bottom: -50px; /* Adjust this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h2 class='custom-header' style='text-align: center; font-size: 20px;'>Select Team</h1>", unsafe_allow_html=True)

with st.sidebar:
    fixed_container = st.container()
    with fixed_container:
        team = st.sidebar.selectbox('', options =
        list(matchfiles['home'].sort_values().unique()))
        
st.sidebar.markdown("<h2 class='custom-header' style='text-align: center; font-size: 20px;'>Select Summary/Season/Pitch Plots</h1>", unsafe_allow_html=True)

tabchoice = st.sidebar.selectbox('',
        ['Match Summary Plots', 'Pitch Plots', 'Season Plots - COMING SOON']
    )

summaryplots = ['Game Momentum', 'Cumulative EPV', 'Shot Maps', 'Average Position Maps', 'Passing Sonars',
'xT Heatmaps', 'Final 3rd Entries', 'Passes In Final 3rd', 'Shot Assists']

@st.cache_data()
def load_team_data(team):
    
    teammatches = matchfiles.loc[(matchfiles['home'] == str(team)) | (matchfiles['away'] == str(team))]
    teammatches['Match'] = 0
    teammatchids = teammatches['matchId']
    teammatchdf = matchdf.loc[matchdf['matchId'].isin(teammatchids)].sort_values(by='startDate')
    teammatchids = np.array(matchdf['matchId'].unique())
    q=1
    for n in teammatchids:
        d = teammatches.loc[teammatches['matchId'] == int(n)]
        if len(d) > 0:
            h = d['home'].iloc[0]
            a = d['away'].iloc[0]
            
            match = str(q) + ' - ' + str(h) + ' - ' + str(a)
            teammatches.loc[teammatches['matchId'] == n, 'Match'] = match
            
            q = q+1
        
    teammatches = teammatches.sort_values(by='Match', key=lambda x: x.str.extract(r'(\d{1,2})')[0].astype(int)).reset_index(drop=True)
    
    events = pd.DataFrame()
    for n in teammatches['file']:
        load = pd.read_csv('2425/' + str(n))
        events = pd.concat((events, load))
    
    return teammatches, events
    
teammatches, events = load_team_data(team)

@st.cache_data()
def load_zones_data():
    pass_start_zones = pd.read_csv('2425/pass_start_zones.csv')
    pass_end_zones = pd.read_csv('2425/pass_end_zones.csv')
    
    return pass_start_zones, pass_end_zones

pass_start_zones, pass_end_zones = load_zones_data()

recipient = []
for n in events['matchId'].unique():
    df = events.loc[events['matchId'] == n]
    passdf = wsde.get_recipient(df)
    rec = list(passdf['pass_recipient'].values)
    recipient = recipient + rec


events['pass_recipient'] = recipient

playerdf = pd.DataFrame()
playerdf['playerName'] = events['playerName'].dropna().unique()
for name in playerdf['playerName']:
    playerdf.loc[playerdf['playerName'] == name, 'playerId'] = events.loc[events['playerName'] == name]['playerId'].iloc[0]

if tabchoice == "Match Summary Plots":
    with st.sidebar:

        st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Choose Match</h2>", unsafe_allow_html=True)
        matchchoice = st.selectbox('',
        list(teammatches['Match']))

        
        st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Choose Summary Plot</h2>", unsafe_allow_html=True)
        plotchoice = st.selectbox('', summaryplots)

    match_file = teammatches.loc[teammatches['Match'] == matchchoice]
    
    matchid = match_file['matchId'].iloc[0]
    match_df = matchdf.loc[matchdf['matchId'] == matchid]
    events_df = events.loc[events['matchId'] == matchid]
    with open('2425/' + str(match_file['file'].iloc[0])[:-4] + '_matchdata.pkl', 'rb') as f:
        match_data = pickle.load(f)
    events_df['Women'] = 0

    shots = shotsdf.loc[shotsdf['matchId'] == matchid]
    events_df = main.addEpvToDataFrame(events_df)
    
    recipient = []

    passdf = wsde.get_recipient(events_df)
    rec = list(passdf['pass_recipient'].values)
    recipient = recipient + rec


    events_df['pass_recipient'] = recipient
    
    if plotchoice == 'Game Momentum':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('The sum of expected threat created is calculated for each team across a 5 minute rolling window. The plot then shows the home team\'s xT sum minus the away team\'s xT sum, scaled and plotted against the minute of the game.')
                st.write('Each team\'s goals are plotted with the dashed red/blue lines, with the scoreline shown above. Half time is denoted by a grey dashed line.')
        fig, ax = sp.momentum(events_df, match_file)
        st.pyplot(fig=fig)
        
    elif plotchoice == 'Cumulative EPV':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('The sum of expected threat created is calculated for each team. The plot then shows the cumulative sum of the home team\'s EPV generated in red, and the away team\'s EPV generated in blue, scaled and plotted against the minute of the game. This only includes positive EPV actions.')
                st.write('Each team\'s goals are plotted with the dashed red/blue lines, with the scoreline shown above. Half time is denoted by a grey dashed line.')
                st.write('You can choose whether to include negative EPV events (passes backwards etc.), which will penalise teams for playing backwards and recycling possession. Please think about your use case for this plot before deciding whether or not to include negative EPV events!')
        neg = col.toggle("Include Negative EPV Events?")
        fig, ax = sp.cumulative_momentum(events_df, match_file, neg)
        st.pyplot(fig=fig)
        
    elif plotchoice == 'Shot Maps':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('The position of each shot in the game is shown on the pitch, with the outer colour of the marker denoting the end result, the inner colour denoting the xG value of the shot (as per the top colourbar), and the shape of the marker denoting the type of assist.')
                st.write('All xG values are calculated using my bespoke xG model, it\'s not recommended to compare these xG values with data from other models.')
        col1, col2 = st.columns(2)
        fig, ax = sp.shotmaps(shots, match_file, match_file['hometeam.id'].iloc[0], match_file['home'].iloc[0], match_file['away'].iloc[0], 'H')
        col1.pyplot(fig=fig)
        fig, ax = sp.shotmaps(shots, match_file, match_file['awayteam.id'].iloc[0], match_file['away'].iloc[0], match_file['home'].iloc[0], 'A')
        col2.pyplot(fig=fig)
    
    elif plotchoice == 'Average Position Maps':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('The position of each marker shows the player\'s average position on the ball, with the size of the marker indicating the number of pass receipts for that player.')
                st.write('Arrows indicate the most frequent passing combinations, with thicker arrows denoting more common passes.')
                st.write('You can select the colour of each marker with the drop-boxes above the plots.')

        colours = ['Black', 'White', 'Gold', 'Red', 'Pink', 'Blue', 'Navy', 'Orange', 'Green', 'Cyan']
        col1, col2 = st.columns(2)
        homecolour = col1.selectbox('Choose Home Kit Colour', list(colours))
        awaycolour = col2.selectbox('Choose Away Kit Colour', list(colours))
        if homecolour in ['White', 'Gold', 'Pink', 'Cyan']:
            hometext = 'k'
        else:
            hometext = 'white'
        if awaycolour in ['White', 'Gold', 'Pink', 'Cyan']:
            awaytext = 'k'
        else:
            awaytext = 'white'
            
        fig, ax = sp.averagepassmaps(match_data, events_df, match_file, homecolour, hometext, 'home')
        col1.pyplot(fig=fig)
        
        fig, ax = sp.averagepassmaps(match_data, events_df, match_file, awaycolour, awaytext, 'away')
        col2.pyplot(fig=fig)
        
    elif plotchoice == 'Passing Sonars':
    
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('For each of the starting XI on the pitch, the bars show the frequency with which passes were attempted in each direction.')
                st.write('White bars indicate pass attempts, and blue bars indicate successful passes. The darker the blue colour, the longer the average passing distance in this direction.')
                st.write('The player\'s position on the pitch is based on Opta formation data, and not indicative of any real positioning during the game.')
    
        col1, col2 = st.columns(2)
        formation = ast.literal_eval(match_df['home'].iloc[0])[0]['formations'][0]["formationName"]
        fig, ax = sp.pass_sonars(events_df, match_df, match_file, match_data['home']['teamId'], formation, 'home')
        col1.pyplot(fig=fig)
        
        formation = ast.literal_eval(match_df['away'].iloc[0])[0]['formations'][0]["formationName"]
        fig, ax = sp.pass_sonars(events_df, match_df, match_file, match_data['away']['teamId'], formation, 'away')
        col2.pyplot(fig=fig)
        
    elif plotchoice == 'xT Heatmaps':
    
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('The heatmap shows the locations from which the highest total Expected Threat was generated.')
                st.write('The arrows then indicate the highest xT passes, with more opaque arrows indicating higher xT.')

    
        col1, col2 = st.columns(2)
        fig, ax = sp.xt_map(events_df, match_df, match_file, match_data['home']['teamId'], 'home')
        col1.pyplot(fig=fig)
        
        fig, ax = sp.xt_map(events_df, match_df, match_file, match_data['away']['teamId'], 'away')
        col2.pyplot(fig=fig)

    elif plotchoice == 'Final 3rd Entries':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('White arrows indicate passes from open play (and goal kicks) starting in the defensive & middle 3rds of the pitch, ending in the final 3rd. Blue arrows indicate ball carries from the defensive and middle 3rds into the final 3rd.')
        col1, col2 = st.columns(2)
        fig, ax = sp.pass_to_fin3rd(events_df, match_df, match_file, match_data['home']['teamId'], 'home')
        col1.pyplot(fig=fig)
        
        fig, ax = sp.pass_to_fin3rd(events_df, match_df, match_file, match_data['away']['teamId'], 'away')
        col2.pyplot(fig=fig)

    elif plotchoice == 'Passes In Final 3rd':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('White arrows indicate successful passes from open play starting in the final 3rd of the pitch, with blue arrows indicating ball carries, and red arrows indicating unsuccessful passes')
    
        col1, col2 = st.columns(2)
        fig, ax = sp.pass_in_fin3rd(events_df, match_df, match_file, match_data['home']['teamId'], 'home')
        col1.pyplot(fig=fig)
        
        fig, ax = sp.pass_in_fin3rd(events_df, match_df, match_file, match_data['away']['teamId'], 'away')
        col2.pyplot(fig=fig)
        
    elif plotchoice == 'Shot Assists':
        _, col, __ = st.columns([1,2,1])
        with col:
            with st.expander('Click Here For Plot Explanation'):
                st.write('White arrows indicate successful passes not originating from corners that lead directly to a shot. The shirt number of the player making the pass is shown in the green marker.')
        col1, col2 = st.columns(2)
        fig, ax = sp.shot_assists(events_df, match_df, match_file, match_data['home']['teamId'], 'home')
        col1.pyplot(fig=fig)
        
        fig, ax = sp.shot_assists(events_df, match_df, match_file, match_data['away']['teamId'], 'away')
        col2.pyplot(fig=fig)

st.markdown("""
    <style>
    .stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
    }
    </style>
""",unsafe_allow_html=True)
if tabchoice == 'Pitch Plots':
    with st.sidebar:
        st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Choose Plot Style</h2>", unsafe_allow_html=True)
        plotchoice = st.selectbox('',
        ['Shot Map', 'Full Pitch', 'Attacking Half - COMING SOON', 'Defending Half - COMING SOON'])
    st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Choose Time Period</h2>", unsafe_allow_html=True)
    matchchoice = st.radio('',
        ['Last Match', 'Last 3 Matches', 'All Season', 'Select Specific Matches'],
        horizontal = True)
    if matchchoice == 'Select Specific Matches':
        sizeindex = 2
        st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'><<< Your Selected Matches Will Be Shown In The Sidebar</h2>", unsafe_allow_html=True)
        specmatch = st.multiselect('',
                                   list(teammatches['Match']))
        specmatch = sorted(specmatch, key=lambda x: int(x.split(' - ')[0]))

        if len(specmatch) > 0:
            with st.sidebar:
                st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 18px;'>SELECTED MATCHES:</h2>", unsafe_allow_html=True)
                for i in range(len(specmatch)):
                    st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 16px;'>- " + str(specmatch[i]) + "</h2>", unsafe_allow_html=True)
    elif matchchoice == 'All Season':
        specmatch = list(teammatches['Match'])
        sizeindex = 1
        with st.sidebar:
            st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 18px;'>SELECTED MATCHES:</h2>", unsafe_allow_html=True)
            for i in range(len(specmatch)):
                st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 16px;'>- " + str(specmatch[i]) + "</h2>", unsafe_allow_html=True)
    elif matchchoice == 'Last Match':
        specmatch = (teammatches['Match'].iloc[-1])
        sizeindex = 2
        with st.sidebar:
            st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 18px;'>SELECTED MATCH:</h2>", unsafe_allow_html=True)
            st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 16px;'>- " + str(specmatch) + "</h2>", unsafe_allow_html=True)
    elif matchchoice == 'Last 3 Matches':
        specmatch = list(teammatches['Match'].iloc[-3:])
        sizeindex = 2
        with st.sidebar:
            st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 18px;'>SELECTED MATCHES:</h2>", unsafe_allow_html=True)
            for i in specmatch:
                st.markdown("<h2 class='custom-header' style='text-align: left; font-size: 16px;'>- " + str(i) + "</h2>", unsafe_allow_html=True)
                
    if type(specmatch) == list:
        if len(specmatch) > 0:
            matchids = teammatches.loc[teammatches['Match'].isin(specmatch)]['matchId']
        elif len(specmatch) <1:
            st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 16px;'>SELECT A MATCH ABOVE</h2>", unsafe_allow_html=True)
            matchids = []
    elif len(specmatch) > 0:
        matchids = teammatches.loc[teammatches['Match'] == specmatch]['matchId']
        
    if plotchoice == 'Shot Map':

        if len(matchids) > 0:
            shots = shotsdf.loc[shotsdf['matchId'].isin(matchids)]
            st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Show Team Or Opposition Data</h2>", unsafe_allow_html=True)
            teamoppchoice = st.radio('',
                                  [str(team), 'Opposition'],
            horizontal = True)
            
            if teamoppchoice == team:
                events_df = shots.loc[shots['teamId'] == (teamdf.loc[teamdf['teamname'] == team]['teamid'].iloc[0])]
            elif teamoppchoice == 'Opposition':
                events_df = shots.loc[shots['teamId'] != (teamdf.loc[teamdf['teamname'] == team]['teamid'].iloc[0])]
            st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>SELECT FROM THE OPTIONS BELOW TO CUSTOMISE YOUR PLOT</h2>", unsafe_allow_html=True)
            st.write('')
            st.write('')
            preopt = len(events_df)
            with st.expander('FURTHER PLOT OPTIONS ARE AVAILABLE IN THIS DROP-DOWN:', expanded=True):
                col1,col2,col3 = st.columns(3)
                    
                half = col1.radio('Choose Time Period',
                                  ['All', 'First Half', 'Second Half'], horizontal=True, index=0)
                                  
                if half == 'First Half':
                    events_df = events_df.loc[events_df['period'] == 'FirstHalf']
                elif half == 'Second Half':
                    events_df = events_df.loc[events_df['period'] == 'SecondHalf']
                
                box = col2.radio('Choose Shot Distance',
                                 ['All', 'Inside Box', 'Outside Box'], horizontal=True, index=0)
                if box == 'Inside Box':
                    events_df = events_df.loc[(events_df['shotPenaltyArea'] == True) | (events_df['shotSixYardBox'] == True)]
                elif box == 'Outside Box':
                    events_df = events_df.loc[(events_df['shotPenaltyArea'] == False) & (events_df['shotSixYardBox'] == False)]
                    
                
                xgchoice = col3.slider('Choose A Range Of xG Values',
                0.00,1.00, (0.00,1.00))
                
                events_df = events_df.loc[(events_df['xG']<= xgchoice[1]) & (events_df['xG']>= xgchoice[0])]
                
                col1,col2 = st.columns(2)
                outcome = col1.multiselect('Choose Shot Outcome(s)',
                ['Goal', 'Saved', 'Off Target', 'Blocked'])
                if outcome:
                    conditions = []
                    if 'Goal' in outcome:
                        conditions.append(events_df['isGoal'] == True)
                    if 'Saved' in outcome:
                        conditions.append(events_df['type'] == 'SavedShot')
                    if 'Off Target' in outcome:
                        conditions.append(events_df['shotOffTarget'] == True)
                    if 'Blocked' in outcome:
                        conditions.append(events_df['shotBlocked'] == True)

                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition |= condition

                    events_df = events_df.loc[combined_condition]
                
                assist = col2.multiselect('Choose Assist Type(s)',
                                ['Pass', 'Cross', 'Open-Play Cross', 'Set Piece Cross', 'Through Ball', 'Defensive Action', 'Unassisted', 'Direct Free Kick'])
                if assist:
                    conditions = []
                    if 'Pass' in assist:
                        conditions.append((events_df['assist_pass'] == True) & (events_df['Assisted'] == True))
                    if 'Cross' in assist:
                        conditions.append(events_df['assist_cross'] == True)
                    if 'Through Ball' in assist:
                        conditions.append(events_df['assist_throughball'] == True)
                    if 'Defensive Action' in assist:
                        conditions.append(events_df['assist_def'] == True)
                    if 'Unassisted' in assist:
                        conditions.append((events_df['Assisted'] == 0) & (events_df['situation'] != 'DirectFreekick') & (events_df['assist_def'] != True))
                    if 'Set Piece Cross' in assist:
                        conditions.append((events_df['situation'].isin(['SetPiece','FromCorner'])) & (events_df['assist_cross'] == True))
                    if 'Open-Play Cross' in assist:
                        conditions.append((events_df['situation'] == 'OpenPlay') & (events_df['assist_cross'] == True))
                    if 'Direct Free Kick' in assist:
                        conditions.append(events_df['situation'] == 'DirectFreekick')
                        
                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition |= condition

                    events_df = events_df.loc[combined_condition]
                    
                playershot = col1.multiselect('Choose Player(s)',
                                              sorted(list(events_df['playerName'].unique()), key=lambda x: x.split(' ')[-1]))
                if playershot:
                    if len(playershot) == 1:
                        events_df = events_df.loc[events_df['playerName'] == playershot[0]]
                    elif len(playershot) > 1:
                        events_df = events_df.loc[events_df['playerName'].isin(list(playershot))]
                
                assistids = events_df['relatedPlayerId'].unique()
                assistnames = sorted(list(playerdf.loc[playerdf['playerId'].isin(assistids)]['playerName']), key=lambda x: x.split(' ')[-1])
                playerassist = col2.multiselect('Choose Player(s) Assisting Shot',
                assistnames)
                if playerassist:
                    chosen_assistids = playerdf.loc[playerdf['playerName'].isin(playerassist)]['playerId']
                    events_df = events_df.loc[events_df['relatedPlayerId'].isin(chosen_assistids)]
                    
        if matchchoice and len(matchids)>0:
            st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>IF YOU'D LIKE TO CHANGE THE TITLE, INPUT YOUR CUSTOM TEXT BELOW</h2>", unsafe_allow_html=True)
            st.write('')
            st.write('')
            col1,col2,col3 = st.columns(3)
            if teamoppchoice == 'Opposition':
                teamoppchoice = 'Opposition vs ' + str(team)
            text1 = col1.text_input('Custom Title Part 1', str(teamoppchoice))
            
            if matchchoice == 'Select Specific Matches':
                if len(matchids) == 1:
                    game = str(specmatch[0].split('-')[1]) + 'vs.' + str(specmatch[0].split('-')[2])
                    matchchoice = str(game)
                elif len(matchids) > 1:
                    matchchoice = 'Selected ' + str(len(matchids)) + ' Matches'
            elif matchchoice == 'Last Match':
                    game = str(specmatch.split('-')[1]) + 'vs.' + str(specmatch.split('-')[2])
                    matchchoice = str(game)

            text2 = col2.text_input('Custom Title Part 2', str(matchchoice))
            
            if len(events_df) == preopt:
                t3 = "All Shots"
            else:
                t3 = "Custom Selected Shots"
            text3 = col3.text_input('Custom Title Part 3', t3)
            
            markersize = col2.radio('Choose Marker Size',
                                    ['Small', 'Medium', 'Large'], horizontal=True, index=sizeindex)
                                    
            if markersize == 'Large':
                size = 600
            elif markersize == 'Medium':
                size = 450
            elif markersize == 'Small':
                size = 300
            
            if len(events_df.loc[(events_df['penaltyScored'] !=True) & (events_df['penaltyMissed'] != True)]) == 0:
                st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>NO OPEN-PLAY SHOTS OCCURRED WITHIN YOUR CHOSEN OPTIONS, PLEASE CHANGE AT LEAST ONE OPTION ABOVE</h2>", unsafe_allow_html=True)
            else:
                fig, ax = pp.shotmaps(events_df, 'match_file', 'teamid', 'teamname', text1, text2, text3, size)
                st.pyplot(fig=fig)
    elif plotchoice == 'Full Pitch':
        if len(matchids) > 0:
            events_df = events.loc[events['matchId'].isin(matchids)]
#            st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Choose Which Actions You Want To Show</h2>", unsafe_allow_html=True)
#            actionchoice = st.radio('',
#                                    ['Passes', 'Defensive Actions - Coming Soon!'], horizontal=True)
            actionchoice= 'Passes'
            if actionchoice == 'Passes':
                events_df = events_df.loc[events_df['type']=='Pass']
                st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>Show Team Or Opposition Data</h2>", unsafe_allow_html=True)
                teamoppchoice = st.radio('',
                                  [str(team), 'Opposition'], horizontal = True)
                if teamoppchoice == team:
                    events_df = events_df.loc[events_df['teamId'] == (teamdf.loc[teamdf['teamname'] == team]['teamid'].iloc[0])]
                elif teamoppchoice == 'Opposition':
                    events_df = events_df.loc[events_df['teamId'] != (teamdf.loc[teamdf['teamname'] == team]['teamid'].iloc[0])]
                preopt = len(events_df)
                ev_pre = events_df
                st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>SELECT CHALKBOARD MAP OR HEATMAP</h2>", unsafe_allow_html=True)
                chalk = st.radio("", ['Chalkboard', 'Heatmap'], horizontal = True)
                if chalk == 'Chalkboard':
                    st.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>SELECT FROM THE OPTIONS BELOW TO CUSTOMISE YOUR PLOT</h2>", unsafe_allow_html=True)
                    st.write('')
                    st.write('')
                    with st.expander('CUSTOMISE THE DATA YOU SHOW IN THIS DROP-DOWN:'):
                        
                        col1,col2,col3,col4 = st.columns(4)
                        half = col1.radio('Choose Time Period',
                                      ['All', 'First Half', 'Second Half'], horizontal=True, index=0)
                                      
                        if half == 'First Half':
                            events_df = events_df.loc[events_df['period'] == 'FirstHalf']
                        elif half == 'Second Half':
                            events_df = events_df.loc[events_df['period'] == 'SecondHalf']
                        
                        mins = col2.slider('Select Range of Minutes', int(events_df['minute'].min()),int(events_df['minute'].max()), (int(events_df['minute'].min()),int(events_df['minute'].max())))
                        if mins:
                            events_df = events_df.loc[(events_df['minute'] >= mins[0]) & (events_df['minute']<= mins[1])]
                        
                        openplay = col3.multiselect('Choose Situation', ['Open Play', 'Corner', 'Free Kick', 'Goal Kick', 'Throw In'])
                        
                        if openplay:
                            conditions = []
                            if 'Open Play' in openplay:
                                conditions.append((events_df['passFreekick'] != True) & (events_df['passCorner'] != True) & (['GoalKick' not in i for i in events_df['qualifiers']]) & (events_df['throwIn'] != True))
                            if 'Corner' in openplay:
                                conditions.append(events_df['passCorner'] == True)
                            if 'Free Kick' in openplay:
                                conditions.append(events_df['passFreekick'] == True)
                            if 'Goal Kick' in openplay:
                                conditions.append(events_df['qualifiers'].str.contains('GoalKick'))
                            if 'Throw In' in openplay:
                                conditions.append(events_df['qualifiers'].str.contains('ThrowIn'))
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                        
                        success = col4.radio('Choose Outcome', ['All', 'Successful', 'Unsuccessful'], horizontal=True)
                        
                        if success != 'All':
                            events_df = events_df[events_df['outcomeType'] == success]
                        
                        col1, col2 = st.columns(2)
                        startzone_h = col1.multiselect('Choose Horizontal Start Zone(s)', ['Defensive Penalty Area', 'Defensive 3rd', 'Middle 3rd', 'Final 3rd', 'Attacking Penalty Area'])
                        
                        if startzone_h:
                            conditions = []
                            if 'Defensive 3rd' in startzone_h:
                                conditions.append(events_df['defensiveThird'] == True)
                            if 'Middle 3rd' in startzone_h:
                                conditions.append(events_df['midThird'] == True)
                            if 'Final 3rd' in startzone_h:
                                conditions.append(events_df['finalThird'] == True)
                            if 'Defensive Penalty Area' in startzone_h:
                                conditions.append((events_df['x'] <=16) & (events_df['y']<=81) & (events_df['y']>=19))
                            if 'Attacking Penalty Area' in startzone_h:
                                conditions.append((events_df['x'] >=81) & (events_df['y']<=81) & (events_df['y']>=19))
                            
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        endzone_h = col2.multiselect('Choose Horizontal End Zone(s)', ['Defensive Penalty Area', 'Defensive 3rd', 'Middle 3rd', 'Final 3rd', 'Attacking Penalty Area'])
                        if endzone_h:
                            conditions = []
                            if 'Defensive 3rd' in endzone_h:
                                conditions.append(events_df['endX'] <= 33)
                            if 'Middle 3rd' in endzone_h:
                                conditions.append((events_df['endX']>33) & (events_df['endX']<= 66))
                            if 'Final 3rd' in endzone_h:
                                conditions.append(events_df['endX']> 66)
                            if 'Defensive Penalty Area' in endzone_h:
                                conditions.append((events_df['endX'] <=16) & (events_df['endY']<=81) & (events_df['endY']>=19))
                            if 'Attacking Penalty Area' in endzone_h:
                                conditions.append((events_df['endX'] >=81) & (events_df['endY']<=81) & (events_df['endY']>=19))
                                
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        startchannel_h = col1.multiselect('Choose Vertical Start Channel(s)', ['Left Wing', 'Left Half Space', 'Central', 'Right Half Space', 'Right Wing'])
                        
                        if startchannel_h:
                            conditions = []
                            if 'Left Wing' in startchannel_h:
                                conditions.append(events_df['y'] <= 19)
                            if 'Left Half Space' in startchannel_h:
                                conditions.append((events_df['y'] > 19) & (events_df['y'] < 37))
                            if 'Central' in startchannel_h:
                                conditions.append((events_df['y'] >= 37) & (events_df['y'] <= 63))
                            if 'Right Wing' in startchannel_h:
                                conditions.append(events_df['y'] >= 81)
                            if 'Eight Half Space' in startchannel_h:
                                conditions.append((events_df['y'] < 81) & (events_df['y'] < 63))
                                
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        endchannel_h = col2.multiselect('Choose Vertical End Channel(s)', ['Left Wing', 'Left Half Space', 'Central', 'Right Half Space', 'Right Wing'])
                        
                        if endchannel_h:
                            conditions = []
                            if 'Left Wing' in endchannel_h:
                                conditions.append(events_df['endY'] <= 19)
                            if 'Left Half Space' in endchannel_h:
                                conditions.append((events_df['endY'] > 19) & (events_df['endY'] < 37))
                            if 'Central' in endchannel_h:
                                conditions.append((events_df['endY'] >= 37) & (events_df['endY'] <= 63))
                            if 'Right Wing' in endchannel_h:
                                conditions.append(events_df['endY'] >= 81)
                            if 'Eight Half Space' in endchannel_h:
                                conditions.append((events_df['endY'] < 81) & (events_df['endY'] < 63))
                                
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        col1, col2, col3, col4 = st.columns(4)
                        
                        startx = col1.slider('Choose Specific Horizontal Start Co-ordinates (Opta)',
                        0,100, (0,100))
                        
                        if startx:
                            events_df = events_df.loc[(events_df['x'] >= startx[0]) & (events_df['x']<=startx[1])]
                            
                        endx = col2.slider('Choose Specific Horizontal End Co-ordinates (Opta)',
                        0,100, (0,100))
                        
                        if endx:
                            events_df = events_df.loc[(events_df['endX'] >= endx[0]) & (events_df['endX']<=endx[1])]
                            
                        starty = col3.slider('Choose Specific Vertical Start Co-ordinates (Opta)',
                        0,100, (0,100))
                        
                        if starty:
                            events_df = events_df.loc[(events_df['y'] >= starty[0]) & (events_df['y']<=starty[1])]
                            
                        starty = col4.slider('Choose Specific Vertical End Co-ordinates (Opta)',
                        0,100, (0,100))
                        
                        if starty:
                            events_df = events_df.loc[(events_df['endY'] >= starty[0]) & (events_df['endY']<=starty[1])]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        eventchoice = col1.multiselect('Choose Specific Event(s)', ['Assist', 'Key Pass', 'Cross', 'Through Ball', 'Progressive', 'Switch'])
                        
                        if eventchoice:
                            conditions = []
                            if 'Key Pass' in eventchoice:
                                conditions.append(events_df['passKey'] == True)
                            if 'Assist' in eventchoice:
                                conditions.append(events_df['assist'] == True)
                            if 'Cross' in eventchoice:
                                conditions.append((events_df['passCrossAccurate']==True) | (events_df['passCrossInaccurate'] == True))
                            if 'Through Ball' in eventchoice:
                                conditions.append((events_df['passThroughBallAccurate']==True) | (events_df['passThroughBallInaccurate'] == True))
                            if 'Progressive' in eventchoice:
                                conditions.append((events_df['x']>33) & (((100 - events_df['endX'])**2 + (50 - events_df['endY'])**2 < 0.75 * ((100 - events_df['x'])**2 + (50 - events_df['y'])**2))))
                            if 'Switch' in eventchoice:
                                conditions.append((((events_df['endY']<50) & (events_df['y']>50)) | ((events_df['endY']>50) & (events_df['y']<50))) & ((events_df['endY'] <= 19) | (events_df['endY'] >= 81)))
                                
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        player = col2.multiselect('Choose Player(s) Making Passes', sorted(list(events_df['playerName'].unique()), key=lambda x: x.split(' ')[-1]))
                        
                        if player:
                            events_df = events_df.loc[events_df['playerName'].isin(player)]
                        
                        receivedids = events_df.loc[events_df['passAccurate'] == True]['pass_recipient'].unique()
                        receivednames = sorted(list(playerdf.loc[playerdf['playerId'].isin(receivedids)]['playerName']), key=lambda x: x.split(' ')[-1])
                        playerrec = col3.multiselect('Choose Player(s) Receiving Passes', receivednames)
                        
                        if playerrec:
                            idrec = playerdf.loc[playerdf['playerName'].isin(playerrec)]['playerId']
                            events_df = events_df.loc[events_df['pass_recipient'].isin(idrec)]
                        
                        col1, col2 = st.columns(2)
                        direction = col1.multiselect('Choose Pass Direction(s)', ['Forwards', 'Backwards', 'Left', 'Right'])
                        
                        if direction:
                            conditions = []
                            if 'Forwards' in direction:
                                conditions.append(events_df['passForward'] == True)
                            if 'Backwards' in direction:
                                conditions.append(events_df['passBack'] == True)
                            if 'Left' in direction:
                                conditions.append(events_df['passLeft'] == True)
                            if 'Right' in direction:
                                conditions.append(events_df['passRight'] == True)
                                
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                            
                        length = col2.multiselect('Choose Pass Distance(s)', ['Short', 'Medium', 'Long'])

                        if length:
                            conditions = []
                            if 'Short' in length:
                                conditions.append(((events_df['endX'] - events_df['x'])**2 + (events_df['endY'] - events_df['y'])**2) < 225)
                            if 'Medium' in length:
                                conditions.append((((events_df['endX'] - events_df['x'])**2 + (events_df['endY'] - events_df['y'])**2) >= 225) & (((events_df['endX'] - events_df['x'])**2 + (events_df['endY'] - events_df['y'])**2) < 900))
                            if 'Long' in length:
                                conditions.append(((events_df['endX'] - events_df['x'])**2 + (events_df['endY'] - events_df['y'])**2) >= 900)
                            combined_condition = conditions[0]
                            for condition in conditions[1:]:
                                combined_condition |= condition

                            events_df = events_df.loc[combined_condition]
                
                
                if chalk == 'Heatmap':
                    _,col,_ = st.columns([1,2,1])
                    col.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>SHOW START POINTS OR END POINTS OF PASSES (END POINTS AUTOMATICALLY SELECTS ONLY SUCCESSFUL PASSES)</h2>", unsafe_allow_html=True)
                    end = col.radio('', ['Start Points', 'End Points'] ,horizontal=True)
                    col.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>SHOW SCATTER POINTS ON PLOT?</h2>", unsafe_allow_html=True)
                    st.write("")
                    _,col,_ = st.columns([10,1,10])
                    scatter = col.toggle('', True)
                    _,col,_ = st.columns([1,2,1])
                    col.markdown("<h2 class='custom-header' style='text-align: center; font-size: 18px;'>COMPARE WITH LEAGUE AVERAGE?</h2>", unsafe_allow_html=True)
                    st.write("")
                    _,col,_ = st.columns([10,1,10])
                    bins = col.toggle(' ',True)
                    if end == 'End Points':
                        end = True
                        events_df = events_df[events_df['passAccurate'] == True]
                        average_values = pass_end_zones
                    else:
                        end = False
                        average_values = pass_start_zones
                    matchlen = len(events_df['matchId'].unique())
                    if bins == True:
                        events_df = events_df
                        binx=0
                        biny=0
                    else:
                        col1,col2 = st.columns(2)
                        binx = col1.slider('Select Number Of Horizontal Bins', 2,24, (6))
                        biny =col2.slider('Select Number Of Vertical Bins', 2,16, (5))
                
                col1,col2,col3 = st.columns(3)
                if teamoppchoice == 'Opposition':
                    teamoppchoice = 'Opposition vs ' + str(team)
                text1 = col1.text_input('Custom Title Part 1', str(teamoppchoice))
                
                if matchchoice == 'Select Specific Matches':
                    if len(matchids) == 1:
                        game = str(specmatch[0].split('-')[1]) + 'vs.' + str(specmatch[0].split('-')[2])
                        matchchoice = str(game)
                    elif len(matchids) > 1:
                        matchchoice = 'Selected ' + str(len(matchids)) + ' Matches'
                elif matchchoice == 'Last Match':
                        game = str(specmatch.split('-')[1]) + 'vs.' + str(specmatch.split('-')[2])
                        matchchoice = str(game)

                text2 = col2.text_input('Custom Title Part 2', str(matchchoice))
                
                if len(events_df) == preopt:
                    t3 = "All Passes"
                else:
                    t3 = "Custom Selected Passes"
                if chalk == 'Heatmap':
                    if end == True:
                        t3 = "Endpoint of Passes"
                    else:
                        t3 = "Startpoint of Passes"
                    if bins == True:
                        t3 = t3 + ' - Compared With League Average'
                text3 = col3.text_input('Custom Title Part 3', t3)
                if chalk == 'Chalkboard':
                    if len(events_df) > 500:
                        button = st.button("Lots of data! To avoid lag every time something is changed, click here when you're ready to make the plot", use_container_width=True)
                        if button == True:
                            fig, ax = pp.fullpitch_pass(events_df, text1, text2, text3)
                            st.pyplot(fig=fig)
                    elif len(events_df) > 0:
                        fig, ax = pp.fullpitch_pass(events_df, text1, text2, text3)
                        st.pyplot(fig=fig)
                else:
                    fig, ax = pp.fullpitch_pass_hmap(events_df,average_values, end,scatter, bins, text1, text2, text3, matchlen, binx, biny)
                    st.pyplot(fig=fig)
                    



