"""Module containing functions to add custom events to WhoScored-style data

Functions
---------
pre_assist(events):
    Calculate pre-assists from whoscored-style events dataframe, and returns with pre_assist column

progressive_action(single_event, inplay=True, successful_only=True)
    Identify progressive pass or carry from WhoScored-style event.

box_entry(single_event, inplay=True, successful_only=True):
    Identify pass or carry into box from whoscored-style event.

create_convex_hull(events_df, name='default', min_events=3, include_percent=100, pitch_area = 10000):
    Create a dataframe of convex hull information from statsbomb-style event data.

passes_into_hull(hull_info, events, opp_passes=True, xt_info=False):
    Add pass into hull information to dataframe of convex hulls for whoscored-style event data.

insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    Add carry events to whoscored-style events dataframe

get_xthreat(events_df, interpolate=True, pitch_length=105, pitch_width=68):
    Add expected threat metric to whoscored-style events dataframe
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay





def progressive_action(single_event, inplay=True, successful_only=True):
    """ Identify progressive pass or carry from WhoScored-style event.

    Function to identify progressive actions. An action is considered progressive if the distance between the
    starting point and the next touch is: (i) at least 30 meters closer to the opponent’s goal if the starting and
    finishing points are within a team’s own half, (ii) at least 15 meters closer to the opponent’s goal if the
    starting and finishing points are in different halves, (iii) at least 10 meters closer to the opponent’s goal if
    the starting and finishing points are in the opponent’s half. The function takes in a single event and returns a
    boolean (True = successful progressive action.) This function is best used with the dataframe apply method.

    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from WhoScored-style event dataframe.
        inplay (bool, optional): selection of whether to include 'in-play' events only. True by default.
        successful_only (bool, optional): selection of whether to only include successful actions. True by default

    Returns:
        bool: True = progressive action, nan = non-progressive action, unsuccessful action or not a pass/carry
    """

    # Determine if event is pass
    if single_event['eventType'] in ['Carry', 'Pass']:

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Check pass made in-play (if inplay = True)
        if inplay:
            check_inplay = not any(item in single_event['satisfiedEventsTypes'] for item in [48, 50, 51, 42, 44,
                                                                                             45, 31, 34, 212])
        else:
            check_inplay = True

        # Determine pass start and end position in yards (assuming standard pitch), and determine whether progressive
        x_startpos = 120*single_event['x']/100
        y_startpos = 80*single_event['y']/100
        x_endpos = 120*single_event['endX']/100
        y_endpos = 80*single_event['endY']/100
        delta_goal_dist = (np.sqrt((120 - x_startpos) ** 2 + (40 - y_startpos) ** 2) -
                           np.sqrt((120 - x_endpos) ** 2 + (40 - y_endpos) ** 2))

        # At least 30m closer to the opponent’s goal if the starting and finishing points are within a team’s own half
        if (check_success and check_inplay) and (x_startpos < 60 and x_endpos < 60) and delta_goal_dist >= 32.8:
            return True

        # At least 15m closer to the opponent’s goal if the starting and finishing points are in different halves
        elif (check_success and check_inplay) and (x_startpos < 60 and x_endpos >= 60) and delta_goal_dist >= 16.4:
            return True

        # At least 10m closer to the opponent’s goal if the starting and finishing points are in the opponent’s half
        elif (check_success and check_inplay) and (x_startpos >= 60 and x_endpos >= 60) and delta_goal_dist >= 10.94:
            return True
        else:
            return float('nan')

    else:
        return float('nan')


def box_entry(single_event, inplay=True, successful_only=True):
    """ Identify pass and carry into box from whoscored-style event.

    Function to identify successful passes and carriesthat end up in the opposition box. The function takes in a single
    event, and returns a boolean (True = successful action into the box.) This function is best used with the dataframe
    apply method.

    Args:
        single_event (pandas.Series): series corresponding to a single event (row) from whoscored-style event dataframe.
        inplay (bool, optional): selection of whether to include 'in-play' events only. True by default.
        successful_only (bool, optional): selection of whether to only include successful events. True by default

    Returns:
        bool: True = successful action into the box, nan = not box action, unsuccessful action or not a pass/carry.
    """

    # Determine if event is pass and check pass success
    if single_event['eventType'] in ['Pass', 'Carry']:

        # Check success (if successful_only = True)
        if successful_only:
            check_success = single_event['outcomeType'] == 'Successful'
        else:
            check_success = True

        # Check pass made in-play (if inplay = True)
        if inplay:
            check_inplay = not any(item in single_event['satisfiedEventsTypes'] for item in [48, 50, 51, 42, 44,
                                                                                             45, 31, 34, 212])
        else:
            check_inplay = True

        # Determine pass end position, and whether it's a successful pass into box
        x_position = single_event['x']
        y_position = single_event['y']
        x_position_end = single_event['endX']
        y_position_end = single_event['endY']

        # Check whether action moves ball into the box
        if (check_success and check_inplay) and (x_position_end >= 83) and (21.1 <= y_position_end <= 78.9) and \
                ((x_position < 83) or ((y_position < 21.1) or (y_position > 78.9))):
            return True
        else:
            return float('nan')

    else:
        return float('nan')
    


def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1, max_carry_duration=10):
    """ Add carry events to whoscored-style events dataframe

    Function to read a whoscored-style events dataframe (single or multiple matches) and return an event dataframe
    that contains carry information.

    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        min_carry_length (float, optional): minimum distance required for event to qualify as carry. 5m by default.
        max_carry_length (float, optional): largest distance in which event can qualify as carry. 60m by default.
        min_carry_duration (float, optional): minimum duration required for event to quality as carry. 2s by default.
        max_carry_duration (float, optional): longest duration in which event can qualify as carry. 10s by default.

    Returns:
        pandas.DataFrame: whoscored-style dataframe of events including carries
    """

    # Initialise output dataframe
    events_out = pd.DataFrame()

    # Carry conditions (convert from metres to opta)
    min_carry_length = min_carry_length
    max_carry_length = max_carry_length
    min_carry_duration = min_carry_duration
    max_carry_duration = max_carry_duration

    for matchId in events_df['matchId'].unique():

        match_events = events_df[events_df['matchId'] == matchId].reset_index()
        match_carries = pd.DataFrame()

        for idx, match_event in match_events.iterrows():

            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True

                while incorrect_next_evt:

                    next_evt = match_events.loc[next_evt_idx]

                    if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                        incorrect_next_evt = True

                    elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt[
                                'outcomeType'] == 'Unsuccessful')
                          or (next_evt['type'] == 'Foul')):
                        incorrect_next_evt = True

                    else:
                        incorrect_next_evt = False

                    next_evt_idx += 1

                # Apply some conditioning to determine whether carry criteria is satisfied

                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105*(match_event['endX'] - next_evt['x'])/100
                dy = 68*(match_event['endY'] - next_evt['y'])/100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']

                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period

                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt

                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                                prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                        (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(
                        ((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                         (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(
                        lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['matchId'] = nex['matchId']
                    carry['type'] = 'Carry'
                    carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2

                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

        match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
        match_events_and_carries = match_events_and_carries.sort_values(
            ['matchId', 'period', 'cumulative_mins']).reset_index(drop=True)

        # Rebuild events dataframe
        events_out = pd.concat([events_out, match_events_and_carries])

    return events_out


def get_xthreat(events_df, interpolate=True, pitch_length=100, pitch_width=100):
    """ Add expected threat metric to whoscored-style events dataframe

    Function to apply Karun Singh's expected threat model to all successful pass and carry events within a
    whoscored-style events dataframe. This imposes a 12x8 grid of expected threat values on a standard pitch. An
    interpolate parameter can be passed to impose a continous set of expected threat values on the pitch.

    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        interpolate (bool, optional): selection of whether to impose a continous set of xT values. True by default.
        pitch_length (float, optional): extent of pitch x coordinate (based on event data). 100 by default.
        pitch_width (float, optional): extent of pitch y coordinate (based on event data). 100 by default.

    Returns:
        pandas.DataFrame: whoscored-style dataframe of events, including expected threat
    """

    # Define function to get cell in which an x, y value falls
    def get_cell_indexes(x_series, y_series, cell_cnt_l, cell_cnt_w, field_length, field_width):
        xi = x_series.divide(field_length).multiply(cell_cnt_l)
        yj = y_series.divide(field_width).multiply(cell_cnt_w)
        xi = xi.astype('int64').clip(0, cell_cnt_l - 1)
        yj = yj.astype('int64').clip(0, cell_cnt_w - 1)
        return xi, yj

    # Initialise output
    events_out = pd.DataFrame()

    # Get Karun Singh expected threat grid
    path = "https://karun.in/blog/data/open_xt_12x8_v1.json"
    xt_grid = pd.read_json(path)
    init_cell_count_w, init_cell_count_l = xt_grid.shape

    # Isolate actions that involve successfully moving the ball (successful carries and passes)
    move_actions = events_df[(events_df['eventType'].isin(['Carry', 'Pass'])) &
                             (events_df['outcomeType'] == 'Successful')]

    # Set-up bilinear interpolator if user chooses to
    if interpolate:
        cell_length = pitch_length / init_cell_count_l
        cell_width = pitch_width / init_cell_count_w
        x = np.arange(0.0, pitch_length, cell_length) + 0.5 * cell_length
        y = np.arange(0.0, pitch_width, cell_width) + 0.5 * cell_width
        interpolator = interp2d(x=x, y=y, z=xt_grid.values, kind='linear', bounds_error=False)
        interp_cell_count_l = int(pitch_length * 10)
        interp_cell_count_w = int(pitch_width * 10)
        xs = np.linspace(0, pitch_length, interp_cell_count_l)
        ys = np.linspace(0, pitch_width, interp_cell_count_w)
        grid = interpolator(xs, ys)
    else:
        grid = xt_grid.values

    # Set cell counts based on use of interpolator
    if interpolate:
        cell_count_l = interp_cell_count_l
        cell_count_w = interp_cell_count_w
    else:
        cell_count_l = init_cell_count_l
        cell_count_w = init_cell_count_w

    # For each match, apply expected threat grid (we go by match to avoid issues with identical event indicies)
    for matchId in move_actions['matchId'].unique():
        match_move_actions = move_actions[move_actions['matchId'] == matchId]

        # Get cell indices of start location of event
        startxc, startyc = get_cell_indexes(match_move_actions['x'], match_move_actions['y'], cell_count_l,
                                            cell_count_w, pitch_length, pitch_width)
        endxc, endyc = get_cell_indexes(match_move_actions['endX'], match_move_actions['endY'], cell_count_l,
                                        cell_count_w, pitch_length, pitch_width)

        # Calculate xt at start and end of eventa
        xt_start = grid[startyc.rsub(cell_count_w - 1), startxc]
        xt_end = grid[endyc.rsub(cell_count_w - 1), endxc]

        # Build dataframe of event index and net xt
        ratings = pd.DataFrame(data=xt_end-xt_start, index=match_move_actions.index, columns=['xThreat'])

        # Merge ratings dataframe to all match events
        match_events_and_ratings = pd.merge(left=events_df[events_df['matchId'] == matchId], right=ratings,
                                            how="left", left_index=True, right_index=True)
        events_out = pd.concat([events_out, match_events_and_ratings], ignore_index=True, sort=False)
        events_out['xThreat_gen'] = events_out['xThreat'].apply(lambda xt: xt if (xt > 0 or xt != xt) else 0)

    return events_out
