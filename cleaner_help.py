######################################################################
#                                                                    #
# Contains a series of helper functions intended to clean dataframes #
#                                                                    #
######################################################################


import pandas as pd
import numpy as np
import json
from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def person_fill_na(p, transcipt_int,offer_to_time):
    
    """
    Takes in a transcript for a customer and determines what transactions are made because of offer.
    The offer must have been seen to count
    The window of influence is from time seen to time completed
    The first offer to be seen is the one that governs.
    
    Inputs:
    - person (int): an identifier for a person
    - transcript(pandas): dataframe with columns aptly named person, time, offer_id, offer received, offer viewed
    - offer_to_time(dict): maps an offer id to a time duration
    
    Output:
    - person_df_copy(pandas): a returned copy of the original dataframe with offer id given to transactions
    - user_time_stats(pandas): time stats for user and offers
    
    """
    
    person_df_copy = transcipt_int[transcipt_int['person'] == p].copy()
    
    #get idx where offer received
    offer_received_idx = person_df_copy[person_df_copy['offer received'] == 1].index

    #list to store sliced dataframes
    df_list = []

    
    #store slices
    for v, w in pairwise(offer_received_idx):
        df_list.append(person_df_copy.loc[v:w])
        
        
    #list to store user stats
    user_stats = []
    
    for df in df_list:
        #get start time
        time_start = df.iloc[0,:]['time']
        #get offer id
        offer_id = df.iloc[0,:]['offer_id']
        #get duration of offer
        offer_time = offer_to_time[offer_id]
        #time at end of offer
        time_end = time_start + offer_time
        #get idx at which offer seen
        offer_seen_idx = df[df['offer viewed']==1].index
        #get offer completion idx
        offer_complete_idx = df[(df['offer completed']==1) & (df['offer_id']==offer_id)].index
        
        
        ### record user stats ###
        #time to open as fraction of duration
        if offer_seen_idx.empty:
            time_to_open = np.nan
        else:
            time_to_open = (df.loc[offer_seen_idx[0]]['time'] - time_start)/offer_time
        
        #time to complete as fraction of duration
        if offer_complete_idx.empty:
            time_to_complete = np.nan
        else:
            time_to_complete = (df.loc[offer_complete_idx[0]]['time'] - time_start)/offer_time
        
        #make note of interaction
        user_offer_stats = {'person': p,
                            'offer_id': offer_id,
                            'time_to_open': time_to_open, 'time_to_complete': time_to_complete}
        #append to stats
        user_stats.append(user_offer_stats)
        
        
        #check if offer actually seen
        if not offer_seen_idx.empty:
            #get idx where it was seen
            offer_seen_idx = offer_seen_idx[0]
            #record time seen
            time_seen = df.loc[offer_seen_idx]['time']
            #get influence window
            infl_df = df[(df['time']>=time_seen) & (df['time']<=time_end)]
            #cut off the window when offer is complete
            if not offer_complete_idx.empty:
                infl_df=infl_df.loc[:offer_complete_idx[0]]
            #locationn of purchases made due to offer
            missing_offer_idx = infl_df[infl_df['offer_id'].isna()].index
            #replace in original dataframe
            person_df_copy.loc[missing_offer_idx, 'offer_id'] = offer_id
            
    user_time_stats = pd.DataFrame(user_stats)

    return person_df_copy, user_time_stats


def transcript_fill_na(transcript_int, offer_to_time):
    
    """
    Returns a filled transcript dataframe
    
    Inputs:
    - transcript(pandas): a pandas dataframe
    - offer_to_time(dict): a dictionary that maps offer to duration
    Returns:
    - transcript_full(pandas): dataframe with nans filled with corresponding values
    - transcript_time(pandas): dataframe with time stats for users & offers
    """
    
    #get unique users
    persons = transcript_int['person'].unique()
    #store outputs here
    person_list = []
    time_list = []

    for per in persons:
        person_df_copy, user_time_stats = person_fill_na(per, transcript_int, offer_to_time)
        
        person_list.append(person_df_copy)
        time_list.append(user_time_stats)
        
        
    transcript_full = pd.concat(person_list)
    transcript_time = pd.concat(time_list)
    
    #reorder columns
    transcript_full = transcript_full[['person', 'offer_id',
                                       'time', 'offer received',
                                       'offer viewed', 'offer completed',
                                       'transaction', 'reward', 'amount']]
    
    #more roeordering
    transcript_time = transcript_time[['person', 'offer_id',
                                       'time_to_open', 'time_to_complete']]
    
    return transcript_full, transcript_time
        

    
def transcript_cleaner(transcript, offer_to_int, offer_to_time, human_to_int):
    """
    Performs cleaning steps on transcript with columns:
    - event (str) - record description (ie transaction, offer received, offer viewed, etc.)
    - person (str) - customer id
    - time (int) - time in hours since start of test. The data begins at time t=0
    - value - (dict of strings) - either an offer id or transaction amount depending on the record
    
    Inputs:
    - transcript(pandas): dataframe with columns:
        - event (str) - record description (ie transaction, offer received, offer viewed, etc.)
        - person (str) - customer id
        - time (int) - time in hours since start of test. The data begins at time t=0
        - value - (dict of strings) - either an offer id or transaction amount depending on the record
    - offer_to_int(dict): a dictionary that maps offer to an integer
    - offer_to_time(dict): a dictionary that maps offer to duration
    - human_to_int(dict): a dictionary that maps a user to an int
    
    
    Output:
    transcript_full(pandas): dataframe with nans filled with corresponding values
    transcript_time(pandas): dataframe with time stats for users & offers
    """

    
    #apply mapping
    transcript_human_int = transcript.copy()
    transcript_human_int['person'] = transcript_human_int['person'].map(human_to_int)
    
    #get dummies for events
    event_dum = pd.get_dummies(transcript_human_int['event'])
    #reorder columns
    event_dum = event_dum[['offer received', 'offer viewed', 'offer completed', 'transaction']]

    #explode dicts keys and values to dataframe
    transcipt_series = pd.concat([transcript_human_int.drop(['value'], axis=1),
                                  transcript_human_int['value'].apply(pd.Series)], axis=1)
    
    #combine the two
    transcipt_dum = pd.concat([transcipt_series.drop(columns=['event']), event_dum], axis = 1)
    #fill up offer_id with "offer id"
    transcipt_dum['offer_id'] = transcipt_dum['offer_id'].fillna(transcipt_dum['offer id'])
    #drop the unecessary column
    transcipt_dum = transcipt_dum.drop(columns=['offer id'])
    
    #map offers to integers
    transcipt_int = transcipt_dum.copy()
    transcipt_int['offer_id'] = transcipt_int['offer_id'].map(offer_to_int)
    
    transcript_full, transcript_time = transcript_fill_na(transcipt_int, offer_to_time)
    
    return transcript_full, transcript_time

