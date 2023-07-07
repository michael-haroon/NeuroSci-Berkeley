#!/usr/bin/env python
# coding: utf-8

# ### LOAD AND PLOT DATA FROM A SESSION #### 
# This notebook should run from within the iblenv.
# The functions are taken from the source code of the IBL library. 
# 
# For documentation check:
# 
# https://int-brain-lab.github.io/iblenv/_autosummary/ibllib.io.raw_data_loaders.html?highlight=raw%20data%20loaders#module-ibllib.io.raw_data_loaders
# 
# 

# In[1]:


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Mora Ogando
# @Date: Sunday, July 4th 2021, 1:28:46 pm
"""

Raw Data Loader functions for PyBpod rig

Module contains one loader function per raw datafile
"""
#import things
import json
import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

import itertools
from glob import glob
import pickle

import numpy as np
import pandas as pd

#from ibllib.io import jsonable
from ibllib.io import raw_data_loaders
from ibllib.misc import version
# Helpers Mori

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def appendColumnToDF(df,OneDArray,columnName,appendTO):
# the number of elements in the 1dArray and the number of elements in the appendTO column (i.e.: 'cell') has to match.
    # 1. create a pandas series with your array
    series = pd.Series(list(OneDArray),name=columnName)
    # 2. create the index (has to match the index in "appendTO")
    index= pd.Series(np.arange(0,len(OneDArray)),name=appendTO)
    # 3. make a DataFrame with these 
    appendDF = pd.DataFrame(series,index)
    
    #4. join it onto a new dataframe
    try:
        df.reset_index(inplace=True)
    except:
        pass
    df2 = df.set_index([appendTO])
    df2 = df.join(appendDF,on=appendTO,how='left')
    return df2

def saveTrialWiseInfo_inServer(session_path):
    #Load data 
    session_data = raw_data_loaders.load_data(session_path)
    settings =raw_data_loaders.load_settings(session_path)
    encoder_events = raw_data_loaders.load_encoder_events(session_path, settings=False)
    encoder_positions =raw_data_loaders.load_encoder_positions(session_path, settings=False)
    encoder_trial_info = raw_data_loaders.load_encoder_trial_info(session_path)

    # correct responses is a boolean list of len = number of trials.
    correct_responses = [session_data[i]['trial_correct'] for i in range(len(session_data))]

    # append correct responses to trial info

    # ((the number of elements in the 1dArray and the number of elements in the appendTO column (i.e.: 'cell') has to match.))
    OneDArray = correct_responses
    columnName = 'correct'
    appendTO = 'trial_num'
    trialInfo= appendColumnToDF(encoder_trial_info,OneDArray,columnName,appendTO)

    # get the time in the same units and merge the encoder position with the trialInfo dataFrames
    trialInfo['relative_time_ms'] = (pd.to_datetime(trialInfo.bns_ts).values -pd.to_datetime(trialInfo.bns_ts).values[0])/1000000
    trialInfo['relative_time_ms'] = trialInfo.relative_time_ms.astype(np.int64) 
    encoder_positions['relative_time_ms'] = (encoder_positions.re_ts.values- encoder_positions.re_ts.values[0])/1000
    merged_df = pd.merge(trialInfo, encoder_positions, on = 'relative_time_ms',how='outer')

    # save

    name = 'behavior_analysis_df'

    merged_df.to_csv(path + name + '.csv')
    out_path = r'W:\Satsuma\Mora//' + path.split('C:\\')[-1]
    out_path 
    import os
    try:
        os.makedirs(out_path)
        print('created a new results folder')
    except:
        print('couldnt make folder')
        print(out_path)

    merged_df.to_csv(out_path + name + '.csv')
    trialInfo.to_csv(out_path + 'trialInfo.csv')
    return  merged_df,trialInfo





def render_plots(date: str, subject: str, session: str, path=r'Z:\moraogando\IBLdata\Current_Mice_(Mike)//'):
    
  

    session_path = path + subject + '//' + date + '//' + session

    #Load data 
    session_data = raw_data_loaders.load_data(session_path)
    settings =raw_data_loaders.load_settings(session_path)
    encoder_events = raw_data_loaders.load_encoder_events(session_path, settings=False)
    encoder_positions =raw_data_loaders.load_encoder_positions(session_path, settings=False)
    encoder_trial_info = raw_data_loaders.load_encoder_trial_info(session_path)

    # correct responses is a boolean list of len = number of trials.
    correct_responses = [session_data[i]['trial_correct'] for i in range(len(session_data))]

    # append correct responses to trial info

    # ((the number of elements in the 1dArray and the number of elements in the appendTO column (i.e.: 'cell') has to match.))
    OneDArray = correct_responses
    columnName = 'correct'
    appendTO = 'trial_num'
    trialInfo= appendColumnToDF(encoder_trial_info,OneDArray,columnName,appendTO)

    # # get the time in the same units and merge the encoder position with the trialInfo dataFrames
    trialInfo['relative_time_ms'] = (pd.to_datetime(trialInfo.bns_ts).values -pd.to_datetime(trialInfo.bns_ts).values[0])/1000000
    trialInfo['relative_time_ms'] = trialInfo.relative_time_ms.astype(np.int64) 
    encoder_positions['relative_time_ms'] = (encoder_positions.re_ts.values- encoder_positions.re_ts.values[0])/1000
    merged_df = pd.merge(trialInfo, encoder_positions, on = 'relative_time_ms',how='outer')



    window = 50# how many trials to average 
    firsttrial=0
    lasttrial=4000
    print('fraction correct for session ' + 'was ' + str(np.mean(correct_responses[:lasttrial])))
    # get the info of which angle was shown at each trial 
    stim_angle= [session_data[i]['stim_angle'] for i in range(len(session_data[:lasttrial]))]

    deg90 = np.where(np.array(stim_angle)>0)[0][firsttrial:lasttrial] # array of trial numbers where the angle was more than 0
    deg0= np.where(np.array(stim_angle)<=0)[0][firsttrial:lasttrial] # array of trial numbers where the angle was 0
    correct90 = [correct_responses[i] for i in deg90]#@[firsttrial:lasttrial]
    correct0 = [correct_responses[i] for i in deg0]#[firsttrial:lasttrial]

    #get the % of correct responses for each angle across the whole session
    print('fraction correct for angle 0 = ' + str(np.nanmean(correct0)))
    print('fraction correct for angle 90 = ' + str(np.mean(correct90)))


    # plot a moving average of the correct responses across trials for a given window
    plt.plot(moving_average(correct_responses[firsttrial:lasttrial],window))
    plt.title('Correct choices vs trial number',fontsize = 20)
    plt.xlabel('trial number',fontsize = 15)
    plt.ylabel('correct responses',fontsize = 15)

    plt.ylim(0,1)

    window = 20
    plt.plot(moving_average(correct90,window))
    plt.plot(moving_average(correct0,window))
    plt.title('Correct choices vs trial number for each angle',fontsize = 20)
    plt.xlabel('trial number',fontsize = 15)
    plt.ylabel('correct resposes',fontsize = 15)
    plt.ylim(0,1.1)
    xmax = len(correct90)+len(correct0)
    plt.hlines(0.5, 0, xmax, colors='black',linestyles='dashed')
    plt.legend(['moving average']+['angle =  ' + str(i) for i in [-45,45]])
    plt.savefig(session_path + '//Correct_vs_trial_per_angle')

    percent_left = (len(correct0) + (len(deg90)-len(correct90)))/(len(deg0)+len(deg90))
    abs_bias = np.abs(0.5 - percent_left)

    plt.show()

    print('\nPercent Left Choices:')
    print(percent_left)
    print('Percent Absolute Bias:')
    print(abs_bias)


if __name__ == '__plots__':
    plots(date, sunject, session)

