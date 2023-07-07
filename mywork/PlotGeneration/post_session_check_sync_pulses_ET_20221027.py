#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, February 21st 2019, 7:13:37 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 21-02-2019 07:35:12.1212


# In[3]:


import os
os.getcwd() 


# In[2]:


from pathlib import Path
import ibllib.io.raw_data_loaders as raw
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd



def get_port_events(events: dict, name: str = '') -> list:
    out: list = []
    for k in events:
        if name in k:
            out.extend(events[k])
    out = sorted(out)

    return out

basePath = r'C:\iblrig_data\Subjects//'
# m503 = 'MBOT77_503'
# m513 = 'MBOT77_513'
# day = 23
# date = '2023-06-' + day
# session = '007'
# top is for locally stored files, bottom is for files stored on the server 
# session_path = basePath + subject + '//' + date + '//' + session
#session_path = serverPath + subject + '//' + date + '//' + session

#local test mouse path: r'C:\iblrig_data\Subjects\_iblrig_test_mouse\2022-09-31\004//'
#synapse path: r'Z:\moraogando\IBLdata\Box1_basement\MBOT42_1840\2022-08-23\002//'
#local path:  r'C:\iblrig_data\Subjects\MBOT42_1840\2022-08-19\005//'


# In[16]:


def render_sync_image(date: str, subject: str, session: str, path=r'Z:\moraogando\IBLdata\Current_Mice_(Mike)//') -> None:
    session_path = path + subject + '//' + date + '//' + session
    
    
    session_data_file = Path(session_path) #getting path to session folder
    if not session_data_file.exists():  #check if the directory (folder) exists
        raise (FileNotFoundError)
    data = raw.load_data(session_data_file) #trial-wise list of dictionaries containing info from each trial
    unsynced_trial_count = 0 #setting initial unsynced trial count to 0
    frame2ttl = [] #empty list, which we will add to the times when a frametottl signal is sent
    sound = [] #empty list, which we will add to the times when a sound signal is sent
    camera = [] #empty list, which we will add to the times when a camera signal is sent
    trial_end = [] #empty list, which we will add to the times when a trial ends
    trial_start = []
    for trial_data in data: #for loop: for every trial, get the port events of bnc1, bnc2, port1.  if no event recorded in any of these ports, report an unsynced trial
        tevents = trial_data['behavior_data']['Events timestamps']
        ev_bnc1 = get_port_events(tevents, name='BNC1')
        ev_bnc2 = get_port_events(tevents, name='BNC2')
        ev_port1 = get_port_events(tevents, name='Port1')
        if not ev_bnc1 or not ev_bnc2 or not ev_port1:
            unsynced_trial_count += 1
        frame2ttl.extend(ev_bnc1)
        sound.extend(ev_bnc2)
        camera.extend(ev_port1)
        trial_end.append(trial_data['behavior_data']['Trial end timestamp'])
        trial_start.append(trial_data['behavior_data']['Trial start timestamp'])
    print(f'Found {unsynced_trial_count} trials with bad sync data for Box 1')
    f = plt.figure()  #figsize=(19.2, 10.8), dpi=100)
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax.plot(camera, np.ones(len(camera)) * 1, '|')
    ax.plot(sound, np.ones(len(sound)) * 2, '|')
    ax.plot(frame2ttl, np.ones(len(frame2ttl)) * 3, '|')
    [ax.axvline(t, alpha=0.5) for t in trial_end]
    ax.set_ylim([0, 4])
    ax.set_yticks(range(4))
    ax.set_yticklabels(['', 'camera', 'sound', 'frame2ttl'])
    plt.show() #this plots the bpod times and whether a signal was received at that time by the frame2ttl, sound, and camera



if __name__ == "__render_sync_image__":
    render_sync_image(date, sunject, session)