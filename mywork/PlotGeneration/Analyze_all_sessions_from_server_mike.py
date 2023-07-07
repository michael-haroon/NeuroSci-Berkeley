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

# In[2]:


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
import seaborn as sns
import os
import itertools
from glob import glob
import pickle
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from ibllib.io import raw_data_loaders
from ibllib.misc import version
# Helpers Mori

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# # TODO:
# 
# ## Mean session performance as a function of the session number, color coded by the "stage" of training (habituation, repeat on error ON, repeat on error OFF, contrasts) for each mouse.
# ## Mean session performance as a function of the session number for all mice together, color code mice by training cohort (we're now in our third cohort).
# ## Response time as a function of contrast for the multicontrast phase.

# # Find good sessions for each mouse

# In[77]:


# find data
basePath  = r'Z:\moraogando\IBLdata\Box1_basement//'
#basePath  = r'Z:\moraogando\IBLdata\Box2_basement//'
#basePath = r'C:\iblrig_data\Subjects//'
basePath  = r'Z:\moraogando\IBLdata\Current_Mice_(Mike)//'
subdirs = glob(r'Z:\moraogando\IBLdata\Current_Mice_(Mike)//MBO*')
#subdirs = glob(r'C:\iblrig_data\Subjects//MBO*')

AllMice = [s.split('\\')[-1] for s in subdirs]

mouseList = ['MBOT77_513', 'MBOT77_503', 'MBOT77_505', 'MBOT88_777']

allSessionsMice = []
imouse = 0

for imouse in range(0,len(mouseList)):
    fullpath = basePath+mouseList[imouse ]+'//*//*//raw_behavior_data//_iblrig_encoderTrialInfo.raw.ssv'
    allsessions = glob(fullpath,recursive=True)

    # preallocate the info for each file
    filePath = []
    mouse = []
    date = []
    session = []
    size = [] # ---> important, get the size of the file so that we can filter by that later on


    # loop over the file name to extract the relevant info
    for thisData in allsessions:
        filePath.append(thisData)
        mouse.append( thisData.split(basePath)[-1].split('\\')[0])
        date.append(thisData.split(basePath)[-1].split('\\')[1])
        session.append(thisData.split(basePath)[-1].split('\\')[2])
        size.append((os.path.getsize(thisData))/1000)

        print(f' DataPath: {thisData} , \nMouse: {mouse[-1]},Date:{date[-1]},session: {session[-1]}, fileSize:{size[-1]}')

    # Create a dictionary with the info for this mouse
    ThisDict = {'DataPath': filePath,'Mouse': mouse,'Date':date,'session': session, 'fileSize (MB)':size}
    # Turn it into a DF
    MouseTrainingDF = pd.DataFrame(ThisDict)

    # Get the largest file for each date (We assume that's the one we want to analyze)
    BestSessionDay = MouseTrainingDF.groupby(['Date']).max().reset_index()
    print(f'this mouse trained for a total of {len(BestSessionDay)} sessions')

    allSessionsMice.append(BestSessionDay)


# In[78]:


AllSessionsAllMice = pd.concat(allSessionsMice,axis=0).reset_index()
print(f'total sessions: {len(AllSessionsAllMice)}')


# # Load the good sessions (20220216) continue here

# In[79]:


AllDFs =[]

for nyn in range(len(allSessionsMice)):
    AllBestSessionDay = allSessionsMice[nyn]
    for isession in range(len(AllBestSessionDay)):
        # preallocate the info for each file
        filePath =AllBestSessionDay.loc[isession,'DataPath']
        mouse =AllBestSessionDay.loc[isession,'Mouse']
        date =  AllBestSessionDay.loc[isession,'Date']
        session =  AllBestSessionDay.loc[isession,'session']
        size = AllBestSessionDay.loc[isession,'fileSize (MB)']

        print(f'computing session # {isession},\n {filePath}')

        # load the data
        session_path =filePath
        session_path = session_path.split('\\raw_behavior_data')[0]

        try:
            print('loading data ... ')
            #Load data 
            session_data = raw_data_loaders.load_data(session_path)
            settings =raw_data_loaders.load_settings(session_path)
            trialInfoDF = raw_data_loaders.load_encoder_trial_info(session_path)
            
            print('Data loaded! ... ')

            # correct responses is a boolean list of len = number of trials.
            correct_responses = [session_data[i]['trial_correct'] for i in range(len(session_data))]

            # in case the last trial wasn't saved in the encoder_trial_info
            if ~(len(trialInfoDF) == len(correct_responses)):
                trialInfoDF = trialInfoDF[:len(correct_responses)]

            assert (len(trialInfoDF) == len(correct_responses))

            # append correct responses
            trialInfoDF.loc[:,'correct'] = correct_responses

            # get trials with left responses
            Left1 = list(np.where((trialInfoDF['stim_angle'].values<=0)& (trialInfoDF['correct'].values>0))[0]) # option 1, angle less or = 0 and correct response
            Left2 = list(np.where((trialInfoDF['stim_angle'].values>0)& (trialInfoDF['correct'].values==0))[0]) # option 2, angle bigger than 0 and incorrect response

            # join the 2 lists and add 1 to correct for the trialnum indexing starting at 1
            lefresponses =np.array(Left1 + Left2)+1

            trialInfoDF.loc[:,'LeftTurn'] = False

            # assign the left responses to the corresponding trials
            trialInfoDF.loc[trialInfoDF.trial_num.isin(lefresponses),'LeftTurn'] = True

            # append signed contrasts
            trialInfoDF.loc[:,'signedContrast'] = trialInfoDF.stim_contrast.values*trialInfoDF.stim_pos_init.values

            # calculate left bias (% of left responses overal)
            bias = sum(trialInfoDF.LeftTurn.values)/len(trialInfoDF.LeftTurn.values)
            # calculate the mean performance (sum of correct trials over total trials)
            performance = sum(correct_responses)/len(correct_responses)

            Contrasts = trialInfoDF.stim_contrast.unique()
            print(f'The mean performance was {performance}, the bias was {bias}, the contrasts were {Contrasts}')

            # add all the rest of the info to the DF
            trialInfoDF.loc[:,'filePath']=filePath
            trialInfoDF.loc[:,'mouse']=mouse
            trialInfoDF.loc[:,'date']=date
            trialInfoDF.loc[:,'session']=session
            trialInfoDF.loc[:,'fileSize']=size
            trialInfoDF.loc[:,'NcontrastsDisplayed']=len(Contrasts)
            trialInfoDF.loc[:,'meanPerformance']=performance
            trialInfoDF.loc[:,'Leftbias']=bias
            trialInfoDF.loc[:,'AbsoluteBias']=np.abs(bias-0.5)
            trialInfoDF.loc[:,'trainingDay']=isession
            trialInfoDF.loc[:,'REPEAT_ON_ERROR']=settings['REPEAT_ON_ERROR']
            trialInfoDF.loc[:,'INTERACTIVE_DELAY']=settings['INTERACTIVE_DELAY']
            trialInfoDF.loc[:,'ITI_CORRECT']=settings['ITI_CORRECT']
            trialInfoDF.loc[:,'ITI_ERROR']=settings['ITI_ERROR']
            trialInfoDF.loc[:,'is_good_session'] = True
            trialInfoDF.loc[:,'STIM_PROBABILITY_LEFT'] = settings['STIM_PROBABILITY_LEFT']
            
            #tracking correct response as boolean
            correct_responses_boolean = [d['trial_correct'] for d in session_data]
            correct_boolean = [i for i in correct_responses_boolean]
            window = 1
            firsttrial = 0
            lasttrial = 10000
#             li = [0 for i in range(window-1)] 
#             moving_avga = moving_average(correct_boolean[firsttrial:len(correct_boolean)],window)
#             li.extend(moving_avga)
#             moving_avg = li
            movingPerformance = [np.nan]*(window) + list(moving_average(correct_responses,window))[:-1]#EXCLUDE the current trial in the performance criteria
            trialInfoDF.loc[:,'moving_avg'] = movingPerformance
            trialInfoDF.loc[:,'correct_boolean'] = correct_boolean

        except:
            print('couldnt process the data, appending dummy DF')
            nn = [0]
            
            dummyDict = { 'trial_num': nn, 'stim_pos_init':nn, 'stim_contrast':nn, 'stim_freq':nn,
           'stim_angle': nn, 'stim_gain': nn, 'stim_sigma': nn, 'stim_phase': nn, 'bns_ts': nn,
           'correct': nn, 'LeftTurn': nn, 'signedContrast': nn, 'filePath': nn, 'mouse':[mouse], 'date':[date],
           'session':[session], 'fileSize':[size], 'contrastsDisplayed': nn, 'meanPerformance': nn,
           'Leftbias': nn, 'AbsoluteBias': nn, 'trainingDay': nn, 'REPEAT_ON_ERROR': nn,
           'INTERACTIVE_DELAY': nn, 'ITI_CORRECT': nn, 'ITI_ERROR': nn }
            
            trialInfoDF = pd.DataFrame(dummyDict)
            trialInfoDF.loc[:,'is_good_session'] = False




        AllDFs.append(trialInfoDF)


# In[80]:


concatDF = pd.concat(AllDFs)
concatDF.head()


# In[81]:


concatDF


# In[82]:


#Convert the concatDF to a CSV, uncomment to run

#name = '20220517_allSessions'
#concatDF.to_csv(r'C:\Users\behavior\Desktop\moriBehaviorScripts\outputData//' + name + '.csv')
#name = '2_Good_allSessions'
#concatDF.to_csv(r'C:\dataAnalysis\Pool_and_Plot\Erin_Mice_Data//' + name + '.csv')


# # Plot Mice Performance across SESSIONS

# In[83]:


#Plot of ALL MICE performance across sessions
x ='trainingDay'
y = 'meanPerformance'
hue= 'mouse'


data = concatDF

conds = (data.trainingDay>0)
data = data[conds]
palette = sns.color_palette('BuGn',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (8,5))

ax = sns.pointplot(data = data,x=x,y=y,hue=hue,palette= palette)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Learning curve for all mice',fontsize = 15)
ax.set_xlabel('Training session',fontsize = 15)
ax.set_ylabel('Mean performance',fontsize = 15)
minX = data.trainingDay.min()
maxX = data.trainingDay.max()
#ax.set_xticklabels(labels =np.round(np.arange(minX,maxX,5),1),fontsize = 15)
ax.xaxis.set_major_locator(ticker.LinearLocator(10))
ax.set_xticklabels(labels =np.round(np.arange(minX,maxX,5),0),fontsize = 15)
ax.axes.hlines(0.5,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')

plt.show()


# In[170]:


#Code to split training stages for ONE MOUSE
x ='trainingDay'
y = 'meanPerformance'
hue= 'mouse'


data = concatDF

conds = (data.trainingDay>0)
data = data[conds]
palette = sns.color_palette('BuGn',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (8,5))

ax = sns.pointplot(data = data,x=x,y=y,hue=hue,palette= palette)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Training Stages',fontsize = 15)
ax.set_xlabel('Training session',fontsize = 15)
ax.set_ylabel('Mean performance',fontsize = 15)
#minX = data.trainingDay.min()
minX = data.trainingDay.min() - 1
maxX = data.trainingDay.max()
#ax.set_xticklabels(labels =np.round(np.arange(minX,maxX,5),1),fontsize = 15)
ax.xaxis.set_major_locator(ticker.LinearLocator(10))
ax.set_xticklabels(labels =np.round(np.arange(minX,maxX,5),0),fontsize = 15)
ax.axes.hlines(0.5,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')

#defining learning stage
perf = concatDF.meanPerformance.unique()
for i in range(len(perf)-2): #can only search from session 1 to (total sessions - 2)
    if perf[i] <= 0.5 and perf[i+1] <= 0.5 and perf[i+2] <= 0.5:
        perf = i-1
        break
ax.axes.vlines(perf,0,1,linestyles='dashed',color='purple')

#defning expert stage
arr_perf = []
perf = concatDF.meanPerformance.unique()
for i in range(len(perf)-2):
    if perf[i] >= 0.7 and perf[i+1] >= 0.7 and perf[i+2] >= 0.7:
        arr_perf =arr_perf + [i-1]
for i in range(len(arr_perf)):
    if len(concatDF[concatDF['trainingDay'] == arr_perf[i]].REPEAT_ON_ERROR.unique()) == 1 and concatDF[concatDF['trainingDay'] == arr_perf[i]].REPEAT_ON_ERROR.unique()[0] == 0:
        perf = arr_perf[i]
        break
ax.axes.vlines(perf,0,1,linestyles='dashed',color='red')

#defining contrast stage
array = concatDF[concatDF['stim_contrast'] < 1.0000].trainingDay.unique()
ax.axes.vlines(array[0],0,1,linestyles='dashed',color='blue')

plt.show()


# In[301]:


#pd.set_option('display.max_columns', None)
#nan_rows = concatDF[concatDF.isna().any(axis=1)]
#nan_rows


# In[ ]:


#divide training stages
#Early stage: everything before
#Learning stage: 1st session of 3 consecutive sessions of consistent performance <= 50% (mice are trying to learn what is wrong and right, so performance takes a dive)
#Advanced stage: 1st session of 3 consecutive sessions of consistent performance >= 70% with repeat on error OFF
#Contrasts stage: the session when the first contrast besides 100% is introduced


# In[157]:


x ='stim_pos_init'
y = 'LeftTurn'
hue= 'mouse'
minTrial = 30
maxTrial = 1000


data = concatDF

conds = (data.meanPerformance>0.75)& (data.trial_num>minTrial)& (data.trial_num<maxTrial) & ~(data.mouse.isin(['MBOT22_143','MBOT24_478']))
data = data[conds]
palette = sns.color_palette('viridis',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (3,5))
order = [1,0.5,0.25,0,-0.25,-0.5,-1]
ax = sns.pointplot(data = data,x=x,y=y,hue=hue,palette= palette,aspect = 1.5,dodge=0.2)
ax = sns.pointplot(data = data,x=x,y=y,color='black',aspect = 1.5,dodge=0.2)
#ax = sns.pointplot(data = data,x=x,y=y)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Psychometric function',fontsize = 15)
ax.set_xlabel('Signed contrast',fontsize = 15)
ax.set_ylabel('Probability left turn',fontsize = 15)
ax.set_ylim(0,1)
#ax.set_xticklabels([-45,45],fontsize = 15)
ax.set_yticklabels(labels =np.round(np.arange(0,1,0.2),2),fontsize = 15)
ax.axes.hlines(0.5,0,1,linestyles='dashed',color='black')
# ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
plt.show()


# In[ ]:


x ='signedContrast'
y = 'LeftTurn'
hue= 'mouse'
minTrial = 30
maxTrial = 1000


data = concatDF

conds = (data.NcontrastsDisplayed>1)&(data.meanPerformance>0.65)& (data.trial_num>minTrial)& (data.trial_num<maxTrial) & ~(data.mouse.isin(['MBOT22_143','MBOT24_478']))
data = data[conds]
palette = sns.color_palette('viridis',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (3,5))
order = [1,0.5,0.25,0,-0.25,-0.5,-1]
#ax = sns.pointplot(data = data,x=x,y=y,hue=hue,palette= palette,order = order,alpha = 0.5,aspect = 1.5,dodge=0.2)
ax = sns.pointplot(data = data,x=x,y=y,color='black',order = order,aspect = 1.5,dodge=0.2,linewidtth=4)
#ax = sns.pointplot(data = data,x=x,y=y)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Psychometric function',fontsize = 15)
ax.set_xlabel('Signed contrast',fontsize = 15)
ax.set_ylabel('Probability left turn',fontsize = 15)
ax.set_ylim(0,1)
#ax.set_xticklabels([-45,45],fontsize = 15)
ax.set_yticklabels(labels =np.round(np.arange(0,1,0.2),2),fontsize = 15)
ax.axes.hlines(0.5,0,6,linestyles='dashed',color='black')
# ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
plt.show()


# In[ ]:


x ='signedContrast'
y = 'LeftTurn'
hue= 'mouse'
minTrial = 0
maxTrial = 1000


data = concatDF
order = [1,0.5,0.25,0,-0.25,-0.5,-1]
conds = (data.NcontrastsDisplayed>1)&(data.meanPerformance>0.0)& (data.trial_num>minTrial)& (data.trial_num<maxTrial) & ~(data.mouse.isin(['MBOT22_143','MBOT24_478']))
data = data[conds]
palette = sns.color_palette('viridis',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (5,5))

#ax = sns.regplot(data = data,x=x,y=y)
ax = sns.pointplot(data = data,x=x,y=y,hue='mouse',palette=palette,order = order,aspect = 1,dodge=0.2)
#ax = sns.pointplot(data = data,x=x,y=y)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Psychometric function',fontsize = 15)
ax.set_xlabel('Signed contrast',fontsize = 15)
ax.set_ylabel('Probability left turn',fontsize = 15)
ax.set_ylim(0,1.2)
# #ax.set_xticklabels([-45,45],fontsize = 15)
# ax.set_yticklabels(labels =np.round(np.arange(0,1,0.2),2),fontsize = 15)
# ax.axes.hlines(0.5,0,6,linestyles='dashed',color='black')
# ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')

analysisDir = r'C:\Users\behavior\Desktop\moriBehaviorScripts\outputData//'
plt.savefig(analysisDir + 'PsychometricFunctionsExperts.pdf')
plt.savefig(analysisDir + 'PsychometricFunctionsExperts.png')
plt.show()


# In[ ]:


x ='stim_contrast'
y = 'meanPerformance'
hue= 'mouse'
minTrial = 30
maxTrial = 1000


data = concatDF

conds =(data[x]<2)&(data.NcontrastsDisplayed>1)&(data.meanPerformance>0.0)& (data.trial_num>minTrial)& (data.trial_num<maxTrial) & ~(data.mouse.isin(['MBOT22_143','MBOT24_478']))
data = data[conds]
palette = sns.color_palette('viridis',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (3,5))
order = [1,0.5,0.25,0,-0.25,-0.5,-1]
#ax = sns.pointplot(data = data,x=x,y=y,hue=hue,palette= palette,alpha = 0.5,aspect = 1.5,dodge=0.2)
ax = sns.pointplot(data = data,x=x,y=y,color='black',aspect = 1.5,dodge=0.2)
#ax = sns.pointplot(data = data,x=x,y=y)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Psychometric function',fontsize = 15)
ax.set_xlabel('Contrast',fontsize = 15)
ax.set_ylabel('% correct',fontsize = 15)
ax.set_ylim(0,1)

#ax.set_xticklabels([-45,45],fontsize = 15)
#ax.set_yticklabels(labels =np.round(np.arange(0,1,0.2),2),fontsize = 15)
#ax.axes.hlines(0.5,0,6,linestyles='dashed',color='black')
# ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
plt.show()


# # Save data frame! 

# In[ ]:


outname = analysisDir + 'DF_2022-02-28to2022-03-07.csv'

concatDF.to_csv(outname)


# In[ ]:


concatDF.date.unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


basePath


# In[ ]:


x ='signedContrast'
y = 'LeftTurn'
hue= 'mouse'
minTrial = 30
maxTrial = 300


data = concatDF

conds = (data.mouse.isin(['MBOT24_481']))&(data.date.isin(['2021-10-29']))
data = data[conds]
data['stim_contrast']


# In[ ]:


x ='stim_pos_init'
y = 'meanPerformance'
hue= 'mouse'
minTrial = 30
maxTrial = 300


data = concatDF

conds = (data.meanPerformance>0.65)& (data.trial_num>minTrial)& (data.trial_num<maxTrial) #& ~(data.mouse =='MBOT33_495')
data = data[conds].groupby('date').agg({'meanPerformance' : 'mean', 'mouse' : 'first'}).reset_index()
palette = sns.color_palette('viridis',len(data[hue].unique()))
fig,ax = plt.subplots(1,1,figsize = (3,5))

ax = sns.pointplot(data = data,x=hue,y=y,hue=hue,palette=palette)

#ax = sns.pointplot(data = data,x=x,y=y)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Performance in experts',fontsize = 15)
ax.set_xlabel('Expert mice',fontsize = 15)
ax.set_ylabel('Mean Performance',fontsize = 15)
ax.set_ylim(0,1.2)

ax.set_yticklabels(labels =np.round(np.arange(0,1.2,0.2),2),fontsize = 15)
ax.set_xticklabels(labels =[1,2,3,4],fontsize = 15)
# ax.axes.hlines(0.5,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
# ax.axes.hlines(0.70,0,data[x].max()-data[x].min(),linestyles='dashed',color='black')
plt.show()


# In[ ]:




