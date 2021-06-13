#!/usr/bin/env python
# coding: utf-8

# # Workflow: Information theoretic based model benchmarking
# A product of : 
# 
# >   [Environmental Sytems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)
#                University of California, Berkeley 
# 
# Authors: 
# 
# > >   **Edom Moges<sup>1</sup>, Laurel Larsen<sup>1</sup>, Ben Ruddell<sup>2</sup>,  Liang Zhang<sup>1</sup>, Jessica Driscoll<sup>3</sup> and Parker Norton<sup>3</sup>**
# 
# <sup>1</sup> University of California, Berkeley
# 
# <sup>2</sup> Northern Arizona University
# 
# <sup>3</sup> United States Geological Survey (USGS)

# ## Notebook description

# This notebook has three steps:
# 
# 1. Loading the calibrated and uncalibrated HJ Andrews NHM-PRMS model product (Section 3)
# 2. Interactively evaluating model performances using the Nash-Sutcliffe coefficient (Section 4)
# 3. Executing information theoretic based model performance evaluation to understand (Section 5): 
# 
# 
#     i. tradeoffs between predictive and function model performance (Section 5.2)
#     ii. model internal function using process network plots of Tranfer Entropy (Section 5.3)
# 

# # Load Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import glob
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.lines import Line2D
import ipywidgets
from termcolor import colored
plt.ion()


import sys
import time
import random
import os
import math
import pickle
from matplotlib import cm

import xarray as xr
import numcodecs
import zarr

from joblib import Parallel, delayed

from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
rcParams["font.size"]=14
plt.rcParams.update({'figure.max_open_warning': 0})
register_matplotlib_converters()


# In[2]:


import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, output_file
hv.extension("bokeh", "matplotlib")


# In[3]:


## Local function
sys.path.insert(1, './Functions')

from PN1_5_RoutineSource import *
from ProcessNetwork1_5MainRoutine_sourceCode import *
from plottingUtilities_Widget import *


# In[4]:


# local file paths
np.random.seed(50)
pathData = (r"./Data/")
pathResult = (r"./Result/")


# # Load the NHM-PRMS data at the HJ Andrews watershed 
# 
# This notebook uses a specific standard data input format. The standard data input format is a tab delimited .txt file. The column names are specified in the following cell.
# 
# In order to apply to a different dataset, please adopt and follow the standard data format. 
# 
# Two products of the NHM-PRMS are loaded:
# 1. the calibrated NHM-PRMS model product - in this model version, the model parameters are mathematically optimized.
# 2. the uncalibrated NHM-PRMS model product - in this model version, the model parameters are estimated based on their physical interpretation.

# In[5]:


# NHM PRMS simulation results names - TableHeader
TableHeader = ['observed_Q','basin_ppt','basin_tmin','basin_tmax','model_Q','basin_soil_moist','basin_snowmelt','basin_actet','basin_potet']

# Calibrated model
CalibMat = np.loadtxt(pathData+'CalibratedHJAndrew.txt',delimiter='\t') # cross validate with matlab
#CalibMat[0:5,:]

# Uncalibrated model
UnCalibMat = np.loadtxt(pathData+ 'UnCalibratedHJAndrews.txt',delimiter='\t') # cross validate with matlab
#UnCalibMat[0:5,:]


# # Traditional model performance metrics
# 
# This section demonstrates model evaluation based on the most common hydrological  model predictive performance measure- the Nash Sutcliffe coefficient (NSE). Two versions of the NSE are adopted - the untransformed NSE and the log trasformed NSE. These two metrics are suited to evaluate model predictive perormance in capturing the different segments of a hydrograph. 
# 
# 1. untransformed Nash-Sutcliffe coefficeint (NSE) - as an L2-norm, NSE is biased towards high flow predictive performance. This makes NSE suited for evaluating model's performance in capturing high flows.
# 2. Log transformed Nash-Sutcliffe coefficeint (logNSE) - biased towards predictive performance of low flows. This makes logNSE suited for evaluating models's performance in capturing low flows.
# 
# Interactive widgets are used to reveal the comparative performance of the calibrated and uncalibrated model performance using the above two common predictive performance measures. 

# **Please click over the interactive widget buttons to compare the performances of the different versions of the model and the performance measures**

# In[6]:


Traditional_widget = ipywidgets.interactive(
    NSE,
    
    ModelVersion = ipywidgets.ToggleButtons(
        options=['Calibrated', 'Uncalibrated'],
        description='Model Version:',
        disabled=False,
        button_style='',
        tooltips=['Model paramters are mathematically optimized.', 'Model parameters are estimated based on their physical interpretation.']
        ),
   
    PerformanceMetrics = ipywidgets.ToggleButtons(
        options=['Untransformed', 'Logarithmic'],
        description='Performance Metric:',
        disabled=False,
        button_style='',
        tooltips=['Biased towards high flows.', 'Biased towards low flows.']
        )
)
display(Traditional_widget)


# # Information theoretic based model benchmarking 
# 
# Beyond the traditional model performance measures, two model diagnostics are undertaken using information theoretic concepts. The diagnostics focus are:
# 
# 1. understanding the tradeoffs between predictive and functional model performance.
# 2. understanding model internal functions that lead to the predictive performance.
# 
# Achieving the above two undertakings, require computations of Mutual Information (MI) and Transfer Entropy (TE). While MI is used as a predictive performance metrics, TE is used as an indicator of model functional performance. 
# 
# The computation of MI and TE requires different [joint](https://en.wikipedia.org/wiki/Joint_probability_distribution) and [marginal](https://en.wikipedia.org/wiki/Marginal_distribution) probabilities of the different flux and store hydrological variables referred in the input table header. In order to compute these probabilities, we used histogram based probability estimation. 
# 
# Histogram based probabilities are sensitive to:
# 
# 1. the number of bins (*nBins*) used to develop the histogram,
# 2. how to handle higher and lower values in the first and last bins of the histogram (*low and high binPctlRange*). 
# 
# In order to understand the sensitivity of TE and MI to *nBins* and *low and high binPctlRange*, we rely on the use of interactive widgets. 
# 
# Hydrological time series are seasonal, e.g., seasonality driving both precipitation and streamflow. This seasonal signal may interfere with the actual information flow signal between two variables (e.g., information flow from precipitation to streamflow).  As such, MI and TE values are sensitive to the seasonality of the hydrological time series (flux and store variables). Therefore, in this notebook we compute MI and TE values based on the anomaly time series. The anomaly time series removes the seasonal signal of any store and flux variable by subtracting the long term mean of the day of the year average (DOY) from each time series measurement. In computing the long term DOY average, 'long term' is a relative quantity - particularly in the face of non-stationarity. Therefore, we allow choosing different 'long term' time lengths in computing DOY. 
# This choice can be set using the interactive widget (*AnomalyLength*). 
# 
# In order to evaluate the statistical significance of MI and TE, we shuffle the flux and store variables repeatedly. For each repeated shuffle sample, we compute both TE and MI. Out of these repeated shuffle based MI and TE, we compute their 95 percentile as a statistical threshold to be surpassed by the actual MI and TE to be statistically significant. Here, as the 95% is sensitive to the number of repeated shuffles (*nTests*), we implemented interactive widget to understand this sensitivity.  
# 
# We used interactive widgets to understand the sensitivity of TE/MI to these factors. These factors are abbreviated as follows.
# >
# 1. ***nBins*** - the number of bins to be used in computing MI and TE
# 2. ***UpperPerct*** - upper percentile of the data to be binned
# 3. ***LowerPerct*** - lower percentile of the data to be binned
# 4. ***TransType*** - defines the options whether to perform MI and TE computation based on the anomaly time series or the raw data time series. Two options are implemented. (option 0 - raw data) and (option 1 - anomaly tranform)
# 5. ***AnomalyLength*** - length of 'long term' in computing DOY average for annomal time series generation.
# 6. ***nTests*** - the number of shuffles to be used in computing the 95% statistical threshold.
# 
#  

# ## Executing the info-flow code for the calibrated NHM-PRMS model at the HJ Andrews
# 
# In setting up the info-flow computation, the following basic info-theoretic options can be left to their default values. For understanding TE/MI sensitivity, please refer to the interactive widgets below. 

# In[7]:


optsHJ= {'SurrogateMethod': 2, # 0 no statistical testing, 1 use surrogates from file, 2 surrogates using shuffling, 3 IAAF method
        'NoDataCode': -9999,
        'anomalyPeriodInData' : 365 , # set it to 365 days in a year or 24 hours in a day #### 365days/year
        'anomalyMovingAveragePeriodNumber': 5, # how many years for computing mean for the anomaly ***
        'trimTheData' : 1,  #% Remove entire rows of data with at least 1 missing value? 0 = no, 1 = yes (default)
        'transformation' : 1 } # % 0 = apply no transformation (default), 1 = apply anomaly filter ***


# In[8]:


# TE inputs 
optsHJ['doEntropy'] = 1
optsHJ['nTests'] = 10 # Number of surrogates to create and/or test (default = 100) ***
optsHJ['oneTailZ'] = 1.66
optsHJ['nBins'] = np.array([11]).astype(int) # ***
optsHJ['lagVect'] = np.arange(0,20) # lag days
optsHJ['SurrogateTestEachLag'] = 0
optsHJ['binType'] = 1 # 0 = don't do classification (or data are already classified), 1 = classify using locally bounded bins (default), 2 = classify using globally bounded bins
optsHJ['binPctlRange'] = [0, 99] # ***

# Input files and variable names
optsHJ['files'] =  ['./Data/CalibratedHJAndrew.txt']  
optsHJ['varSymbols'] = TableHeader
optsHJ['varUnits'] = TableHeader
optsHJ['varNames'] = TableHeader

# parallelization
optsHJ['parallelWorkers'] = 16 # parallelization on lags H and TE for each lag on each core.

# Saving results and preprocessed outputs
optsHJ['savePreProcessed'] = 0
optsHJ['preProcessedSuffix'] = '_preprocessed'
optsHJ['outDirectory'] = './Result/'
optsHJ['saveProcessNetwork'] = 1 
optsHJ['outFileProcessNetwork'] = 'Result'

# optsHJ['varNames']


# In[9]:


# Define Plotting parameters
popts = {}
popts['testStatistic'] = 'TR' # Relative transfer intropy T/Hy
popts['vars'] = ['basin_ppt','model_Q'] # source followed by sink
popts['SigThresh'] = 'SigThreshTR' # significance test critical value
popts['fi'] = [0]
popts['ToVar'] = ['model_Q']
#popts


# In[10]:


def WidgetInfoFlowMetricsCalculator(TransType, AnomalyLength, nTests, nBins, UpperPct, LowerPct):
    optsHJ['transformation'] = TransType
    optsHJ['anomalyMovingAveragePeriodNumber'] = AnomalyLength
    optsHJ['nTests'] = nTests
    optsHJ['nBins'] = np.array([nBins]).astype(int)
    optsHJ['binPctlRange'] = [LowerPct, UpperPct]
    
    
    CouplingAndInfoFlowPlot(optsHJ,popts) 


# **Below are the sensitivity analysis widgets. Please use the sliding widgets to interact with the different options and values**

# In[11]:


get_ipython().run_cell_magic('time', '', 'InfoFlowWidgetPlotter = ipywidgets.interactive(\n    WidgetInfoFlowMetricsCalculator,\n    \n    TransType = ipywidgets.IntSlider(min=0, max=1, value=1),\n    AnomalyLength = ipywidgets.IntSlider(min=0, max=10, value=5),\n    nTests = ipywidgets.IntSlider(min=10, max=1000, value=10),\n    nBins = ipywidgets.IntSlider(min=5, max=15, value=11),\n    UpperPct = ipywidgets.IntSlider(min=90, max=100, value=99),\n    LowerPct = ipywidgets.IntSlider(min=0, max=10, value=0) \n    \n\n)\nInfoFlowWidgetPlotter')


# ## Interpreting the MI and TE results for model diagnostics
# 
# This section demonstrates the interpretation and use of the info-flow results to diagnose model performances in understanding the tradeoff between predictive and functional performance and revealing model internal functions in generating the predictive performance.  

# In[12]:


# Loading Info-flow Results for further Interpretation

RCalib = pd.read_pickle(r'./Result/Result_R.pkl')
optsHJCal = pd.read_pickle(r'./Result/Result_opts.pkl')


# ### Tradeoff between functional and predictive model performances
# 
# Model development and parameter optimization can lead to tradeoffs between functional and predictive performances. In the figure below x-axis refers to model functional performance (the difference between observed and model information flow from input precipitation to output streamflow) while predictive model performance refers to the mutual information between observed and modeled streamflow. 
# 
# The ideal model should have MI=1. (or  1-MI = 0) and TE<sub>model</sub> - TE<sub>observed</sub> = 0. As such, the (0,0) coordinate is the ideal model location.
# 
# In contrast, 
# 1. a model in the right panel (TE<sub>model</sub> - TE<sub>observed</sub> > 0) is an overly deterministic model - i.e., a model that abstracts more information from input precipitation than the observed.
# 2. a model in the left panel (TE<sub>model</sub> - TE<sub>observed</sub> < 0) is an random model - i.e., a model that extracts very poor information from input precipitation compared to the observation.
# 
# 
# As it is shown in the above TE/MI results, TE and MI are computed at different lags. The lags refer to the timestep by which the source variable is lagged in relation to the sink variable. The sink variable refers to model streamflow estimate (model_Q) while the remaining store/flux variable are source variables.
# 
# The interactive widgets below offer an opportunity to understand the performance tradeoff of the model at the different lag timesteps. 

# In[13]:


def plot_widgetPerformanceTradeoff(Lag, xmin=None, xmax=None, ymin=None, ymax=None):
    
    plotPerformanceTradeoff(Lag, RCalib, 'Calibrated')   


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
PerformanceTradeoff_widget = ipywidgets.interactive(
    plot_widgetPerformanceTradeoff,
   
    Lag =ipywidgets.IntSlider(min=0, max=15, value=1),
    xmin=ipywidgets.FloatText(value=-0.1),
    xmax=ipywidgets.FloatText(value=0.2),
    ymin=ipywidgets.FloatText(value=0),
    ymax=ipywidgets.FloatText(value=1)
)
PerformanceTradeoff_widget


# ### Process Network plots
# 
# Process Networks(PN) offer platform of presenting model internal functions based on transfer entropy.
# Similar to the above widgets, the process network plot below is generated at different lag timesteps. Please use the interactive widgets to reveal model internal working at the different lags considered. 

# In[15]:


def plot_widgetProcessNetworkChord(Lag):
    
    generateChordPlots2(RCalib,Lag,optsHJ,'Calibrated') # lag=0 
    


# PN_Chord_widget = ipywidgets.interactive(
#     plot_widgetProcessNetworkChord,
#    
#     Lag = ipywidgets.IntSlider(min=0, max=15, value=0)
# )
# PN_Chord_widget

# ## Exercise - Execute PN for the Uncalibrated model

# In[16]:


get_ipython().system(" echo 'This can be accomplished by changing the input data table.'")


# In[ ]:




