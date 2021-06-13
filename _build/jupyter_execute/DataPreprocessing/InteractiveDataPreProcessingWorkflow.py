#!/usr/bin/env python
# coding: utf-8

# #           Workflow: Jupyter Supported Interactive Data Preprocessing 
# 
# A product of : 
# 
# >   [Environmental Sytems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)
#                University of California, Berkeley 
# 
# Authors: 
# 
# > >  ***Edom Moges<sup>1</sup>, Liang Zhang <sup>1</sup>, Laurel Larsen<sup>1</sup>,  Fernando Perez<sup>1</sup>, and Lindsey Heagy<sup>1</sup>***
# 
# <sup>1</sup> University of California, Berkeley

# ## Purpose
# 
# Raw hydrometeorological datasets contain errors, gaps and unrealistic values that needs preprocessing. 
# The objective of this work is developing an interactive data preprocessing platform that enables acquiring and transforming publicly available raw hydrometeorological data to a ready to use  dataset. This interactive platform is at the core of the Comprehensive Hydrologic Observatory SEnsor Network CHOSEN dataset (Zhang et al. 2021 submitted to HP). [CHOSEN](https://gitlab.com/esdl/chosen) provides a multitude of intensively measured hydrometeorological datasets (e.g., snow melt, soil moisture besides the common precipitation, air temperature and streamflow) across 30 watersheds covering the conterminous US. 

# ## Technical Contribution
# 
# * Bringing together common data quality control (QC) and missing value filling techniques such as interpolation and regression
# * Making data QC and filling missing values interactive to facilitate ease of computation and choosing parameters
# * Developing a new missing value filling technique named the Climate Catalog method that leverages similarity in annual hydrological cycles

# ## Methodology
# 
# This notebook starts with a cell that acquires a standard raw hydrometeorological data table and proceeds with cells that perform interactive computation to fill missing values. Three data filling methods are adopted:
# 1. Interpolation
# 2. Regression
# 3. Climate Caltalog
# 
# The details of the methods are described below in section 6 (Filling missing values).

# ## Results
# 
# This notebook presents data preprocessing performed on one of the [CHOSEN](https://gitlab.com/esdl/chosen) datasets, the Dry Creek watershed. Using this notebook the range of interpolation (interplimit), thresholds for potention regression data donors (RegThreshold) and climate catalog thresholds (corrThr and thrLen) are determined interactively. Through these three interactive data preprocessing methods missing values are filled. Having  a filled hydrometeorological dataset enables hydrological modeling and other water resources management analysis. 
# 
# In summary, this work:
# 
# * introduced a new missing value filling technique (Climate Catalog)
# * developed Jupyter based interactive open source data preprocessing tool
# * interactively transforms raw hydrometeorological data to a ready to use dataset
# * enables further collection and dissemination of datasets such as the CHOSEN data
# 

# ## Citation
# 
# Edom Moges, Liang Zhang, Laurel Larsen, Lindsey Heagy and Fernando Perez, 2021. EM_v01_Jupyter Supported Interactive Data Processing Workflow. Accessed 05/15/2021 at https://github.com/EMscience/CHOSENDryCreek

# # Setup

# ## Library import 

# In[1]:


# data manuplation
import numpy as np
import pandas as pd
# from scipy import signal

# visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ipywidgets

# basic date and file
import datetime as dt
import copy
import os


# math and statistics
from pandas.plotting import register_matplotlib_converters
from sklearn import linear_model
from sklearn.metrics import r2_score
from math import sqrt, pi
#import handcalcs.render

rcParams["font.size"]=14
plt.rcParams.update({'figure.max_open_warning': 0})
register_matplotlib_converters()
os.getcwd()


# ## Local library import

# In[2]:


import sys
sys.path.insert(1, './Functions')

from Source_QC_Widgets_functions_EM import regressorFunc, funcClimateCatalogWg,widgetInterpolation,widgetRegression,Date_to_float


# # Parameter definitions
# 
# 

# In[3]:


watershed = 'DryCreek' # name of the example watershed
main_str = 'LG' # name of the main watershed station in focus


# # Data import
# 
# The input data table needs to follow a standard naming. A column in the input table should have station name, variable name and measurement depths separated by underscore respectively.
# - e.g., LS_SoilTemperature_Pit2_2cm - represent a station LS, for soil temperature data, at pit 2 at a depth of 2cm.
# 

# In[4]:


# Read the raw original data table
table = pd.read_csv('1_'+watershed+'_Download_Aggregation.csv',header = 0,index_col = 'DateTime',
                    parse_dates = True, infer_datetime_format = True,low_memory=False)
display(table.head(2))
display(table.tail(2))


# In[5]:


# Double Check the station names
# breakdown discharge and hydrometeorological station names
all_stations = table.columns.str.extract(r'([^_]+)')[0]
print('All stations names: ', all_stations.unique())
print ('  ')
nameStrflwStation=[]
nameHydrMetStation=[]
for i in np.arange(len(table.columns)):
    if table.columns[i][-9:]=='Discharge':  ### 
        if not all_stations[i] in nameStrflwStation:
            nameStrflwStation.append(all_stations[i]) ### 
    else:
        if not all_stations[i] in nameHydrMetStation:
            nameHydrMetStation.append(all_stations[i])  ### 

print('Discharge stations :',nameStrflwStation)
print('  ')
print('Meteorology stations:',nameHydrMetStation)      


# # Data processing and analysis

# ##  Trim the original table
# This step adjusts the input table and removes lengthy missing values at the beigning and end of the input table.

# In[6]:


t = table.notna() 
t = ~np.isnan(table)
col = len(t.columns)
b = np.zeros([table.shape[1]])
c = np.array([table.shape[0]] * table.shape[1])

for i in range(col):
    if any(t.iloc[:,i]): # Since some are empty
        b[i] = list(np.where(t.iloc[:,i] == True))[0][0] # the first non nan value location
        c[i] = list(np.where(t.iloc[:,i] == True))[0][-1] # the last non nan value location
        
st_tab = b.min()
table1 = table.iloc[int(b.min()):int(c.max()) + 1,:] 

# Display the trimmed table
display(table1.head(2))
display(table1.tail(2))
print('trimmed row number is ', int(table.shape[0] -  table1.shape[0]))


# ## Define Quantity of interest (QOI)
# 
# QOI defines the variable name to be quality controlled and gap filled.
# 
# This variable can be assigned to any variable among the column names of the above table (raw data table).

# In[7]:


QOI = 'LS_SoilTemperature_Pit2_2cm'


# #  Filling missing values

# ## Interpolation
# 
# - Interpolation is the first gap filling technique adopted to fill short length missing values. It is not recommended for longer period missing values as it predicts unrealiable/unrealistc values. 
# 
# - This unrealiabilty can easily be detected by the interactive plots under the control parameter interpolation length limit (interplimit).
# 
# - As an example, compare interplimit = 7 and interplimit = 30 using the interactive widgets. This shows how the value 30 leads to unrealistic (e.g., horizontal straight lines) filling of missing values.

# In[8]:


def plot_widgetInterpolation(QOI=QOI, interplimit=7, Intpmethod='time', Intplimit_direction='both',
                        xmin=None, xmax=None, ymin=None, ymax=None):
    
    y, yIntp = widgetInterpolation(table1,QOI, interplimit, 'time', 'both')
    
    indx = Date_to_float(y.index)
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(indx, yIntp,'r',linewidth=2,label='Interpolated')
    ax.plot(indx, y, 'b',linewidth=2,label='Original')
    
    ax.set_xlabel("Year")
    ax.set_ylabel(QOI.split('_')[1])
    ax.set_title(QOI)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin, ymax])
    ax.grid(color='gray', linestyle='-.', linewidth=1)
    #ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')
    ax.legend(fontsize='small')
    
    # interaction responsiveness
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


# Beside the interplimit parameter, the interactive widget can also be used to choose different alternatives of the interpolation function (interpolation method and direction). Please take a look at the widget.

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

Interpolation_widget = ipywidgets.interactive(
    plot_widgetInterpolation,
    
    QOI=table1,
    Intpmethod = ['linear','time','space','index','pad','nearest', 'zero', 'slinear', 'quadratic', 'cubic'],
    Intplimit_direction=['forward', 'backward', 'both'],
    
    interplimit=ipywidgets.IntSlider(min=2, max=100, value=7),
    xmin=ipywidgets.FloatText(value=2008),
    xmax=ipywidgets.FloatText(value=2018),
    ymin=ipywidgets.FloatText(value=0),
    ymax=ipywidgets.FloatText(value=50)
)
Interpolation_widget


# ## Regression
# 
# 
# As an alternative to interpolation, longer period missing values are filled  by linear one at a time regression. 
# 
# - Regression requires measurements from more than one station. By comparing these stations using correlation coefficient, a station with the highest correlation coefficient is adopted as a donor regression station. 
# 
# - However, if the donor station's correlation coefficient is below a user defined regression thresold (Regthresh), regression will not be adopted. 
# 
# - This threshold (Regthresh) is identified interactively.
# 
# 

# In[10]:


def plot_widgetRegression(QOI=QOI, RegThreshold=0.7, xmin=None, xmax=None, ymin=None, ymax=None):
    
    y, yReg = widgetRegression(table1, QOI, RegThreshold=RegThreshold)
    
    indx = Date_to_float(y.index)
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(indx, yReg,'r',linewidth=2,label='After Regression')
    ax.plot(indx, y, 'b',linewidth=2,label='Raw Data')
    
    ax.set_xlabel("Year")
    ax.set_ylabel(QOI.split('_')[1])
    ax.set_title(QOI)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin, ymax])
    ax.grid(color='gray', linestyle='-.', linewidth=1)
    #ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')
    ax.legend(fontsize='small')
    
    # interaction responsiveness
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    # fig.gcf().canvas.set_window_title('Regression')


# Consider setting the RegThreshold widget to different values to understand its significance in performing regression for the purpose of filling missing values. For instance, RegThreshold = 1.0 will lead to no regression. 

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib qt, - a separate plot
# %matplotlib notebook - the plot reside within the notebook with save and zooming options
# %matplotlib widget - for jupyter lab
Regression_widget = ipywidgets.interactive(
    plot_widgetRegression,
    
    QOI=table1,
    
    RegThreshold=ipywidgets.FloatSlider(min=.5, max=1, value=.8),
    xmin=ipywidgets.FloatText(value=2008),
    xmax=ipywidgets.FloatText(value=2018),
    ymin=ipywidgets.FloatText(value=0),
    ymax=ipywidgets.FloatText(value=50)
)
Regression_widget


# ## Climate catalog
# 
# 
# Climate Catalog performs gap filling based on comparing the similarity of years. For instance, a missing value in a calendar day for a given year can be filled by a data from a year $(D_\text{missing year})$ that has the highest correlation with the missing year $(D_\text{year with the highest $r^2$})$ among the other years plus a sample from a normal distribution with a standard deviation of all the years for the missing calandar day $(N(0, \sigma ^2))$. Below is the mathematical formulation:
# 
# \begin{equation}
# D_\text{missing year} = D_\text{year with the highest $r^2$} + N(0, \sigma ^2)
# \end{equation}
# 
# 
# - In using Climate Catalog, we set two interactive parameters: 
#     1. corrThr - a minimum correaltion coefficient a  candidate year needs to satisfy to qualify as a donor year and 
#     2. thrLen - the minimum number of days the missing year has to have in order to perform Climate Catalog based gap filling. 

# In[12]:


def plot_widgetClimateCatalog(QOI=QOI, thrLen=200, corrThr=0.7, xmin=None, xmax=None, ymin=None, ymax=None):
    
    y_CC, yraw, indx_cc = funcClimateCatalogWg(table1, QOI, thrLen, corrThr)
    
    indx = Date_to_float(yraw.index)
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(indx, y_CC,'r',linewidth=2,label='After Climate Catalog')
    ax.plot(indx, yraw, 'b',linewidth=2,label='Raw Data')
    
    ax.set_xlabel("Year")
    ax.set_ylabel(QOI.split('_')[1])
    ax.set_title(QOI)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin, ymax])
    ax.grid(color='gray', linestyle='-.', linewidth=1)
    #ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')
    ax.legend(fontsize='small')
    
    # interaction responsiveness
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


# As a demonstration, consider setting thrLen and corrThr at different values to understand the tradeoff between the two parameters in filling missing values using the climate catalog method.

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

# %matplotlib qt, - a separate plot
# %matplotlib notebook - the plot reside within the notebook 
# %matplotlib widget - for jupyter lab
ClimateCatalog_widget = ipywidgets.interactive(
    plot_widgetClimateCatalog,
    
    QOI=table1,
    
    corrThr=ipywidgets.FloatSlider(min=.5, max=1., value=.7),
    thrLen=ipywidgets.IntSlider(min=1, max=365, value=7),
    xmin=ipywidgets.FloatText(value=2008),
    xmax=ipywidgets.FloatText(value=2018),
    ymin=ipywidgets.FloatText(value=0),
    ymax=ipywidgets.FloatText(value=50)
)
ClimateCatalog_widget


# In[ ]:





# In[ ]:




