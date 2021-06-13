# Information Theoretic Based Model Benchmarking

Hydrological model performance is commonly evaluated based on different statistical metrics e.g., 
the Nash Sutcliffe coefficient ([NSE](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient)). 
However, these metrics do not reveal model functional performances, such as how different flux and store variables interact 
within the model. As such, they are poor in model diagnostics and fail to indicate whether the model is right for the right 
reason. In contrast, information theoretic metrics are capable of revealing model internal functions and their tradeoffs with
predictive performance. In this, notebook we demonstrate the use of interactive and reproducible computation of information 
flow metrics, particularly [Transfer Entropy (TE)](https://en.wikipedia.org/wiki/Transfer_entropy) and 
[Mutual Information(MI)](https://en.wikipedia.org/wiki/Mutual_information), in diagnosing model performance.

The model in focus is the the National Hydrologic Model using the PRMS model ([NHM-PRMs](https://pubs.er.usgs.gov/publication/tm6B9)). 
NHM-PRMS has two model products covering the CONUS - the calibrated and uncalibrated model products. 
Out of the CONUS wide NHM-PRMS products, this notebook focused on the NHM-PRMS product at the 
[HJ Andrews watershed, OR](https://andrewsforest.oregonstate.edu/). 

Please click the binder link below to launch the notebook on cloud.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMscience/NHM_PRMS_Bechmarking/HEAD)

# Citation

Edom Moges, Laurel Larsen, Ben Ruddell, Liang Zhang, Jessica M. Driscoll and Parker Norton, 2021. 
EM_v01_Information theoretic based model benchmarking. Accessed 06/11/2021 at 
https://github.com/EMscience/NHM_PRMS_Bechmarking 

# Acknowledgements


This work is supported by the NSF Earth Cube Program under awards 
[1928406](https://nsf.gov/awardsearch/showAward?AWD_ID=1928406) and 
[1928374](https://nsf.gov/awardsearch/showAward?AWD_ID=1928374).


>>
                               =============[********]============== 
\
*Edom Moges* \
*edom.moges@berkeley.edu* \
*[Environmental Systems Dynamics Laboratory (ESDL)](https://www.esdlberkeley.com/)*\
*University of California, Berkeley* 
