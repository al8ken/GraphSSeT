# GraphSSeT
Code and model inputs to run graph-based subglacial sediment transport from SHMIP hydrology models

Input Data (for examples):
Input data are the matlab output files (.mat) from running SHMIP models in GlaDS
Input data for series A and B models are in the folder InputData
<<<<<<< Updated upstream
Input data for series C models are in the associated zenodo repository (https://zenodo.org/doi/10.5281/zenodo.12570096)
=======
Input data for series C models are in the associated zenodo repository (https://doi.org/10.5281/zenodo.12570096)
>>>>>>> Stashed changes

Model Scripts: 

GraphSSeT_NetworkX.py contains the original GraphSSeT model implemented in serial using the NetworkX python module. 
    >>  This version is deprecated but is kept for legacy use. 

GraphSSeT_NetworkX_parallel.py contains the updated GraphSSeT model implemented in parallel using ray and the NetworkX python module.
    >>  This version is the current working model

NetworkX_funcs.py contains functions for manipulating and displaying GraphSSeT graphs.
ReadGladsMat.py reads a GlaDS output matlab file to a NetworkX graph for either a steady state or non-steady state forcing.

SHMIP_graph_ABseries.py generates a main graph and set of subgraphs for GraphSSet for a steady state model forcing. 
    >>  This script is suited to interactive use to generate the best realistions of your hydrology network and nice pictures etc

SHMIP_SedModel_ABseries.py will run the GraphSSeT model for a set of pickle files output from SHMIP_graph_ABseries.py. 
    >>  This script is suited to interactive or programmatic use to generate one or multiple scenarios from one or several hydrology models. 

SHMIP_graph_CDseries.py generates a main graph and (first) set of subgraphs for GraphSSet for a non-steady state model forcing. 
    >>  This script is suited to interactive use to generate the best realistions of your hydrology network and nice pictures etc. Data for these models are toolarge for github (get from Zenodo)

SHMIP-mw_example_SedModel_Cseries.py will run the GraphSSeT model for a set of pickle files output from SHMIP_graph_CDseries.py. 
    >>  This script is suited to inteactive or programmatic use to generate one or multiple scenarios from one or several hydrology models.

----------------------------------------------------------------------------------------------------------------------------------
Getting Started:

First ensure your python distribution is reasonably up to date with the following modules installed:

numpy and scipy - required for doing the calculations
networkx - required for graph management
ray - required to manage parallelisation
mat73 - required for loading the matlab files.
matplotlib - required for imaging 
pickle - required for storing model outputs and reading inputs

To run a model from the included hydrology set, there are two steps:

STEP 1:

To generate a NetworkX representation of an input hydrology model use the script SHMIP_graph_ABseries.py (or SHMIP_graph_CDseries.py). The script should run interactively in your favourite IDE (it was made and tested in Spyder).
    >>  The output will be a set of pickle files each containing a graph (or subgraph) plus any plots that you choose to save

STEP 2: 

To run the GraphSSeT model, run the script SHMIP_SedModel_ABseries.py (or SHMIP_SedModel_CDseries.py) either interactively or from a shell script.

The output will be in a (new) directory ./Output_ModelName/ and will contain

    >> images showing the evolution of selected model variables for selected timesteps
    >> overall model evolution plots for volume, concentration, grainsize and detritus through time
    >> pickles containing the numerical data for the above at the selected timesteps
    >> pickles containing the graph output at slected timesteps and the final result (this contains all data and new subgraphs can be made from this)

More customised images of the output data can be generated using the functions included in NetworkX_funcs.py

If you want to change the duration, subsampling rate, or any other variables/parameters for models - go for it. Most things should work if you don't do something silly.

Running a model with your own hydrology input:

If you have an 'original' GlaDS matlab output (as.mat) you should be able to run this as above, with changes to whatever parameters you want. See comments in ReadGladsMat.py

If you have an ISSM GlaDS model output (as.nc and/or.mat) you can use the Read_ISSM_nc.py and/or read ISSM_mat.py.
    >> Multi level structs in the .mat format will not be read into python with the .mat reader(s) so it is recommended to export to NetCDF using the ISSM NetCDF export functionality
    >> These scripts are not tested across all possible ISSM outputs, so you may need to improvise, please contact Alan Aitken if you need help.

If you have a VTU format (e.g. from Elmer-Ice) you can use the ReadVTU.py script. 
    >> This script is not tested across all possible Elmer-Ice outputs, so you may need to improvise, please contact Alan Aitken if you need help.

If you have another hydrology model output GraphSSeT can probably handle it, as long as you can 

    >>  derive hydraulic potential (or its gradient), 
    >>  identify channel beginning and end points, 
    >>  calculate channelised flux and channel area, 
    >>  calculate (or assume) ice sheet basal velocity(+/- basal shear stress).

You'll need to get it into NetworkX graph form yourself. We'd appreciate a pull request if you get it to work for other hydrology model output formats!

If you have only an ice-sheet model - see the above requirements to define - also, watch this space for integrated hydrology!

---------------------------------------------------------------------------------------------------------------------------------

<<<<<<< Updated upstream
Please don't forget to cite/acknowledge this GitHub repository, the associated zenodo repository(https://zenodo.org/doi/10.5281/zenodo.12570096) and the paper(Aitken, A. R. A., Delaney, I. A., Pirot, G., and Werder, M.: Modelling subglacial fluvial sediment transport with a graph-based model, GraphSSeT, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-274, 2024.).
=======
Please don't forget to cite/acknowledge this GitHub repository, the associated zenodo repository(https://doi.org/10.5281/zenodo.12570096) and the paper(https://doi.org/10.5194/tc-18-4111-2024).
>>>>>>> Stashed changes
Any questions or problems can be directed to Alan Aitken (alan.aitken@uwa.edu.au)

Happy modelling!
