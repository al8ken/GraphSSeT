# GraphSSeT
Code and model inputs to run graph-based subglacial sediment transport from SHMIP hydrology models

Input Data:
Input data are the matlab output files (.mat) from running SHMIP models in GlaDS
Input data for series A and B models are in the folder InputData
Input data for series C models are in the associated zenodo repository (doi: //////)

Model Scripts: 

GraphSSeT_NetworkX.py contains the GraphSSet model implemented in the NetworkX python module.
NetworkX_funcs.py contains functions for manipulationg and displaying graphs.
ReadGladsMat.py reads a GlaDS output matlab file to a NetworkX graph.

SHMIP_mw_example_graph_ABseries.py generates a main graph and set of subgraphs for GraphSSet for a steady state model. This file is suited to interactive use to generate the best realistions of your hydrology network and nice pictures etc
SHMIP_mw_example_graph_Cseries.py generates a main graph and (first) set of subgraphs for GraphSSet for a dynamic model. This file is suited to interactive use to generate the best realistions of your hydrology network and nice pictures etc

SHMIP-mw_example_SedModel_ABseries.py will run the GraphSSeT model for a set of pickle files output from SHMIP_mw_example_graph_ABseries.py. This script is suited to programmatic use to generate multiple scenarios from one or several hydrology models. 
SHMIP-mw_example_SedModel_Cseries.py will run the GraphSSeT model for a set of pickle files output from SHMIP_mw_example_graph_Cseries.py. This script is suited to programmatic use to generate multiple scenarios from one or several hydrology models.

Runmodel.bat is a windows batch script to run the above scripts

----------------------------------------------------------------------------------------------------------------------------------

Getting started:

First ensure your python distribution is reasonably up to date with the following modules installed:

numpy
networkx
scipy and/or mat73 - required for loading the matlab files.
matplotlib - required for imaging
pickle - required for storing model outputs and inputs

To run a model from the included set:

To generate a network representation of a hydrology model use the script SHMIP_mw_example_graph_ABseries.py (or SHMIP_mw_example_graph_Cseries.py). The script should run interactively in your favourite IDE (it was made and tested in Spyder).
    >>> The output will be a set of pickle files each containing a graph (or subgraph) plus any figures that you choose to save

To run the GraphSSet model, configure your batch file or shell script, plus in the script SHMIP-mw_example_SedModel_ABseries.py (or SHMIP-mw_example_SedModel_Cseries.py) anything like paths or variables that are not standard. If you want to run interactively you'll just need to replace the sys.argv[n] inputs with the desired values. The output will be in a (new) directory ./Output_ModelName/ and will contain

    >> weekly images showing some model variables
    >> overall modell evolution plots for volume, concentration, grainsize and detritus
    >> pickles containing the numerical data for the above
    >> a pickle file with the final main graph (this contains all data and new subgraphs can be made from this)

Custom images of the output data can be generated using the functions included in NetworkX_funcs.py

Running your own model:

If you want to change the duration, subsampling rate, or timestep length or any other variables/parameters for models - go for it. It should work if you don't do something silly (like negative dt)

If you have a matlab GlaDS model output (as.mat) you should be able to run this the above way, with changes to whatever parameters you want. See comments in ReadGladsMat.py.
If you have another hydrology model output GraphSSet can probably handle it, as long as you can a) derive hydraulic potential (or its gradient), b) identify channel beginnings and end points, c) calculate channelised flux (+/- channel area) d) calculate (or assume) ice sheet basal velocity(+/-shear stress). You'll need to get it into NetworkX graph form yourself. We'd appreciate a pull request if you get it to work for other models!
If you have only an ice-sheet model - see the above requirements to define - also, watch this space!

---------------------------------------------------------------------------------------------------------------------------------

Please don't forget to cite/acknowledge this GitHub repository, the associated zenodo repository() and the paper().
Any questions or problems can be directed to Alan Aitken (alan.aitken@uwa.edu.au)

Happy modelling!