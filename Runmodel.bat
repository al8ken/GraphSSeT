:: The below is the reference model with default parameters. Do not run this line but copy and run below if desired
::python SHMIP_mw_example_SedModel_ABseries.py A5 default 0.25 1.0 2.2 1.5 2650 0.001 2.7e-7 2.02 1000 3
:: This is a standard model with dtritus as init, bedrock, basal. Change parameters as desired
python SHMIP_mw_example_SedModel_ABseries.py A5 default 0.25 1.0 2.2 1.5 2650 0.001 2.7e-7 2.02 1000 3
:: The addition of 'D' will run a model with detritus tracking from nodal properties. Change parameters as desired
python SHMIP_mw_example_SedModel_ABseries.py A5D default 0.25 1.0 2.2 1.5 2650 0.001 2.7e-7 2.02 1000 3
:: 'C' series models use a different script. Otherwise all is the same
python SHMIP_mw_example_SedModel_Cseries.py C1 default 0.25 1.0 2.2 1.5 2650 0.001 2.7e-7 2.02 1000 3
:: 'C' series models use a different script. Otherwise all is the same
python SHMIP_mw_example_SedModel_Cseries.py C1D default 0.25 1.0 2.2 1.5 2650 0.001 2.7e-7 2.02 1000 3