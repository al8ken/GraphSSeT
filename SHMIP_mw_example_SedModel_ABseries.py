import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from NetworkX_funcs import *
from ReadGladsMat import *
import pickle
from GraphSSet_NetworkX import SubglacialErosionandSedimentFlux as SGST
import sys
import os

InputModel = sys.argv[1]
ModelInstance = sys.argv[2]
ModelName = InputModel + "_" + ModelInstance

OutputDir = os.path.join("./Output/", ModelName)
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)
    
#run control parameters
InitTillH = float(sys.argv[3]) #m
MaxTillH = float(sys.argv[4]) #m
phi = float(sys.argv[5])
meanD = (2**-phi)/1000 # input is phi so convert to m
stdD = float(sys.argv[6]) # standard deviation of ln(grainsize)
rhog = float(sys.argv[7]) #kg m-3
Dsig = float(sys.argv[8]) #m^-1
K = float(sys.argv[9])
L = float(sys.argv[10])
samp_n = int(sys.argv[11]) #number of samples to define grain size populations...less is quicker but more variable
dt = float(sys.argv[12])*3600 # input in hours

if InputModel[-1] == 'D':
    DetMode = "NodeProp"
    InputModel = InputModel[:-1]
else:
    DetMode = "SedErod"

#define functions for later tracking and to make figures

#volumetric output at a node set    
def TrackVol(t,Graph,Output = [[0,0],[0.0],[0.0],[0.0],[0.0]], prop = 'VSo'):
    OutletVolumeFlux = np.array([Graph.nodes[key][prop] for key in Graph.nodes])
    TotalVolumeFlux = np.sum(OutletVolumeFlux)
    Output[0].append(t) # time of timestep (s)
    Output[1].append(OutletVolumeFlux) # volume flux at each node for that timestep (m^3)
    Output[2].append(Output[2][-1]+OutletVolumeFlux) # cumulative volume flux at each node by that timestep (m^3)
    Output[3].append(TotalVolumeFlux) # total volume across all nodes for that timestep (m^3)
    Output[4].append(Output[4][-1]+TotalVolumeFlux) # cumulative total volume flux across all nodes by that timestep (m^3)

#volumetric concentration at a node set
def TrackConc(t,Graph,Output = [[0.0],[0.0]], VWprop = 'VW', VSprop = 'VSo'):
    OutletVSFlux = np.array([Graph.nodes[key][VSprop] for key in Graph.nodes])
    TotalVSFlux = np.sum(OutletVSFlux)
    OutletVWFlux = np.array([Graph.nodes[key][VWprop] for key in Graph.nodes])
    TotalVWFlux = np.sum(OutletVWFlux)
    OutletSConc = OutletVSFlux/OutletVWFlux
    TotalSConc = TotalVSFlux/TotalVWFlux
    Output[0].append(t) # time of timestep (s)
    Output[1].append(OutletSConc) # volumetric concentration at each node for that timestep
    Output[2].append(TotalSConc) # mean volumetric concentration across all nodes for that timestep

def TrackGS(t,Graph,Output = [[0.0],[0.0],[0.0]], Vprop = 'VSo', GSprop = 'd_dist_node'):
    OutletVolumeFlux = np.array([Graph.nodes[key][Vprop] for key in Graph.nodes])
    MeanOutletGS = np.array([np.exp(Graph.nodes[key][GSprop][0]) for key in Graph.nodes])
    MeanTotalGS = np.sum(OutletVolumeFlux*MeanOutletGS)/np.sum(OutletVolumeFlux)
    Output[0].append(t) # time of timestep (s)
    Output[1].append(MeanOutletGS) # grain size at each node for that timestep (m)
    Output[2].append(MeanTotalGS) # volume-weighted mean grain size across all nodes for that timestep (m)

def TrackDet(t,Graph,Output = [[],[]], Vprop = 'VSo', Dprop = 'detritus_node'):
    OutletVolumeFlux = np.array([Graph.nodes[key][Vprop] for key in Graph.nodes])
    TotalVolumeFlux = np.nansum(OutletVolumeFlux)
    OutletDetritus = [Graph.nodes[key][Dprop] for key in Graph.nodes]
    DetritalProps = np.zeros(len(OutletDetritus[0].keys()))
    for m,key in enumerate(OutletDetritus[0].keys()):
        for i,j in enumerate(OutletDetritus):
           DetritalProps[m] += j[key]*OutletVolumeFlux[i]/TotalVolumeFlux
    Output[0].append(t) # time of timestep (s)
    Output[1].append(OutletDetritus) # detrital proportions at each node for that timestep
    Output[2].append(DetritalProps) # volume-weighted detrital proportions across all nodes for that timestep

# Function to make a Figure - here we plot flux density, jam status, till thickness and grain size but any edge property could be plotted
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(4,1, figsize = (8,8), dpi = 300)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'flux_density', lw = 0.75, cmap = 'inferno', minprop= 0, maxprop = 50, fig = fig, ax = axs[0])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'jammed', lw = 0.75, cmap = 'inferno', minprop= 0.0, maxprop = 1.0, fig = fig, ax = axs[1])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'till_thickness', lw = 0.75, cmap = 'inferno', minprop= 0.0, maxprop = 0.75, fig = fig, ax = axs[2])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'd_median', lw = 0.75, minprop = 0, maxprop = 5e-4, cmap = 'inferno', fig = fig, ax = axs[3])

#%% Read in pickle files and remake subgraphs. 
#overall graph (required)
with open('FinalNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    Network = pickle.load(file)
#level 0 sub-graph (required)    
with open('SubNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    SubNetCopy = pickle.load(file)
#L1 and L2 subgraphs (rquird her but not in general)
with open('BC0.0Network_SHMIP_'+InputModel+'.pickle','rb') as file:
    EBC0Copy = pickle.load(file)
with open('BC0.005Network_SHMIP_'+InputModel+'.pickle','rb') as file:
    EBCMinCentCopy = pickle.load(file)

#We need to remake subgraphs so they are views of main Network
SubNet = nx.subgraph(Network,SubNetCopy.nodes)
EBC0 = nx.subgraph(Network,EBC0Copy.nodes)
EBCMinCent = nx.subgraph(Network,EBCMinCentCopy.nodes)
#%% for the level 0 subgraph identify outlet nodes and their coords
Out_nodes = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])<=1]
Xmax = 50.0 #meters from margin
ToRemove = []
for i,j in enumerate(Out_nodes):
    if SubNet.nodes[j]['coords'][0] >= Xmax:
        ToRemove.append(j)
Out_nodes = [u for u in Out_nodes if u not in ToRemove]
Out_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Out_nodes)]

#Make a subgraph involving just the output nodes
OutNet = nx.subgraph(Network,Out_nodes)
#%% Initialise the model parameters
#some typical parameters for erosion definition, for reference
#K = 2.7e-7  # preexponent for m/a (Herman et al., 2015)
#L = 2.02  # dimensionless (Herman et al., 2015)
#K= 1e-4 # linear formulation (Herman et al., 2015)
#L = 1 # linear formulation (Herman et al., 2015)
#W = 2e-10 # work-rate scaling parameter - m/s velocity (Pollard and DeConto, P3, 2003)

#set start time
t = 0.0 #nominally this is one timestep before the 'start' of the model, could be a real datetime (in seconds) but we have zero

#Initialise the arrays to store model outputs
#here we will track at outlet nodes but this could be any node set 
#sediment volume discharged
OutVol = [[0.0],[0.0],[0.0],[0.0],[0.0]]
#sediment concentration
OutConc = [[0.0],[0.0],[0.0]]
#grain size
OutGS = [[0.0],[0.0],[0.0]]
#detrital properties
OutDet = [[0.0],[{'init':0.0}],[0.0]]

#%%instantiate the sugset model(s) for L0, L1 and L2. Excepting the subgraph these are the same but need not be.
#L0
sgst0 = SGST(SubNet,
            potgrad_method = "Direct",
            cflux_method = "FluxArea",
            erosion_method = "Vel",
            transport_method="EngelundHansen",
            detritus_method=DetMode,
            MaxTillH=MaxTillH,
            InitTillH = InitTillH,
            meanD = meanD, #m
            stdD = stdD,
            samp_n = samp_n,
            rhog = rhog, #kg m-3
            Dsig = Dsig, #m^-1
            K = K,
            L = L,
            #W = W,
            )

#L1
sgst1 = SGST(EBC0,
            potgrad_method = "Direct",
            cflux_method = "FluxArea",
            erosion_method = "Vel",
            transport_method="EngelundHansen",
            detritus_method=DetMode,
            MaxTillH=MaxTillH,
            InitTillH = InitTillH,
            meanD = meanD, #m
            stdD = stdD,
            samp_n = samp_n,
            rhog = rhog, #kg m-3
            Dsig = Dsig, #m^-1
            K = K,
            L = L,
            #W = W,
            )

#L2
sgst2 = SGST(EBCMinCent,
            potgrad_method = "Direct",
            cflux_method = "FluxArea",
            erosion_method = "Vel",
            transport_method="EngelundHansen",
            detritus_method="SedErod",
            MaxTillH=MaxTillH,
            InitTillH = InitTillH,
            meanD = meanD, #m
            stdD = stdD,
            samp_n = samp_n,
            rhog = rhog, #kg m-3
            Dsig = Dsig, #m^-1
            K = K,
            L = L,
            #W = W
            )

#%%initialise a stady state model for each
sgst0.initialise_steady()
sgst1.initialise_steady()
sgst2.initialise_steady()
#%% run once at L0 to make sure everything is the same time and has all variables - this saves some time
t += 1 #just one second
sgst0.run_one_step_steady(t)

#Make the figure for the initial network
MakeFig(Network,SubNet)
fn = os.path.join(OutputDir,'model_initial.png')
plt.savefig(fn)
plt.close()

#%% run several cycles of one week each
num_timesteps = 26 #weeks
times = []
for timestep in range(0, num_timesteps, 1):
    for i in range(0,7):
        #L0 to begin every day - nominally 12AM
        t += dt/2
        sgst0.run_one_step_steady(t)
        TrackVol(t,OutNet,OutVol, prop = 'VSo')
        TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
        TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
        TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
        times.append(t)
        #three cycles of L1 and L2
        for j in range (1,4):
            #once on L1
            t += dt/2
            sgst1.run_one_step_steady(t)
            TrackVol(t,OutNet,OutVol, prop = 'VSo')
            TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
            TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
            TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
            times.append(t)
            #once on L2
            t += dt/2
            sgst2.run_one_step_steady(t)
            TrackVol(t,OutNet,OutVol, prop = 'VSo')
            TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
            TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
            TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
            times.append(t)
        #L0 to end every day - nominally 9PM
        t += dt/2
        sgst0.run_one_step_steady(t)
        TrackVol(t,OutNet,OutVol, prop = 'VSo')
        TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
        TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
        TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
        times.append(t)
    #Make a figure every week    
    MakeFig(Network,SubNet)
    fn = os.path.join(OutputDir,'model_week{}.png'.format(timestep))
    plt.savefig(fn)
    plt.close()

#run L0 network one final time to ensure all network is at same time
t += dt
sgst0.run_one_step_steady(t)
#%% write out results to file

#report out the output arrays 
fn = os.path.join(OutputDir,'OutVol.pickle')
with open(fn,'wb') as file:
    pickle.dump(OutVol, file)
fn = os.path.join(OutputDir,'OutConc.pickle')
with open(fn,'wb') as file:
    pickle.dump(OutConc, file)
fn = os.path.join(OutputDir,'OutGS.pickle')
with open(fn,'wb') as file:
        pickle.dump(OutGS, file)
fn = os.path.join(OutputDir,'OutDet.pickle')
with open(fn,'wb') as file:
        pickle.dump(OutDet, file)

#output the final main graph
fn = os.path.join(OutputDir,'ModelResult.pickle')
with open(fn,'wb') as file:
        pickle.dump(Network, file)
#%% make output plots

##%% make line plot of volume flux total and specific high-volume nodes
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[2][-1][n]>=OutVol[4][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutVol[2][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.5, label = key)
plt.plot(times,OutVol[4][1:], label = 'total output', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutVol.png')
plt.savefig(fn, dpi = 300)
plt.close()

# make line plot of sediment concentration
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[2][-1][n]>=OutVol[4][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutConc[1][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.5, label = key)
plt.plot(times,OutConc[2][1:], label = 'total concentration', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutConc.png')
plt.savefig(fn, dpi = 300)
plt.close()

# and of grain size
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[2][-1][n]>=OutVol[4][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutGS[1][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.25, label = key)
plt.plot(times,OutGS[2][1:], label = 'total average grain size', linewidth = 1, color = 'k')    
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutGS.png')
plt.savefig(fn, dpi = 300)
plt.close()

# and of cumulative volume flux by source
PlotOrder = [0,1,2]
DetClasses = ['init','basal','basement']
col = ['gray','lightyellow','lightsalmon']
DetVolArray = np.zeros(shape = (len(PlotOrder),len(times)))
for i,j in enumerate(OutDet[2][1:]):
    VScale = OutVol[3][i]
    for m,n in enumerate(PlotOrder):
        DetVolArray[m][i] = DetVolArray[m][i-1]+j[n] * VScale
plt.stackplot(times,DetVolArray, labels = DetClasses, colors = col)
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutDetVol.png')
plt.savefig(fn, dpi = 300)
plt.close()