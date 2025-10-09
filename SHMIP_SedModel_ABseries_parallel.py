#!/usr/env/python

"""
This is a sample script to run the GraphSSeT model for a member of the SHMIP 'A' or 'B' model ensembles. 

Before you can run this script you will need to have made the input graphs by running the script 

SHMIP_graph_ABseries.py for the desired model output 

For the GraphSSeT model description see the paper of Aitken et al. (2024)

https://doi.org/10.5194/tc-18-4111-2024

codeauthor:: Alan Aitken

This version October 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from NetworkX_funcs import *
from ReadGladsMat import *
from GraphSSeT_NetworkX_parallel import SubglacialErosionandSedimentFlux as SGST
from GraphSSeT_NetworkX_parallel import SGST_supervisor
import ray
import sys
import os
import pickle

#Model information
try:
    InputModel = sys.argv[1]
except IndexError:
    InputModel = 'A5'
try:    
    ModelInstance = sys.argv[2]
except IndexError:
    ModelInstance = 'test'

ModelName = InputModel + "_" + ModelInstance

OutputDir = os.path.join("./Output/", ModelName)
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

n_proc = 8 #number of ray processes

max_dt = 60.*60.*24. #max 1 day
min_dt = 60.* 60. #min 1 hour
one_week = 60.*60.*24.*7. # number of seconds per week
num_weeks = 2 # number of weeks for the model run

#run control parameters (input as a string from command line, or use defults if no (or incomplete) string provided
try:
    InitTillH = float(sys.argv[3]) #m
    MaxTillH = float(sys.argv[4]) #m
    meanD = float(sys.argv[5]) #mean Phi
    stdD = float(sys.argv[6]) # standard deviation of ln(grainsize)
    rhog = float(sys.argv[7]) #kg m-3
    sedl_factor = float(sys.argv[8]) #sediment uptake e-folding length
    Dsig = float(sys.argv[9]) #m^-1
    K = float(sys.argv[10]) #erosion law preexponent
    L = float(sys.argv[11]) #erosion law exponent
    samp_n = int(sys.argv[12]) #number of samples to define grain size populations...less is quicker but more variable
    dt = float(sys.argv[13])*3600 # input in hours
except IndexError:
    InitTillH = 0.25 #m
    MaxTillH = 1.0 #m
    meanD = 2.2 #mean Phi
    stdD = 1.5 # standard deviation of ln(grainsize)
    rhog = 2650 #kg m-3
    sedl_factor = 1.5 #sediment uptake e-folding length factor (* edge length)
    Dsig = 0.001 #m^-1
    K = 2.7e-7 #erosion law preexponent
    L = 2.02 #erosion law exponent
    samp_n = 1000 #number of samples to define grain size populations...less is quicker but more variable
    dt = 1.0*3600 # input in hours -- > seconds

#some typical parameters for erosion definition, for reference
#K = 2.7e-7  # preexponent for m/a (Herman et al., 2015)
#L = 2.02  # dimensionless (Herman et al., 2015)
#K= 1e-4 # linear formulation (Herman et al., 2015)
#L = 1 # linear formulation (Herman et al., 2015)
#W = 2e-10 # work-rate scaling parameter - m/s velocity (Pollard and DeConto, P3, 2003)

MinPhi = meanD-2*stdD
MaxPhi = meanD+2*stdD 

if InputModel[-1] == 'D':
    Dmode = "NodeProp"
    InputModel = InputModel[:-1]
else:
    Dmode = "SedErod"

#run modes 
HPGMode = 'Potential'
CFmode = 'FluxArea'
Emode = 'Vel'
Tmode = 'EngelundHansen'

# functions to convert grainsize: mm to Phi and vice versa
def phitomm(phi):
    mm = (2**-phi)/1000.
    return mm

def mmtophi(Graph):
    ds = nx.get_edge_attributes(Graph,'d_median')
    phis = {key: -np.log2(ds[key]*1000) for key in ds.keys()}
    nx.set_edge_attributes(Graph,phis,'phi_median')

#functions for tracking

#volumetric flux rates at output nodes
def TrackQ(t,Graph,Output = [[0,0],[0.0],[0.0]], prop = 'VSo'):
    node_dt = np.array([Graph.nodes[key]['VW'] for key in Graph.nodes])/np.array([Graph.nodes[key]['QW'] for key in Graph.nodes])
    OutletVolumeFlux = np.array([Graph.nodes[key][prop] for key in Graph.nodes])
    OutletQ = OutletVolumeFlux/node_dt
    TotalQ = np.nansum(OutletVolumeFlux)/np.nanmean(node_dt)
    Output[0].append(t)
    Output[1].append(OutletQ)
    Output[2].append(TotalQ)

#volumetric concentrations at output nodes
def TrackConc(t,Graph,Output = [[0.0],[0.0],[0.0]], VWprop = 'VW', VSprop = 'VSo'):
    OutletVSFlux = np.array([Graph.nodes[key][VSprop] for key in Graph.nodes])
    TotalVSFlux = np.nansum(OutletVSFlux)
    OutletVWFlux = np.array([Graph.nodes[key][VWprop] for key in Graph.nodes])
    TotalVWFlux = np.nansum(OutletVWFlux)
    OutletSConc = OutletVSFlux/OutletVWFlux
    TotalSConc = TotalVSFlux/TotalVWFlux
    Output[0].append(t)
    Output[1].append(OutletSConc)
    Output[2].append(TotalSConc)

#volume-weighted grain sizes at output nodes
def TrackGS(t,Graph,Output = [[0.0],[0.0],[0.0]], Vprop = 'VSo', GSprop = 'd_dist_node'):
    OutletVolumeFlux = np.array([Graph.nodes[key][Vprop] for key in Graph.nodes])
    MeanOutletGS = np.array([2**-Graph.nodes[key][GSprop][0] for key in Graph.nodes]) #output in mm
    MeanTotalGS = np.nansum(OutletVolumeFlux*MeanOutletGS)/np.nansum(OutletVolumeFlux)
    Output[0].append(t)
    Output[1].append(MeanOutletGS)
    Output[2].append(MeanTotalGS)

#volume-weighted detritus proportions
def TrackDet(t,Graph,Output = [[],[],[]], Vprop = 'VSo', Dprop = 'detritus_node'):
    OutletVolumeFlux = np.array([Graph.nodes[ID][Vprop] for ID in Graph.nodes])
    TotalVolumeFlux = np.nansum(OutletVolumeFlux)
    OutletDetritus = [Graph.nodes[ID][Dprop] for ID in Graph.nodes]
    keysLOL = [list(ID.keys()) for ID in OutletDetritus]
    D_keys = list({item for sublist in keysLOL for item in sublist})
    DetritalProps = {key: 0.0 for key in D_keys}
    for key in DetritalProps.keys():
        for i,j in enumerate(OutletDetritus):
           val1 = DetritalProps[key]
           try:
               val2 = j[key]*OutletVolumeFlux[i]/TotalVolumeFlux
           except(KeyError):
               val2 = 0.0
               #print('no value for key {} at outlet {}'.format(key,j))
           DetritalProps[key] = val1+val2
    Output[0].append(t)
    Output[1].append(OutletDetritus)
    Output[2].append(DetritalProps)

#function to draw a figure    
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(4,1, figsize = (8,8), dpi = 300)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'dHdt', lw = 0.25, minprop = -1e-8, maxprop = 1e-8,cmap = 'bwr', fig = fig, ax = axs[0], label = 'dHdt (m/s)', ordered = False)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'jammed', lw = 0.25, cmap = 'inferno', minprop= 0.0, maxprop = 1.0, fig = fig, ax = axs[1],label = 'jam status',ordered = True)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'till_thickness', lw = 0.25, cmap = 'inferno', minprop= 0.0, maxprop = MaxTillH, fig = fig, ax = axs[2],label = 'H (m)',ordered = False)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'phi_median', lw = 0.25, minprop = MinPhi, maxprop = MaxPhi, cmap = 'viridis_r', fig = fig, ax = axs[3],label = 'Phi',ordered = False)

# Read in pickle files and remake subgraphs. 

#overall graph (required)
with open('FinalNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    Network = pickle.load(file)
#level 0 sub-graph (required)    
with open('SubNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    SubNetCopy = pickle.load(file)

#We need to remake subgraphs so they are views of main Network
SubNet = nx.subgraph(Network,SubNetCopy.nodes)

#for the level 0 subgraph identify head and outlet nodes and their coords

# identify head nodes (nodes with <1 predecessor nodes)
Head_nodes = [j for i,j in enumerate(Network.pred) if len(Network.pred[j])<1]

#get any moulin nodes with status == 4.
node_status = Network.nodes(data = 'node_status')
Moulins = [i[0] for i in node_status if i[1]==4]

# identify end-point nodes (nodes with <1 successor nodes)
Endpoints = [j for i,j in enumerate(Network.succ) if len(Network.succ[j])<1]

Outlet_nodes = []
OOM_nodes = []
Sink_nodes = []
# select the outlet nodes by status to focus only on the outlet edge
for i,j in enumerate(Endpoints):
    if Network.nodes[j]['node_status'] == 0: #is an outlet node
        Outlet_nodes.append(j)
    elif Network.nodes[j]['node_status'] == 2: #is on the boundary
        OOM_nodes.append(j)
    else:
        Sink_nodes.append(j)

print('Number of Outlets, OOM and Sink nodes are {},{},{} respectively'.format(len(Outlet_nodes),len(OOM_nodes), len(Sink_nodes)))

Head_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Head_nodes)]
Outlet_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Outlet_nodes)]
OOM_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(OOM_nodes)]
Sink_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Sink_nodes)]
Moulins_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Moulins)]

#Make a subgraph involving just the output nodes (for tracking)
OutNet = nx.subgraph(Network,Endpoints) # here we track all endpoints

#Plot the L0 sub-network including nodes

NodeSets = {
            'Head nodes': Head_node_coords,
            'Moulin nodes': Moulins_coords
            'Outlet nodes': Outlet_node_coords,
            'OOM_nodes': OOM_node_coords,
            'Sink_nodes': Sink_node_coords,
            }

fig = PlotSubNetworksEdgesandNodes(Network,SubNet,NodeSets, lw = 0.1, ps = 1.6, label = 'L0 subgraph')
fn = os.path.join(OutputDir,ModelName+'GraphEdgesandNodes.png')
plt.savefig(fn)
plt.close

node_atts = set([k for n in Network.nodes for k in Network.nodes[n].keys()])

#Generate graticule property if detritus is bedrock property
if Dmode == 'NodeProp':
    #try to read the geology class
    if 'detritus_prop' not in node_atts: 
        Basement = Graticule_Attribute(Network, num_x = 5, num_y = 3)
        Basement_n = {key: float(Basement[key]) for key in Basement}
        nx.set_node_attributes(Network,Basement_n,'detritus_prop')
    #make a plot
    fig,ax = plt.subplots(figsize = (8,5), dpi = 300)
    PlotNetworkNodeProp(Network, SubNetworks = Network, prop = 'detritus_prop', lw = 0.05, ps = 0.4, cmap = 'viridis', fig = fig, ax = ax)
    fn = os.path.join(OutputDir,'DetritusProp.png')
    plt.savefig(fn, dpi = 300)
    plt.close()
#%% If all is good now we can initialise the model

#set start time
t = 0.0 #nominally this is one timestep before the 'start' of the model, could be a real datetime (in seconds) but we have zero

#Initialise the arrays to store model outputs
#here we will track at outlet nodes but this could be any node set 
#sediment volume discharge rate (m^3/s)
OutVol = [[0.0],[0.0],[0.0]]
#sediment volumetric concentration
OutConc = [[0.0],[0.0],[0.0]]
#grain size
OutGS = [[0.0],[0.0],[0.0]]
#detrital properties
OutDet = [[0.0],[{'init':0.0}],[0.0]]

#make the random number array of n samples for every edge every timesteps
rng = np.random.default_rng()

# Generate random number arrays for edges...there are always more edges than nodes so this will do for nodes too
nsteps = 1

# a normal distribution for phi
RNarray0 = rng.normal(size = (nsteps, SubNet.number_of_edges(),samp_n))

# initialise ray and put things into the ray memory
ray.shutdown() #close any existing ray instance
ray.init() #start a new one
RNA0 = ray.put(RNarray0) #put the RNA into the ray memory
del RNarray0
print("made RNA(s)")

# instantiate the L0 model(s) as an actor
sgst_kwargs = dict(potgrad_method = HPGMode,
               cflux_method = CFmode,
               erosion_method = Emode,
               transport_method = Tmode,
               detritus_method = Dmode,
               MaxTillH = MaxTillH,
               InitTillH = InitTillH,
               meanD = meanD, #m
               stdD = stdD,
               samp_n = samp_n,
               rhog = rhog, #kg m-3
               Dsig = Dsig, #m
               SedimentUptakeLengthFactor = sedl_factor,
               K = K,
               L = L,
               )

sgst0 = SGST.remote(SubNet, RNarray = ray.get(RNA0),**sgst_kwargs)
print("instantiated model on L0 subnet")

# initialise graph in serial - so far this is proving to be faster
InitSerial = sgst0.initialise_steady.remote()
t = 1 #just one second to get all variables on the graph
G0 = ray.get(sgst0.run_one_step_steady.remote(t))
print("initialised model on L0 subnet", flush = True)

mmtophi(G0)

# Make a figure
MakeFig(Network,G0)
fn = os.path.join(OutputDir,'model_init.png')
plt.savefig(fn)
plt.close()

#%% run the model up to a time limit expressed in weeks
week = 0
times = []

#variable dt by dHdT/H criterion, with a minimum a maximum and a break condition for very low dHdt
def dt_from_dHdt(G,min_dt,max_dt,target_maxdeltaH = 0.5, BreakCondition = 1e-11):
    A = nx.get_edge_attributes(G,'dHdt')
    dHdt = np.abs([A[key] for key in A.keys()])
    if np.nanmax(dHdt) <= BreakCondition:
        dt = np.inf #!
    A = nx.get_edge_attributes(G,'till_thickness')
    H = np.array([A[key] for key in A.keys()])
    #deal with small H to avoid instability, here 10 cm
    H = np.where(H<0.1,0.1,H)
    next_dt = target_maxdeltaH/np.nanmax(dHdt/H)
    if 'dt' in locals():
        dt = dt
    elif next_dt < min_dt:
        dt = min_dt
    elif next_dt > max_dt:
        dt = max_dt
    else:
        dt = next_dt
    return dt

cycle = 0
last_output_t = 1 #from initialisation
cycle_its = 6  #cycles here are 6 iterations...tracking is once per cycle
week_recording = 1 #how often do we want to record full results

while t < one_week*num_weeks:
    print('starting cycle {} at week {}, time {}'.format(cycle,week,int(t)), flush = True)
    #setup supervisor actor with Level 0 Subgraph
    sgst_super0 = SGST_supervisor.remote(G0,RNarray = ray.get(RNA0), n = n_proc)
    #make partitions and spawn worker actors for L0 graph
    sgst_super0.MakePartitions.remote()
    sgst_super0.PartitionGraph.remote()
    print('Spawning new actors: out-of-scope workers from previous cycles may be killed by ray', flush = True)
    actor_set0 = sgst_super0.SpawnWorkerActors.remote(sgst_kwargs)
    for m in range(0,cycle_its):
        #calculate dt
        dt = dt_from_dHdt(G0,min_dt,max_dt)
        #we break if we meet the break condition
        if dt == np.inf:
            times.append(t)
            G0 = ray.get(sgst_super0.get_graph.remote())
            #do tracking
            ON = nx.subgraph(G0,OutNet.nodes())
            TrackQ(t,ON,OutVol, prop = 'VSo')
            TrackConc(t,ON,OutConc, VWprop = 'VW',VSprop = 'VSo')
            TrackGS(t,ON,OutGS, GSprop = 'd_dist_node')
            TrackDet(t,ON, OutDet, Dprop = 'detritus_node')
            print('stopping run at t = {} and cycle {} because dHdt met break condition'.format(int(t),cycle))
            t = one_week*num_weeks
            break
        t+=dt
        week = t//one_week
        weeks_since_last_output = (t-last_output_t)//one_week
        if week % week_recording == 0 and weeks_since_last_output > 0: #if we hit this mark then truncate the time
            #truncate the time to match exactly the week    
            t = week*one_week
            last_output_t = t
        #run the model! here we use semi-strict harmonisation    
        sgst_super0.RunWorkerActorsSemiStrict.remote(actor_set0, t)
        if week % week_recording == 0 and weeks_since_last_output > 0: #if we hit this mark then report results
            print('writing out results at week {} and time {}'.format(week, int(t)))
            times.append(t)
            G0 = ray.get(sgst_super0.get_graph.remote())
            #do tracking
            ON = nx.subgraph(G0,OutNet.nodes())
            TrackQ(t,ON,OutVol, prop = 'VSo')
            TrackConc(t,ON,OutConc, VWprop = 'VW',VSprop = 'VSo')
            TrackGS(t,ON,OutGS, GSprop = 'd_dist_node')
            TrackDet(t,ON, OutDet, Dprop = 'detritus_node')
            # make an output figure
            mmtophi(G0)
            MakeFig(Network,G0)
            fn = os.path.join(OutputDir,'model_week_{}.png'.format(week))
            plt.savefig(fn)
            plt.close()
            # pickle results to file
            fn = os.path.join(OutputDir,'model_week_{}.pickle'.format(week))
            Output = G0.copy()
            with open(fn,'wb') as file:
                pickle.dump(Output, file)
            # pickle results for restart
            fn = os.path.join(OutputDir,'last_model.pickle')
            Output = G0.copy()
            with open(fn,'wb') as file:
                pickle.dump(Output, file)
            #pickle data for analysis
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
        #or get results and do tracking if at the end of the cycle        
        elif m == cycle_its-1:  
            times.append(t)
            G0 = ray.get(sgst_super0.get_graph.remote())
            #do tracking
            ON = nx.subgraph(G0,OutNet.nodes())
            TrackQ(t,ON,OutVol, prop = 'VSo')
            TrackConc(t,ON,OutConc, VWprop = 'VW',VSprop = 'VSo')
            TrackGS(t,ON,OutGS, GSprop = 'd_dist_node')
            TrackDet(t,ON, OutDet, Dprop = 'detritus_node')
        else:
            pass #just keep going
    cycle+=1

#One final timestep
print('final timestep run', flush = True)
dt = dt_from_dHdt(G0,min_dt,max_dt)
#we break if we meet the break condition
if dt == np.inf:
    times.append(t)
    G0 = ray.get(sgst_super0.get_graph.remote())
    #do tracking
    ON = nx.subgraph(G0,OutNet.nodes())
    TrackQ(t,ON,OutVol, prop = 'VSo')
    TrackConc(t,ON,OutConc, VWprop = 'VW',VSprop = 'VSo')
    TrackGS(t,ON,OutGS, GSprop = 'd_dist_node')
    TrackDet(t,ON, OutDet, Dprop = 'detritus_node')
    print('stopping run at t = {} and cycle {} because dHdt met break condition'.format(int(t),cycle))
    t = one_week*num_weeks
#otherwise we run the last model step
else:
    t+=dt
    times.append(t)
    sgst_super0 = SGST_supervisor.remote(G0, RNarray = ray.get(RNA0), n = n_proc)
     #make partitions and spawn worker actors
    sgst_super0.MakePartitions.remote()
    sgst_super0.PartitionGraph.remote()
    actor_set0 = sgst_super0.SpawnWorkerActors.remote(sgst_kwargs)
    #run once only
    sgst_super0.RunWorkerActorsSemiStrict.remote(actor_set0, t)
    G0 = ray.get(sgst_super0.get_graph.remote())
    #do tracking
    ON = nx.subgraph(G0,OutNet.nodes())
    TrackQ(t,ON,OutVol, prop = 'VSo')
    TrackConc(t,ON,OutConc, VWprop = 'VW',VSprop = 'VSo')
    TrackGS(t,ON,OutGS, GSprop = 'd_dist_node')
    TrackDet(t,ON, OutDet, Dprop = 'detritus_node')

#do the final output

MakeFig(Network,G0)
fn = os.path.join(OutputDir,'model_final_time_{}.png'.format(int(t)))
plt.savefig(fn)
plt.close()

#update the overall graph
Network.update(G0)

# pickle results for restart
fn = os.path.join(OutputDir,'last_model.pickle')
with open(fn,'wb') as file:
    pickle.dump(G0, file)

#output the final main graph
fn = os.path.join(OutputDir,'ModelResult.pickle')
with open(fn,'wb') as file:
        pickle.dump(Network, file)

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

#%% make some basic output plots - this step can safely be omitted

##%% make line plot of volume flux total and specific high-volume nodes
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[1][-1][n]>=OutVol[2][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutVol[1][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.5, label = key)
plt.plot(times,OutVol[2][1:], label = 'volume discharge rate (m^3/s)', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutVol.png')
plt.savefig(fn, dpi = 300)
plt.close()

# make line plot of sediment concentration
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[1][-1][n]>=OutVol[2][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutConc[1][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.5, label = key)
plt.plot(times,OutConc[2][1:], label = 'volumetric concentration', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutConc.png')
plt.savefig(fn, dpi = 300)
plt.close()

# and of grain size
fig = plt.figure()
values= {}
for n,key in enumerate(OutNet.nodes):
    if OutVol[1][-1][n]>=OutVol[2][-1]*0.02:
        values[key] = [u[n] for i,u in enumerate(OutGS[1][1:])]
for key in values:
    plt.plot(times,values[key], linewidth = 0.25, label = key)
plt.plot(times,OutGS[2][1:], label = 'total average grain size (m)', linewidth = 1, color = 'k')    
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutGS.png')
plt.savefig(fn, dpi = 300)
plt.close()

#and of volume flux by source
#plotting if we do not have bedrock detritus enabled
def MakeFig(OutDets,OutVols):
    fig,axs = plt.subplots(figsize = (6,4.5), dpi = 300)
    DetClasses = ['init','basal','basement']
    PlotOrder = [0,1,2]
    col = ['gray','lightyellow','lightsalmon']
    DetVolArray = np.zeros(shape = (len(DetClasses),len(OutDets[0])-1))
    Otimes = [n for n in OutVols[0][1:]]
    for i,j in enumerate(OutDets[2][1:]):
        VScale = OutVols[2][i+1]
        for m,n in enumerate(PlotOrder):
            key = DetClasses[m]
            DetVolArray[m][i] = j[key] * VScale
    axs.stackplot(Otimes,DetVolArray, labels = DetClasses, colors = col)
    axs.legend(fontsize = 'x-small', ncol = 3, loc = 'upper left')

#plotting function if we have detritus enabled
def MakeFigD(OutDets,OutVols):
    fig,axs = plt.subplots(figsize = (6,4.5), dpi = 300)
    DetClasses= ['init','basal',0.0,1.0,2.0,10.0,11.0,12.0,20.0,21.0,22.0,30.0,31.0,32.0,40.0,41.0,42.0]
    col = ['gray','lightyellow','lightsalmon','salmon','darksalmon','lightgreen','limegreen','lime','skyblue','deepskyblue','steelblue','mediumslateblue','slateblue','darkslateblue','thistle','violet','darkviolet']
    PlotOrder = [0,1,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
    DetVolArray = np.zeros(shape = (len(DetClasses),len(OutDets[0])-1))
    Otimes = [n for n in OutVols[0][1:]]
    for i,j in enumerate(OutDets[2][1:]):
        VScale = OutVols[2][i+1]
        for m,n in enumerate(PlotOrder):
            key = DetClasses[m]
            DetVolArray[m][i] = j[key] * VScale
    axs.stackplot(Otimes,DetVolArray, labels = DetClasses, colors = col)
    axs.legend(fontsize = 'x-small', ncol = 3, loc = 'upper left')

if Dmode == 'NodeProp':
    MakeFigD(OutDet,OutVol)
    fn = os.path.join(OutputDir,'OutDetVol.png')
    plt.savefig(fn,dpi = 300)
else:
    MakeFig(OutDet,OutVol)
    fn = os.path.join(OutputDir,'OutDetVol.png')
    plt.savefig(fn,dpi = 300)