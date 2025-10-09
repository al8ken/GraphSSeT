#!/usr/env/python

"""
This is a sample script to run the GraphSSeT model for a member of the SHMIP 'C' or 'D' model ensembles. 

Before you can run this script you will need to have made the input graphs by running the script 

SHMIP_graph_CDseries.py for the desired model output 

For the GraphSSeT model description see the paper of Aitken et al. (2024)

https://doi.org/10.5194/tc-18-4111-2024

codeauthor:: Alan Aitken

This version October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import networkx as nx
from NetworkX_funcs import *
from GraphSSeT_NetworkX_parallel import SubglacialErosionandSedimentFlux as SGST
from GraphSSeT_NetworkX_parallel import SGST_supervisor
import copy
import pickle
import ray
import sys
import os

#Model Information
try:
    InputModel = sys.argv[1]
except IndexError:
    InputModel = 'C1'    
try:
    ModelInstance = sys.argv[2]
except IndexError:
    ModelInstance = 'test'

ModelName = InputModel + "_" + ModelInstance

OutputDir = os.path.join("./Output/", ModelName)
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)

n_proc = 8 #number of ray processes

#Timestep control
max_dt = 60.*60.*24. #max 1 day
min_dt = 60.* 30. #min 30 minutes

#run control parameters (input as a string from command line, or use defaults if no (or incomplete) string provided
try:
    InitTillH = float(sys.argv[3]) #m
    MaxTillH = float(sys.argv[4]) #m
    ErodeLim = float(sys.argv[5]) #m
    meanD = float(sys.argv[6]) #mean Phi
    stdD = float(sys.argv[7]) # standard deviation of ln(grainsize)
    rhog = float(sys.argv[8]) #kg m-3
    sedl_factor = float(sys.argv[9]) #sediment uptake e-folding length
    Dsig = float(sys.argv[10]) #m^-1
    K = float(sys.argv[11]) #erosion law preexponent
    L = float(sys.argv[12]) #erosion law exponent
    samp_n = int(sys.argv[13]) #number of samples to define grain size populations...less is quicker but more variable
    dt = float(sys.argv[14])*3600 # input in hours
except IndexError:
    InitTillH = 0.25 #m
    MaxTillH = 1.0 #m
    ErodeLim = 0.75 #m
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

#define functions for later tracking and to make figures
#with variable graph-flow characteristics we track output at nodes written onto the graph

#volumetric flux rates
def TrackQ(t,Graph,Output = [[0,0],[0.0]], prop = 'VSo'):
    node_dt = np.array([Graph.nodes[key]['VW'] for key in Graph.nodes])/np.array([Graph.nodes[key]['QW'] for key in Graph.nodes]) # effective dt on the node
    OutletVolumeFlux = np.array([Graph.nodes[key][prop] for key in Graph.nodes])/node_dt
    TotalVolumeFlux = np.nansum(OutletVolumeFlux)
    #write output to graph
    TrackTs = nx.get_node_attributes(Graph,'Track_t_arr')
    if len(TrackTs) > 0: #i.e. object has data 
        print('adding new outputs to graph at time {}'.format(t))
        TrackTs = {item: TrackTs[item] + [t] for item in TrackTs}
    else:
        print('adding first outputs to graph at time {}'.format(t))
        TrackTs = {item: [t] for item in Graph.nodes()}
    nx.set_node_attributes(Graph,TrackTs,'Track_t_arr')
    OQF_arr = nx.get_node_attributes(Graph,'Q_out_arr')
    if len(OQF_arr) > 0:
        OQF_arr = {item: OQF_arr[item] + [OutletVolumeFlux[i]] for i,item in enumerate(OQF_arr)}
    else:
        OQF_arr = {item: [OutletVolumeFlux[i]] for i,item in enumerate(Graph.nodes())}
    nx.set_node_attributes(Graph,OQF_arr,'Q_out_arr')
    #output total to outputs object
    Output[0].append(t)
    Output[1].append(TotalVolumeFlux)

#volumetric concentration
def TrackConc(t,Graph,Output = [[0.0],[0.0]], VWprop = 'VW', VSprop = 'VSo'):
    OutletVSFlux = np.array([Graph.nodes[key][VSprop] for key in Graph.nodes])
    TotalVSFlux = np.nansum(OutletVSFlux)
    OutletVWFlux = np.array([Graph.nodes[key][VWprop] for key in Graph.nodes])
    TotalVWFlux = np.nansum(OutletVWFlux)
    OutletSConc = OutletVSFlux/OutletVWFlux
    TotalSConc = TotalVSFlux/TotalVWFlux
    #write output to graph
    C_arr = nx.get_node_attributes(Graph,'Conc_arr')
    if len(C_arr) > 0:
        C_arr = {item: C_arr[item] + [OutletSConc[i]] for i,item in enumerate(C_arr)}
    else:
        C_arr = {item: [OutletSConc[i]] for i,item in enumerate(Graph.nodes())}
    nx.set_node_attributes(Graph,C_arr,'Conc_arr')
    #output to outputs object
    Output[0].append(t)
    Output[1].append(TotalSConc)

#volume-weighted grain sizes
def TrackGS(t,Graph,Output = [[0,0],[0.0]], Vprop = 'VSo', GSprop = 'd_dist_node'):
    OutletVolumeFlux = np.array([Graph.nodes[key][Vprop] for key in Graph.nodes])
    MeanOutletGS = np.array([2**-Graph.nodes[key][GSprop][0] for key in Graph.nodes]) #output in mm 
    MeanTotalGS = np.nansum(OutletVolumeFlux*MeanOutletGS)/np.sum(OutletVolumeFlux)
    #write output to graph
    GS_arr = nx.get_node_attributes(Graph,'GS_arr')
    if len(GS_arr) > 0:
        GS_arr = {item: GS_arr[item] + [MeanOutletGS[i]] for i,item in enumerate(GS_arr)}
    else:
        GS_arr = {item: [MeanOutletGS[i]] for i,item in enumerate(Graph.nodes())}
    #write output to graph
    nx.set_node_attributes(Graph,GS_arr,'GS_arr')
    #output to outputs object
    Output[0].append(t)
    Output[1].append(MeanTotalGS)

#volume-weighted detritus proportions
def TrackDet(t,Graph,Output = [[],[]], Vprop = 'VSo', PropKeys = ['init','basal','basement'], Dprop = 'detritus_node'):
    OutletVolumeFlux = np.array([Graph.nodes[ID][Vprop] for ID in Graph.nodes])
    TotalVolumeFlux = np.nansum(OutletVolumeFlux)
    OutletDetritus = [Graph.nodes[ID][Dprop] for ID in Graph.nodes]
    DetritalProps = {key: 0.0 for key in PropKeys}
    for key in DetritalProps.keys():
        for i,j in enumerate(OutletDetritus):
           val1 = DetritalProps[key]
           try:
               val2 = j[key]*OutletVolumeFlux[i]/TotalVolumeFlux
           except(KeyError):
               val2 = 0.0
           DetritalProps[key] = val1+val2
    #write output to graph
    Det_arr = nx.get_node_attributes(Graph,'detritus_arr')
    if len(Det_arr) > 0:
        Det_arr = {item: Det_arr[item] + [OutletDetritus[i]] for i,item in enumerate(Det_arr)}
    else:
        Det_arr = {item: [OutletDetritus[i]] for i,item in enumerate(Graph.nodes())}
    #write output to graph
    nx.set_node_attributes(Graph,Det_arr,'detritus_arr')
    Output[0].append(t)
    Output[1].append(DetritalProps)

# a function to run the tracking functions
def DoTracking(t,Graph,Outputs, PropKeys):
    TrackQ(t,Graph,Outputs[0], prop = 'VSo') #suspect here we have issues with dt as the dt for an edge may not be timestep dt 
    TrackConc(t,Graph,Outputs[1], VWprop = 'VW',VSprop = 'VSo')
    TrackGS(t,Graph,Outputs[2], GSprop = 'd_dist_node')
    TrackDet(t,Graph,Outputs[3], PropKeys = PropKeys,Dprop = 'detritus_node')

#function to draw a figure    
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(4,1, figsize = (8,8), dpi = 300)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'dHdt', lw = 0.25, minprop = -1e-8, maxprop = 1e-8,cmap = 'bwr', fig = fig, ax = axs[0], label = 'dHdt (m/s)', ordered = False)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'jammed', lw = 0.25, cmap = 'inferno', minprop= 0.0, maxprop = 1.0, fig = fig, ax = axs[1],label = 'jam status',ordered = True)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'till_thickness', lw = 0.25, cmap = 'inferno', minprop= 0.0, maxprop = MaxTillH, fig = fig, ax = axs[2],label = 'H (m)',ordered = False)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'phi_median', lw = 0.25, minprop = MinPhi, maxprop = MaxPhi, cmap = 'viridis_r', fig = fig, ax = axs[3],label = 'Phi',ordered = False)

#functions to work with graph step transitions

def ReformGraph(G_before,step):
    ''' This function will rebuild the graph when the input model changes. 
    
    This can involve the following

    #changes to edge status (e.g. by becoming floating or outlet edge)
    #changes to node status (e.g. outlet or head nodes may change, moulins may emerge)
    #changes to edge direction (reversal of hydraulic potential gradient)
    #changes to forcing prameters (change in channel flux area etc)

    It does not allow the following:
        
    #changes to mesh structure (no removing nodes/edges in main graph)
        > subgraphs will be edited
    #changes to geometry (coordinates, lengths, widths are assumed static)

    #In future versions this will probably be built into GraphSSeT '''
    
    #edge directions from initial graph
    eds = nx.get_edge_attributes(G_before, 'direction_arr')
    keys = [key for key in eds]
    old_d = np.array([eds[key][step-1] for key in eds])
    new_d = np.array([eds[key][step] for key in eds])
    
    #make the new graph as a copy of the old one
    G_after = G_before.copy() 
    changed_dir = np.where(new_d != old_d)[0]
    print('{} edges changed direction'.format(len(changed_dir)))
    #for edges that changed direction we need to swap the key - i.e. (1,2) becomes (2,1) 
    if len(changed_dir) > 0:
        changed_keys = [j for i,j in enumerate(keys) if i in changed_dir]
        BE_Net = G_before.edge_subgraph(changed_keys).copy()
        #reverse this subgraph
        G_rev = BE_Net.reverse()
        #remove any reversed edges in G_after.
        G_after.remove_edges_from(G_rev.edges())
        #then add back the reversed edges
        G_after.add_edges_from(G_rev.edges(data = True))
    #these are the time variable properties we wish to update
    edge_props = ['status','weight','direction','channel_flux','channel_area', 'hyd_pot_grad']
    node_props = ['node_status','hydraulic_potential','effective_pressure','h_sheet']
    #update the node props from the array - all nodes should exist
    for prop in node_props:
        prop_arr = prop + '_arr'
        v = nx.get_node_attributes(G_before, prop_arr)
        v_a = np.array([v[key][step+1] for key in v])
        v_dict = {key: v_a[i] for i,key in enumerate(v)}
        nx.set_node_attributes(G_after,v_dict,prop)    
    #edge_props:
    if len(changed_dir == 0):
        #with no change in edge directions we can update properties directly
        for prop in edge_props:
            prop_arr = prop + '_arr'
            v = nx.get_edge_attributes(G_before, prop_arr)
            v_a = np.array([v[key][step+1] for key in v])
            v_dict = {key: v_a[i] for i,key in enumerate(v)}
            nx.set_edge_attributes(G_after,v_dict,prop)    
    else:
       #here we have changing directions to attend to
       for prop in edge_props:
           prop_arr = prop + '_arr'
           #set first on the edges that are the same way around
           dv = nx.get_edge_attributes(G_before,prop_arr)
           dv_a = np.array([dv[key][step+1] for key in dv])
           dv_dict = {key: dv_a[i] for i,key in enumerate(dv)}
           nx.set_edge_attributes(G_after,dv_dict,prop)
           #then on the reverse edges
           rv = nx.get_edge_attributes(G_rev,prop_arr)
           if prop == 'hyd_pot_grad': #sign matters here
               rv_a = np.array(-[rv[key][step+1] for key in rv])
           else: #sign does not matter for the rest
               rv_a = np.array([rv[key][step+1] for key in rv])
           rv_dict = {key: rv_a[i] for i,key in enumerate(rv)}
           nx.set_edge_attributes(G_after,rv_dict,prop)
    return G_after    

#functions to transfer edge and node properties 
#these functions accommodate edges that may have reversed direction

def TransferEdgeProp(recipient,donor,donor_rev,prop):
    # first do the attribute for right way edges
    dv = nx.get_edge_attributes(donor,prop)
    nx.set_edge_attributes(recipient,dv,prop)
    #none of the state variables are directional so we can transfer reversed edges the same way
    rv = nx.get_edge_attributes(donor_rev,prop)
    nx.set_edge_attributes(recipient,rv,prop)

def TransferNodeProp(recipient,donor,prop):
    #this will get values at all shared nodes
    dv = nx.get_node_attributes(donor,prop)
    nx.set_node_attributes(recipient,dv,prop)

#functions to update state variables when we need to 
def UpdateEdgeStateVariables(Recipient,Donor,props = []):
    #reverse the donor graph
    Donor_rev = Donor.reverse()
    for prop in props:
        TransferEdgeProp(Recipient,Donor,Donor_rev,prop)

def UpdateNodeStateVariables(Recipient,Donor,props = []):
    for prop in props:
        TransferNodeProp(Recipient,Donor,prop)

#setup containers for tracking
OutQ = [[0.0],[0.0]]
OutConc = [[0.0],[0.0]]
OutGS = [[0.0],[0.0]]
OutDet = [[0.0],[{'init':0.0}]]

Outputs_Outlets = [OutQ,OutConc,OutGS,OutDet]
Outputs_OOM = copy.deepcopy(Outputs_Outlets)
Outputs_Sinks = copy.deepcopy(Outputs_Outlets)

#%% Now we can get started with the model setup
# Read in pickle files. 
#overall graph (required)
with open('FinalNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    Network = pickle.load(file)
#level 0 sub-graph (required)    
with open('SubNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    SubNetCopy = pickle.load(file)

#We need to remake the subgraph so as a view of main Network
SubNet = nx.subgraph(Network,SubNetCopy.nodes)

# identify head nodes for the subgraph (nodes with <=1 predecessor nodes)
Head_nodes = [j for i,j in enumerate(SubNet.pred) if len(SubNet.pred[j])<1]

#get any moulin nodes with status == 4.
node_status = Network.nodes(data = 'node_status')
Moulins = [i[0] for i in node_status if i[1]==4]

#endpoint nodes with no succs
Endpoints = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])<1]

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

print('Number of Head, Outlet, OOM and Sink nodes are {},{},{},{} respectively'.format(len(Head_nodes),len(Outlet_nodes),len(OOM_nodes), len(Sink_nodes)))

Head_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Head_nodes)]
Outlet_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Outlet_nodes)]
OOM_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(OOM_nodes)]
Sink_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Sink_nodes)]
Moulins_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Moulins)]

PropKeys = ['init','basal','basement']
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
#%% Initialise the model
#set start time
t0 = 0.0 #nominally this is one timestep before the 'start' of the model, could be a real time (in seconds) but we have zero here

#make the random number array of n samples for every edge every timesteps
rng = np.random.default_rng()

# Generate random number arrays for edges...there are always more edges than nodes so this will do for nodes too
nsteps = 1 #nsteps will give a number after which the array is shuffled

# a normal distribution for phi gives lognormal grainsize
RNarray = rng.normal(size = (nsteps, Network.number_of_edges(),samp_n))

# initialise ray and put things into the namespace
ray.shutdown()
ray.init()
RNA = ray.put(RNarray)
del RNarray
print("made RNA")

# instantiate the Network-scale model as an actor
sgst_kwargs = dict(potgrad_method = HPGMode,
               cflux_method = CFmode,
               erosion_method = Emode,
               transport_method = Tmode,
               detritus_method = Dmode,
               SedimentUptakeLengthFactor = sedl_factor,
               MaxTillH = MaxTillH,
               InitTillH = InitTillH,
               Herodelim=ErodeLim,
               meanD = meanD, #m
               stdD = stdD,
               samp_n = samp_n,
               rhog = rhog, #kg m-3
               Dsig = Dsig, #m
               K = K,
               L = L,
               )

sgst0 = SGST.remote(SubNet, RNarray = ray.get(RNA),**sgst_kwargs)
print("instantiated model")
# initialise graph in serial - so far this is proving to be faster
sgst0.initialise_steady.remote()
t = t0+1 #just one second
G0 = ray.get(sgst0.run_one_step_steady.remote(t))
print("initialised model on L0 subnet")

#here update works OK..later we need to consider reversed edges.
Network.update(G0)
NET = ray.put(Network)

mmtophi(G0)

# Make a figure
MakeFig(Network,G0)
fn = os.path.join(OutputDir,'model_init.png')
plt.savefig(fn)
plt.close()
#%% Run in cycles with each cycle running until the next input slice time is reached

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

last_output_t = 1 #from initialisation

# input model time-slice times
ts = nx.get_edge_attributes(SubNet,'times_arr')
slice_ts = np.rint(np.array([ts[key] for key in ts])[0]) 
times = [] # these are the times for tracking

#when do we want to output results
OutputSteps = [0,8,16,24,32,40]

#launch the ray supervisor
sgst_super0 = SGST_supervisor.remote(G0,RNarray = ray.get(RNA), n = n_proc)

#now run the model through all steps
for i,j in enumerate(slice_ts[0:-1]):
    step = i
    t_next_step = slice_ts[step+1]
    print('step {} from time {} to max {}, t = {}'.format(step,j,t_next_step,t))
    
    #Every step we rebuild the graph 
    if step == 0:
        Net = G0
        Network = ray.get(NET)
    else:
        Network = ray.get(NET)
        Net = ReformGraph(Network,step)
        #we need to re-initalise after reformation
        sgst0 = SGST.remote(Net, RNarray = ray.get(RNA),**sgst_kwargs)
        sgst0.initialise_steady.remote()
        Net = ray.get(sgst0.get_graph.remote())
        print('reinitialised model for new timestep')
        
    # Identify head nodes (nodes with <1 predecessor nodes)
    Head_nodes = [j for i,j in enumerate(Net.pred) if len(Net.pred[j])<1]
    # identify end-point nodes (nodes with <1 successor nodes)
    Endpoints = [j for i,j in enumerate(Net.succ) if len(Net.succ[j])<1]
        
    Outlet_nodes = []
    OOM_nodes = []
    Sinks = []
    HasOutlets = False
    HasOOMs = False
    HasSinks = False
    
    # select the outlet nodes by status to focus only on the outlet edge
    for i,j in enumerate(Endpoints):
        if Net.nodes[j]['node_status'] == 0: #is a designated outlet node
            Outlet_nodes.append(j)
            HasOutlets = True
        elif Net.nodes[j]['node_status'] == 2: #is boundary
            OOM_nodes.append(j)
            HasOOMs = True
        else:
            Sinks.append(j)
            HasSinks = True
    
    #get any moulin nodes with status == 4
    node_status = Net.nodes(data = 'node_status')
    Moulins = [i[0] for i in node_status if i[1]==4]

    #collate input nodes - removing duplicates. Priority is Moulins,Head then Random
    Head_nodes = [j for i, j in enumerate(Head_nodes) if j not in Moulins]
    In_nodes = Moulins+Head_nodes

    #Weighted path analyses for these in/out nodes (using the whole network) optionally with a maximum weight of x each way
    SubNet = Dijkstra_SubNetSometoSome(Net,In_nodes[:],Endpoints[:], Weight = 'weight2')
    G0 = SubNet.copy()
    
    #reduce array props to save size
    edge_props = ['edge_status_arr','weights_arr','direction_arr','channel_flux_arr','channel_area_arr', 'hyd_pot_grad_arr', 'times_arr']
    node_props = ['node_status_arr','hydraulic_potential_arr','effective_pressure_arr','h_sheet_arr']
    
    for prop in edge_props:
        nx.set_edge_attributes(G0,0.0,name=prop)
    
    for prop in node_props:
        nx.set_node_attributes(G0,0.0,name=prop)
        
    print('Reformed graph from base Network')
    #we need here to 'put' G0 into the supervisor
    sgst_super0.set_graph.remote(G0)
    
    #setup supervised actors with Level 0 Subgraph
    sgst_super0.MakePartitions.remote()
    sgst_super0.PartitionGraph.remote()
    actor_set0 = sgst_super0.SpawnWorkerActors.remote(sgst_kwargs)
    print('launched new worker actor set')
    iteration = 0
    while t < t_next_step:
        dt = dt_from_dHdt(G0,min_dt,max_dt)
        t+=dt
        if t > t_next_step:
            dt = t_next_step-t
            t = t_next_step
        sgst_super0.RunWorkerActorsSemiStrict.remote(actor_set0, t)
        # ray.get() results on first, then every nth iteration, and do tracking
        if iteration % 10 == 0: 
            times.append(t)    
            G0 = ray.get(sgst_super0.get_graph.remote())    
            if HasOutlets:
                OutNet = nx.subgraph(G0,Outlet_nodes)
                DoTracking(t,OutNet,Outputs_Outlets, PropKeys=PropKeys)
            if HasOOMs:
                OOMNet = nx.subgraph(G0,OOM_nodes)
                DoTracking(t,OOMNet,Outputs_OOM, PropKeys=PropKeys)
            if HasSinks:
                SinkNet = nx.subgraph(G0,Sinks)
                DoTracking(t,SinkNet,Outputs_Sinks, PropKeys=PropKeys)
            #update supervisor graph from G0 (with newly added info)
            sgst_super0.set_graph.remote(G0)
        iteration+=1
    #if not done already, get the last iteration for the step
    if t > times[-1]:
        times.append(t)
        G0 = ray.get(sgst_super0.get_graph.remote())
        if HasOutlets:
            #print('doing Outlet tracking')
            OutNet = nx.subgraph(G0,Outlet_nodes)
            DoTracking(t,OutNet,Outputs_Outlets, PropKeys=PropKeys)
        if HasOOMs:
            #print('doing OOM tracking')
            OOMNet = nx.subgraph(G0,OOM_nodes)
            DoTracking(t,OOMNet,Outputs_OOM, PropKeys=PropKeys)
        if HasSinks:
           # print('doing Sink tracking')
            SinkNet = nx.subgraph(G0,Sinks)
            DoTracking(t,SinkNet,Outputs_Sinks, PropKeys=PropKeys)
        #update supervisor graph from G0 (with newly added info)
        sgst_super0.set_graph.remote(G0)
    print('Done for this step with {} iterations and t = {}'.format(iteration,t))
    
    #transfer state variables from G0 to Network
    edge_props = ['d_median',
                  'd_distribution',
                  'sed_d_distribution',
                  'till_thickness',
                  'flux_density',
                  'sed_d_median',
                  'detritus',
                  'sed_detritus',
                  'last_time',
                  'phi_median',
                  ]
    
    UpdateEdgeStateVariables(Network,G0,props = edge_props)  
    
    node_props = ['bed_elevation',
                  'bedrock_elevation',
                  'd_dist_node',
                  'detritus_node',
                  'Track_t_arr',
                  'Q_out_arr',
                  'Conc_arr',
                  'GS_arr',
                  'detritus_arr',
                  ]

    UpdateNodeStateVariables(Network,G0,props = node_props)  
    
    NET = ray.put(Network)
    print('edge and node props transferred to base Network', flush = True)
    
    mmtophi(G0)
    MakeFig(Net,G0) 
    fn = os.path.join(OutputDir,'model_EOS_step_{}_time_{}.png'.format(step,t))
    plt.savefig(fn)
    plt.close()
    
    if step in OutputSteps:
        fn = os.path.join(OutputDir,'model_at_step_{}.pickle'.format(step))
        with open(fn,'wb') as file:
            pickle.dump(G0, file)
    # pickle results for restart
    fn = os.path.join(OutputDir,'last_model.pickle')
    with open(fn,'wb') as file:
        pickle.dump(G0, file)
    if HasOutlets:
        fn = os.path.join(OutputDir,'Outputs_Outlets.pickle')
        with open(fn,'wb') as file:
            pickle.dump(Outputs_Outlets, file)
    if HasOOMs:
        fn = os.path.join(OutputDir,'Outputs_OOM.pickle')
        with open(fn,'wb') as file:
            pickle.dump(Outputs_OOM, file)
    if HasSinks:
        fn = os.path.join(OutputDir,'Outputs_Sinks.pickle')
        with open(fn,'wb') as file:
            pickle.dump(Outputs_Sinks, file)    
    del Network
    del G0
    
# Final Plotting
print('make final plots')
Network = ray.get(NET)            

# make line plot of volume flux total and high-volume nodes - Outlets only here
OutQ = Outputs_Outlets[0]
OutConc = Outputs_Outlets[1]
OutGS = Outputs_Outlets[2]
OutDet = Outputs_Outlets[3]    

#t_spans
diffs = np.diff(OutQ[0])
t_span = [0]+[diffs[i-1]/2+diffs[i]/2 for i,j in enumerate(OutQ[0][1:])]

# plot volume flux through time

Vs = [j*OutQ[1][i] for i,j in enumerate(t_span)]
CS_Vs = np.cumsum(Vs)

fig = plt.figure()
plt.plot(OutQ[0][1:],Vs[1:], label = 'sediment volume',linewidth = 0.5, color = 'gray')
plt.plot(OutQ[0][1:],CS_Vs[1:], label = 'cumulative volume', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutVol.png')
plt.savefig(fn, dpi = 300)
plt.close()

# make line plot of sediment concentration

fig = plt.figure()
plt.plot(OutConc[0][1:],OutConc[1][1:], label = 'concentration',linewidth = 0.5, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutConc.png')
plt.savefig(fn, dpi = 300)
plt.close()

# and of grain size
# volume weights
CS_GS = np.cumsum([j*Vs[i] for i,j in enumerate(OutGS[1])])/CS_Vs

fig = plt.figure()
plt.plot(OutGS[0][1:],OutGS[1][1:], label = 'grain size',linewidth = 0.5, color = 'gray')
plt.plot(OutGS[0][1:],CS_GS[1:], label = 'cumulative grain size', linewidth = 1, color = 'k')
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutGS.png')
plt.savefig(fn, dpi = 300)
plt.close()

# and of cumulative volume flux by detritus source
if Dmode == 'SedErod':
    DetClasses = ['init','basal','basement']
    col = ['gray','lightyellow','lightsalmon']
elif Dmode == 'NodeProp':
    DetClasses = ['init','basal']+list(set([Network.nodes[n]['detritus_prop'] for n in Network.nodes]))
    col1 = ['gray','lightyellow']
    if len(DetClasses)<9:
        col2 = list(mpc.BASE_COLORS)
        col = col1+col2[:len(DetClasses)-2]
    elif len(DetClasses)<19:
            col2a = list(mpc.BASE_COLORS)
            col2b = list(mpc.TABLEAU_COLORS)
            col = col1+col2a[:6]+col2b[:len(DetClasses)-8]
    else:
        try:
            cmap = colormaps['viridis']
        except KeyError:
            cmap = colormaps['greys'] #can recolorize from grayscale
        col2 = cmap(np.linspace(0, 1, len(DetClasses)-2))
        col = col1+col2        
DetVolArray = np.zeros(shape = (len(DetClasses),len(OutDet[0][1:])))
for i,j in enumerate(OutDet[1][1:]):
    for m,n in enumerate(DetClasses):
        try:
            DetVolArray[m][i] = DetVolArray[m][i-1]+j[n] * Vs[i+1]
        except KeyError:
            DetVolArray[m][i] = DetVolArray[m][i-1]
plt.stackplot(OutDet[0][1:],DetVolArray, labels = DetClasses, colors = col)
plt.legend(fontsize = 'x-small', ncol=6, loc = 'upper left')
fn = os.path.join(OutputDir,'OutDetVol.png')
plt.savefig(fn, dpi = 300)
plt.close()
print('plotting completed')