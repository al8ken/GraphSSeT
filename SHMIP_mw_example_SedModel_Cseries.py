import numpy as np
import random
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
meanD = (2**-phi)/1000 # input is phi so convert to mm
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
    DetritalProps = {key: 0.0 for key in OutletDetritus[0].keys()}
    for key in DetritalProps.keys():
        for i,j in enumerate(OutletDetritus):
           val1 = DetritalProps[key]
           val2 = j[key]*OutletVolumeFlux[i]/TotalVolumeFlux
           DetritalProps[key] = val1+val2
    Output[0].append(t)
    Output[1].append(OutletDetritus)
    Output[2].append(DetritalProps)

# Function to make a Figure - here we plot flux density, jam status, till thickness and grain size but any edge property could be plotted
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(4,1, figsize = (8,8), dpi = 300)
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = 'K', prop = 'flux_density', lw = 0.75, cmap = 'inferno', minprop= 0, maxprop = 1e-4, fig = fig, ax = axs[0])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = 'jam status', prop = 'jammed', lw = 0.75, cmap = 'inferno', minprop= 0.0, maxprop = 1.0, fig = fig, ax = axs[1])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = '$h_s$', prop = 'till_thickness', lw = 0.75, cmap = 'inferno', minprop= 0.0, maxprop = 0.75, fig = fig, ax = axs[2])
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = '$d_50$', prop = 'd_median', lw = 0.75, minprop = 0, maxprop = 5e-4, cmap = 'inferno', fig = fig, ax = axs[3])

#%% Read in pickle files and remake subgraphs. 
#overall graph (required)
with open('FinalNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    Network = pickle.load(file)
#level 0 sub-graph (required)    
with open('SubNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
    SubNetCopy = pickle.load(file)
#L1 and L2 subgraphs (required here but not in general)
with open('EBC0Network_SHMIP_'+InputModel+'.pickle','rb') as file:
    EBC0Copy = pickle.load(file)
with open('EBCMinNetwork_SHMIP_'+InputModel+'.pickle','rb') as file:
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

#%%also define the input nodes
Head_nodes = [j for i,j in enumerate(SubNet.pred) if len(SubNet.pred[j])<=1]
Head_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Head_nodes)]

#get any moulin nodes with status == 4
node_status_arr = Network.nodes(data = 'node_status_arr')
Moulins = [i[0] for i in node_status_arr if np.any(i[1]==4)]
Moulins_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Moulins)]

# New random seed node set
NetNotOut_nodes = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])>0]
NetNotOut_nodes_k = random.choices(NetNotOut_nodes, k = 100)
NetNotOut_nodes_k_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(NetNotOut_nodes_k)]

#collate all input nodes - removing duplicates. Priority is Moulins,Head then Random
Head_nodes = [j for i, j in enumerate(Head_nodes) if j not in Moulins]
NetNotOut_nodes_k = [j for i, j in enumerate(NetNotOut_nodes_k) if j not in Head_nodes]
In_nodes = Moulins+Head_nodes+NetNotOut_nodes_k

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


#Generate graticule property if detritus is bedrock property
if DetMode == 'NodeProp':
    Basement = Graticule_Attribute(Network, num_x = 5, num_y = 3)
    Basement_n = {key: float(Basement[key]) for key in Basement}
    nx.set_node_attributes(Network,Basement_n,'detritus_prop')

#%%instantiate the sugset model - Here we just have one as the model is reinitialised every timestep
sgst = SGST(SubNet,
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

#%% run once with whole network to make sure everything is the same time and has all variables
#just one second - we use the 'dynamic' mode of SGST
sgst.run_one_step_dynamic(t+1)

MakeFig(Network,SubNet)
fn = os.path.join(OutputDir,'model_initial.png')
plt.savefig(fn)
plt.close()

#%% Define functions to control dynamic run

#this function gets an edge property from the graph
def GetEdgePropArrayValue(Graph,prop_array,n):
    a = nx.get_edge_attributes(Graph, prop_array)
    arr = np.array([a[key][n] for key in a])
    return arr

#this function gets a node property from the graph
def GetNodePropArrayValue(Graph,prop_array,n):
    a = nx.get_node_attributes(Graph, prop_array)
    arr = np.array([a[key][n] for key in a])
    return arr

#this function makes a subgraph based on an edge property
def EdgeValTandC(graph, val_arr, thresh):
    EdgeSel = {j:val_arr[j] for i,j in enumerate(val_arr) if val_arr[j]>thresh}
    G = graph.edge_subgraph(EdgeSel)
    return G

#In this function every timestep we build a new main graph, and from this define the requird subgraphs
#this is needed because edges can change direction and also can change statues (e.g. by becoming floating)
#n is the index of the desired layer in the input model 
def UpdateGraph(Graph, SubGraph, n, edgeprops = [None], nodeprops = [None]):
    #Graph is the unfrozen main graph, Subgraph is the view of that Graph
    #node keys
    node_keys = np.array(SubGraph.nodes)
    #Get direction information for this layer
    a =  nx.get_edge_attributes(SubGraph, 'direction')
    edge_keys = np.array([key for key in a])
    dirs = np.array([a[key] for key in a])
    new_dirs = GetEdgePropArrayValue(SubGraph,'direction_arr',n)
    #update direction property on graph
    values = {(u,v): new_dirs[i] for i,(u,v) in enumerate(edge_keys)}
    nx.set_edge_attributes(SubGraph,values,'direction')
    #get new edge status for this layer
    new_edge_status = GetEdgePropArrayValue(SubGraph,'edge_status_arr',n)
    #update on graph
    values = {(u,v): new_edge_status[i] for i,(u,v) in enumerate(edge_keys)}
    nx.set_edge_attributes(SubGraph,values,'status')
    #get new node status for this layer
    new_node_status = GetNodePropArrayValue(SubGraph,'node_status_arr',n)
    #update on graph
    values = {j: new_node_status[i] for i,j in enumerate(node_keys)}
    nx.set_node_attributes(SubGraph,values,'node_status')
    
    #update edge and node values for given properties. We assume that each prop has a corresponding prop_arr with a value at 'n'
    for prop in edgeprops:
        prop_array = prop+'_arr'
        prop_value =  GetEdgePropArrayValue(SubGraph,prop_array,n)
        values = {(u,v): prop_value[i] for i,(u,v) in enumerate(edge_keys)}
        nx.set_edge_attributes(SubGraph,values,prop)
    
    for prop in nodeprops:
        prop_array = prop+'_arr'
        prop_value =  GetNodePropArrayValue(SubGraph,prop_array,n)
        values = {j: prop_value[i] for i,j in enumerate(node_keys)}
        nx.set_node_attributes(SubGraph,values,prop)
    
    #check if the direction of any edges has changed and if so replace with the other direction
    #this has to happen on the main (unfrozen) graph not a view. It is wise to work on a copy!!!!!!!
    new_dir_edges = np.where(new_dirs != dirs)[0]
    if new_dir_edges.size > 0:
        keys = edge_keys[new_dir_edges]
        #print('the following edges have changed direction and are replaced: {}'.format(keys))
        for key in keys:
            atts = Graph[key[0]][key[1]]
            Graph.remove_edge(key[0],key[1])
            Graph.add_edge(key[1],key[0])
            Graph[key[1]][key[0]].update(atts)
    
    #edge status changes
    #check if any edges are now entirely floating and remove from edge list 
    new_floating_edges = np.where(new_edge_status == -1)[0]
    if new_floating_edges.size > 0:
        keys = edge_keys[new_floating_edges]
        #print('the following edges are floating and not included: {}'.format(keys))
        all_other_edges = np.where(new_edge_status != -1)[0]
        edge_keys = edge_keys[all_other_edges]
        #make a new subgraph with just the retained edges
        SubGraph = nx.edge_subgraph(Graph,edge_keys) 
    
    #rescale weight by a property as (1-prop/max_of_prop)**n...in this case we use channel area with n = 3
    a = nx.get_edge_attributes(SubGraph,'channel_area')
    a_arr = [np.nanmin(a[key]) for key in a]
    w = (1.0-a_arr/np.nanmax(a_arr))**3
    new_weight = {key: w[i] for i,key in enumerate(a)}
    nx.set_edge_attributes(SubGraph, new_weight, 'weight2')
    
    #return the new Outlet nodes
    Out_edges = np.where(new_edge_status == 0)[0]
    Out_nodes = np.unique([v for i,(u,v) in enumerate(edge_keys[Out_edges])])
    OutNet = nx.subgraph(Graph,Out_nodes)
    return SubGraph,OutNet    

#%% run the model for the selected timeslices
#Access timestep_times
a = Network.edges(data = 'times_arr')
times = [j[2]+2 for i,j in enumerate(a)][0]

#Get the length of the model run in days
n_days = int(times[-1]//(24*3600))+1

#time variable properties on edges and nodes
edge_properties = ['channel_area','channel_flux', 'hyd_pot_grad']
node_properties = ['h_sheet','hydraulic_potential', 'effective_pressure']

#Now we run n_days of 8 3 hour cycles
#We must reinitialise the graph at very step
#For L1 and L2 we also recalculate centrality and reformulate the subgraph
n=0 
for day in range(0,n_days, 1):
    for cycle in range(0,8,1):
        #L2 first
        t = times[n]
        #update graph
        SubNet,OutNet = UpdateGraph(Network,SubNet,n, edgeprops = edge_properties, nodeprops = node_properties)        
        #recompute betweenness centrality
        SEBC = SomeEdgeBetweennessCentrality(SubNet, In_nodes, Out_nodes, weight = 'weight2', Norm = True)
        #L2 Subgraph EBCMInCent
        EBCMinCent = EdgeValTandC(Network,SEBC,0.005)
        # run on L2 subgraph
        sgst = SGST(EBCMinCent,
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
                    #W = W
                    )
        sgst.run_one_step_dynamic(t)
        TrackVol(t,OutNet,OutVol, prop = 'VSo')
        TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
        TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
        TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
        n = n+1
        #Now L1
        #update t and graph
        t = times[n]
        SubNet,OutNet = UpdateGraph(Network,SubNet,n, edgeprops = edge_properties, nodeprops = node_properties)
        #recompute betweenness centrality
        SEBC = SomeEdgeBetweennessCentrality(SubNet, In_nodes, Out_nodes, weight = 'weight2', Norm = True)
        #make new L1 Subgraph EBC0
        EBC0 = EdgeValTandC(Network,SEBC,0.00)
        #run on L1 subgraph
        sgst = SGST(EBC0,
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
        sgst.run_one_step_dynamic(t)
        TrackVol(t,OutNet,OutVol, prop = 'VSo')
        TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
        TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
        TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
        n += 1
        #L0 last
        #update t and graph
        t = times[n]
        SubNet,OutNet = UpdateGraph(Network,SubNet,n, edgeprops = edge_properties, nodeprops = node_properties)
        sgst = SGST(SubNet,
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
        sgst.run_one_step_dynamic(t)
        TrackVol(t,OutNet,OutVol, prop = 'VSo')
        TrackConc(t,OutNet,OutConc, VWprop = 'VW',VSprop = 'VSo')
        TrackGS(t,OutNet,OutGS, GSprop = 'd_dist_node')
        TrackDet(t,OutNet, OutDet, Dprop = 'detritus_node')
        n += 1
    #export this day's output
    MakeFig(Network,SubNet)
    fn = os.path.join(OutputDir,'model_end_day{}.png'.format(day))
    plt.savefig(fn)
    plt.close()

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

#and of cumulative volume flux by source
#plotting if we do not have bedrock detritus enabled
def MakeFig(OutDets,OutVols):
    fig,axs = plt.subplots(figsize = (6,4.5), dpi = 300)
    DetClasses = ['init','basal','basement']
    PlotOrder = [0,1,2]
    col = ['gray','lightyellow','lightsalmon']
    DetVolArray = np.zeros(shape = (len(DetClasses),len(OutDets[0])-1))
    Otimes = [n for n in OutVols[0][1:]]
    for i,j in enumerate(OutDets[2][1:]):
        VScale = OutVols[3][i]
        for m,n in enumerate(PlotOrder):
            key = DetClasses[m]
            DetVolArray[m][i] = DetVolArray[m][i-1]+j[key] * VScale
    axs.stackplot(Otimes,DetVolArray, labels = DetClasses, colors = col)
    axs.legend(fontsize = 'x-small', ncol = 3, loc = 'upper left')

#plotting function if we have bedrock detritus enabled
def MakeFigD(OutDets,OutVols):
    fig,axs = plt.subplots(figsize = (6,4.5), dpi = 300)
    DetClasses= ['init','basal',0.0,1.0,2.0,10.0,11.0,12.0,20.0,21.0,22.0,30.0,31.0,32.0,40.0,41.0,42.0]
    col = ['gray','lightyellow','lightsalmon','salmon','darksalmon','lightgreen','limegreen','lime','skyblue','deepskyblue','steelblue','mediumslateblue','slateblue','darkslateblue','thistle','violet','darkviolet']
    PlotOrder = [0,1,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
    DetVolArray = np.zeros(shape = (len(DetClasses),len(OutDets[0])-1))
    Otimes = [n for n in OutVols[0][1:]]
    for i,j in enumerate(OutDets[2][1:]):
        VScale = OutVols[3][i]#/OutVols[4][-1]
        for m,n in enumerate(PlotOrder):
                key = DetClasses[m]
                DetVolArray[m][i] = DetVolArray[m][i-1]+j[key] * VScale
    axs.stackplot(Otimes,DetVolArray, labels = DetClasses, colors = col)
    axs.legend(fontsize = 'x-small', ncol = 3, loc = 'upper left')

if DetMode == 'NodeProp':
    MakeFigD(OutDet,OutVol)
    fn = os.path.join(OutputDir,'OutDetVol.png')
    plt.savefig(fn,dpi = 300)
else:
    MakeFig(OutDet,OutVol)
    fn = os.path.join(OutputDir,'OutDetVol.png')
    plt.savefig(fn,dpi = 300)