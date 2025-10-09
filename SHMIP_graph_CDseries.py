# -*- coding: utf-8 -*-
"""
This is a sample script to make the input graphs for a GraphSSeT model for a member of the SHMIP 'C' or 'D' model ensembles. 

For the GraphSSeT model description see the paper of Aitken et al. (2024)

https://doi.org/10.5194/tc-18-4111-2024

codeauthor:: Alan Aitken

This version October 2025
"""
import numpy as np
import random
import networkx as nx
from NetworkX_funcs import *
from ReadGladsMat import LoadMesh, GladstoNetworkX_multi
import pickle
import os
import matplotlib.pyplot as plt

modelfiledir = "./InputData/"
model = "sqrt_moulins_diurnal1_mesh4.mat" #this is model C1 --to analyse C2 use diurnal2, C3 diurnal3 and so on
model_name = 'C1'

OutputDir = "./ModelOutputs/"
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)
    
# We wish to sample the Mesh at a frequency (and phase) appropriate for our sed model timesteps without aliasing. 
# In this case we have model results reported every hour and a model duration of 50 days.
# Forcing in this case is sinusoidal with maximum at the beginning and end of each day
min_timestep = 0
max_timestep = 240 #1201 total -- so choose how much time you want..here we take the first 10 days
timestep_ss = 6 # representative sub-sampling used to build the networks ... here every 6 hours

MeshPath = modelfiledir+model
#%% make the Mesh
Mesh = LoadMesh(MeshPath)

# Make a directed network from the Mesh
Network = GladstoNetworkX_multi(Mesh, min_step = min_timestep, max_step = max_timestep, step_size = timestep_ss, weight_by = 'area')

# Store initial graph
fn = 'InitialNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(Network,file)

#%% Unlike for steady state we wish to preserve the whole network as L0 subgraphs will be generated dynamically
#we set up the graph for the first timestep however

#remove ALWAYS invalid edges
edge_status_arr = Network.edges(data = 'edge_status_arr')
EdgeToRemove = [(i[0],i[1]) for i in edge_status_arr if np.all(i[2]<0)]
#careful here - you can't get these back easily!
Network.remove_edges_from(EdgeToRemove)

# for the FIRST TIMESTEP identify outlet nodes and their coords

# identify head nodes for the subgraph (nodes with <=1 predecessor nodes)
Head_nodes = [j for i,j in enumerate(Network.pred) if len(Network.pred[j])<1]
#endpoint nodes with no succs
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

print('Number of Head, Outlet, OOM and Sink nodes are {},{},{},{} respectively'.format(len(Head_nodes),len(Outlet_nodes),len(OOM_nodes), len(Sink_nodes)))

#get any moulin nodes with status == 4.
node_status = Network.nodes(data = 'node_status')
Moulins = [i[0] for i in node_status if i[1]==4]

# New random seed node set, 100 nodes here
NetNotOut_nodes = [j for i,j in enumerate(Network.succ) if len(Network.succ[j])>0]
NetNotOut_nodes_k = random.choices(NetNotOut_nodes, k = 100)

#collate all 'input' nodes for graph analysis - removing duplicates. priority is Moulins,Head then Random
Head_nodes = [j for i, j in enumerate(Head_nodes) if j not in Moulins]
NetNotOut_nodes_k = [j for i, j in enumerate(NetNotOut_nodes_k) if j not in Head_nodes]
In_nodes = Moulins+Head_nodes+NetNotOut_nodes_k

#Get coords for plotting
Head_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Head_nodes)]
Outlet_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Outlet_nodes)]
OOM_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(OOM_nodes)]
Sink_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Sink_nodes)]
Moulins_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Moulins)]
NetNotOut_nodes_k_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(NetNotOut_nodes_k)]

#Weighted path analyses for these in/out nodes (using the whole network) optionally with a maximum weight of x each way
SubNet = Dijkstra_SubNetSometoSome(Network,In_nodes[:],Endpoints[:], Weight = 'weight')
#%% make a plot of the chosen flow catchment and add head/in nodes and outlet nodes
NodeSets = {'network in nodes': NetNotOut_nodes_k_coords,
            'moulin nodes': Moulins_coords,
            'head nodes': Head_node_coords,
            'outlet nodes': Outlet_node_coords,
            'OOM nodes': OOM_node_coords,
            'Sink nodes': Sink_node_coords,
            }
fig = PlotSubNetworksEdgesandNodes(Network, SubNet,NodeSets, lw = 0.1, ps = 0.4, label = 'L0 subgraph')
fn = os.path.join(OutputDir,'GraphEdgesandNodes.png')
plt.savefig(fn)
plt.close
#%%rescale weight by a property array as (1-prop/max_of_prop)**n ... in this case we use channel area with n = 3
a = nx.get_edge_attributes(Network,'channel_area_arr')
a_arr = np.transpose(np.array([(a[key]) for key in a])) #recast as 2D array with time-wise order
w_arr = [(1.0-j/np.nanmax(j))**3 for i,j in enumerate(a_arr)] #for each timeslice reweight all edges
ws = np.transpose(w_arr) #return to edge-wise order
new_weight = {key: ws[i][0] for i,key in enumerate(a)}
nx.set_edge_attributes(Network, new_weight, 'weight2') #record new weight to graph
new_weights = {key: ws[i] for i,key in enumerate(a)}
nx.set_edge_attributes(Network, new_weights, 'weights_arr') #record weights array to graph
#%% compute edge betweenness centrality to map catchment structure
SEBC = SomeEdgeBetweennessCentrality(SubNet, In_nodes, Endpoints, weight = 'weight2', norm = True)
#%% make plots of key edge properties for the graph
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(6,1, figsize = (8,12.75), dpi = 300)
    #edge status
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = 'status', prop = 'status', lw = 0.4, cmap = 'inferno_r',fig = fig, ax = axs[0])
    #hydraulic potential gradient
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = '$∇φ$ (Pa/m)', prop = 'hyd_pot_grad', lw = 0.4, cmap = 'inferno_r',fig = fig, ax = axs[1])
    #channel area
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = '$S$ (${m^2}$)', prop = 'channel_area', lw = 0.4, cmap = 'inferno_r',fig = fig, ax = axs[2])
    #channel flux
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = '$Q_w$ (${m^3}/s$)', prop = 'channel_flux', lw = 0.4, cmap = 'inferno_r',fig = fig, ax = axs[3])
    #edge weight
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = 'edge weight', prop = 'weight2', lw = 0.4, cmap = 'inferno_r', minprop = 0, maxprop = 1,fig = fig, ax = axs[4])
    #edge betweeness centrality
    PlotNetworkEdgeProp(Network, SubNetworks = SubNet, label = 'EBC', prop = 'betweenness_sub', lw = 0.4, cmap = 'inferno_r',fig = fig, ax = axs[5])
    fig.supxlabel('x coordinate (m)', y = 0.085)
    fig.supylabel('y coordinate (m)')
MakeFig(Network,SubNet)
fn = os.path.join(OutputDir,'EdgeProperties.png')
plt.savefig(fn)
plt.show()
plt.close()
#%% make plots of key node properties for the graph
def MakeFig(Network,SubNet):
    fig,axs = plt.subplots(6,1, figsize = (8,12.75), dpi = 300)
    #status
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = 'node status', prop = 'node_status', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[0])
    #bed_elevation
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = '$z_b$ (m)', prop = 'bed_elevation', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[1])
    #hydraulic potential
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = '$φ$ (Pa)', prop = 'hydraulic_potential', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[2])
    #effective pressure
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = 'N (Pa)', prop = 'effective_pressure', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[3])
    #sheet flow thickness
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = '$h_w$ (m)', prop = 'h_sheet', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[4])
    #basal velocity magnitude
    PlotNetworkNodeProp(Network, SubNetworks = SubNet, label = '$u_b$ (m/s)',prop = 'basal_velocity_magnitude', lw = 0.15, ps = 0.4, cmap = 'inferno_r', fig = fig, ax = axs[5])
MakeFig(Network,SubNet)
fn = os.path.join(OutputDir,'NodeProperties.png')
plt.savefig(fn)
plt.show()
plt.close()
#%% Make copies of Subgraphs and pickle. Copies are needed as the subgraphs are views

#whole graph
fn = 'FinalNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(Network,file)

#Level 0 subgraph
SubNetCopy = SubNet.copy()
fn = 'SubNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(SubNetCopy,file)