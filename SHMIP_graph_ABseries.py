# -*- coding: utf-8 -*-
"""
This is a sample script to make the input graphs for a GraphSSeT model for a member of the SHMIP 'A' or 'B' model ensembles. 

For the GraphSSeT model description see the paper of Aitken et al. (2024)

https://doi.org/10.5194/tc-18-4111-2024

codeauthor:: Alan Aitken

This version October 2025

"""
import numpy as np
import random
import networkx as nx
from NetworkX_funcs import *
from ReadGladsMat import LoadMesh, GladstoNetworkX_one
import pickle
import os
import matplotlib.pyplot as plt

ModelFileDir = "./InputData/"
OutputDir = "./ModelOutputs/"
if not os.path.exists(OutputDir):
    os.makedirs(OutputDir)
#A series
model = "sqrt_ch5_mesh4.mat" #this is model A5 --to analyse A4 use ch4, A6 ch6 and so on
model_name = 'A5'
# B series
#model = "sqrt_moulins5_mesh4.mat" #this is model B5 --to analyse B1 use moulins1, B2 moulins2 and so on
#model_name = 'B5'

MeshPath = ModelFileDir+model

#%% make the Mesh and build the graph
Mesh = LoadMesh(MeshPath)

# make a directed graph from the last timestep
Network = GladstoNetworkX_one(Mesh, step = -1, weight_by = 'area')

# Store Initial Graph
fn = 'InitialNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(Network,file)
    
#We want just the largest bi-connected component
SUD = Network.to_undirected()
SubNet = nx.subgraph(Network, max(nx.biconnected_components(SUD),key = len))
#%% Make the L0 subgraph

#remove downstream boundary edges
edge_status = Network.edges(data = 'status')
ToRemove = [(i[0],i[1]) for i in edge_status if i[2]<0]
#careful here - you cant get these back easily!
Network.remove_edges_from(ToRemove)

#for this graph identify head and outlet nodes and their coords

# identify head nodes (nodes with <1 predecessor nodes)
Head_nodes = [j for i,j in enumerate(SubNet.pred) if len(SubNet.pred[j])<1]
# identify end-point nodes (nodes with <1 successor nodes)
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

#get any moulin nodes with status == 4
node_status = Network.nodes(data = 'node_status')
Moulins = [i[0] for i in node_status if i[1]==4]

# New random seed node set, 100 nodes here
NetNotOut_nodes = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])>0]
NetNotOut_nodes_k = random.choices(NetNotOut_nodes, k = 100)

#collate all input nodes - removing duplicates. Priority is Moulins,Head then Random
Head_nodes = [j for i, j in enumerate(Head_nodes) if j not in Moulins]
NetNotOut_nodes_k = [j for i, j in enumerate(NetNotOut_nodes_k) if j not in Head_nodes]
In_nodes = Moulins+Head_nodes+NetNotOut_nodes_k

#Get coords for plotting
Head_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Head_nodes)]
Outlet_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Outlet_nodes)]
OOM_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(OOM_nodes)]
Sink_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Sink_nodes)]
Moulins_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Moulins)]
NetNotOut_nodes_k_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(NetNotOut_nodes_k)]

#Weighted path analyses for these in/out nodes (using the whole network) optionally with a maximum weight of x each way
SubNet = Dijkstra_SubNetSometoSome(Network,In_nodes[:],Endpoints[:], Weight = 'weight')
#%% make a plot of the chosen flow catchment and add head/in nodes and outlet nodes
NodeSets = {'random input nodes': NetNotOut_nodes_k_coords,
            'head nodes': Head_node_coords,
            'moulin nodes': Moulins_coords,
            'outlet nodes': Outlet_node_coords,
            'OOM nodes': OOM_node_coords,
            'Sink nodes': Sink_node_coords,
            }
fig = PlotSubNetworksEdgesandNodes(Network, SubNet,NodeSets, lw = 0.1, ps = 0.4, label = 'L0 subgraph')
fn = os.path.join(OutputDir,'GraphEdgesandNodes.png')
plt.savefig(fn)
plt.close
#%%rescale weight by a property as (1-prop/max_of_prop)**n...in this case we use channel area with n = 3
a = nx.get_edge_attributes(Network,'channel_area')
a_arr = [a[key] for key in a]
w = (1.0-a_arr/np.nanmax(a_arr))**3
new_weight = {key: w[i] for i,key in enumerate(a)}
nx.set_edge_attributes(Network, new_weight, 'weight2')

#%% If desired compute shortest paths or n-shortest paths on the SubNetwork as subgraphs
WS2SS = Dijkstra_SubNetSometoSomeShortest(SubNet,In_nodes[:],Out_nodes[:],Weight = 'weight2')
WS2SSn = Dijkstra_SubNetSometoSomeNShortest(SubNet,In_nodes[:],Out_nodes[:],10,Weight = 'weight2')
#%% compute edge betweenness centrality
SEBC = SomeEdgeBetweennessCentrality(SubNet, In_nodes, Out_nodes, weight = 'weight2', norm = True)
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
#%% Make level 1 and 2 centrality-based subgraphs...but of course this could be another property
def EdgeValTandC(graph, val_arr, thresh):
    EdgeSel = {j:val_arr[j] for i,j in enumerate(val_arr) if val_arr[j]>thresh}
    G = graph.edge_subgraph(EdgeSel)
    return G

EBC0 = EdgeValTandC(Network,SEBC,0.00)
EBCMinCent = EdgeValTandC(Network,SEBC,0.005)
#%% make a plot of these SubGraphs over the main Graph
NodeSets = {'network in nodes': NetNotOut_nodes_k_coords,
            'moulin nodes': Moulins_coords,
            'head nodes': Head_node_coords,
            'outlet nodes': Outlet_node_coords, 
            }
#Shortest Paths
SubNets={
         'L1 Shortest':WS2SS,
         'L2 N Shortest':WS2SSn,
         }

fig = PlotSubNetworksEdgesandNodes(Network, SubNets,NodeSets, lw = 0.2, ps = 0.4)
fn = os.path.join(OutputDir,'DjikstraSubGraphs.png')
plt.savefig(fn)
plt.close

#centrality level 1 and 2
SubNets={
         'L1 EBC > 0.0':EBC0,
         'L2 EBC > 0.005':EBCMinCent
         }

fig  = PlotSubNetworksEdgesandNodes(Network, SubNets,NodeSets, lw = 0.2, ps = 0.4)
fn = os.path.join(OutputDir,'EBCSubGraphs.png')
plt.savefig(fn)
plt.close
#%% Make copies of Subgraphs and pickle. Copies are needed for pickling as the subgraphs are views

#whole graph
fn = 'FinalNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(Network,file)

#Level 0 subgraph (catchment scale)
SubNetCopy = SubNet.copy()
fn = 'SubNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(SubNetCopy,file)
    
#Shortest Paths
WS2SSCopy = WS2SS.copy()
fn = 'ShortestNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(WS2SSCopy,file)

#N Shortest Paths
WS2SSnCopy = WS2SSn.copy()
fn = 'NShortestNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(WS2SSnCopy,file)

#Level 1 centrality subgraph
EBC0Copy = EBC0.copy()
fn = 'EBC0Network_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(EBC0Copy,file)

#Level 2 centrality subgraph
EBCMinCentCopy = EBCMinCent.copy()
fn = 'EBCMinNetwork_SHMIP_'+model_name+'.pickle'
with open(fn,'wb') as file:
    pickle.dump(EBCMinCentCopy,file)