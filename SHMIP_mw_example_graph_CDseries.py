# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:50:04 2023

@author: 00075859
"""
import numpy as np
import random
import networkx as nx
from NetworkX_funcs import *
from ReadGladsMat import *
import pickle

modelfiledir = "./InputData/"
model = "sqrt_moulins_diurnal1_mesh4.mat" #this is model C1 --to analyse C2 use diurnal2, C3 diurnal3 and so on
min_timestep = 0
max_timestep = -1 #1201 total

# We wish to sample the Mesh at the frequency (and phase) needed for our sed model timesteps without aliasing. 
# In this case we have mesh timesteps every hour (3600 seconds) and a model duration of 50 days. 
# Forcing is sinusoidal with maximum at the beginning and end of each day; for the purposes of the algorithm we wish to do every third timestep  
  
timestep_ss = 3 # representative sub-sampling used to build the network.

MeshPath = modelfiledir+model
#%% make the Mesh
Mesh = LoadMesh(MeshPath)
#%% Make a directed network from the Mesh
Network = GladstoNetworkX_multi(Mesh, min_step = min_timestep, max_step = max_timestep, step_size = timestep_ss, weight_by = 'area')
#%% backup initial graph
with open('InitialNetwork_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(Network,file)
#%%# read previously saved graph from backup
#with open('InitialNetwork_SHMIP_C1_S3.pickle','rb') as file:
#    Network = pickle.load(file)
#%% identify label propagation communities - these can resemble catchments
LPcomms = AsynLPA(Network, min_nodes = 20, weight = 'weight') #output is a dict
LPcomms = list(LPcomms.values())
#%% Make n subnets for the community set and plot
SubNets ={}
for i,j in enumerate(LPcomms):
    SubNets[i]= nx.subgraph(Network,LPcomms[i]) #i for list, j for set
PlotSubNetworksEdges(Network, SubNets, lw = 0.5, legend = False)
#%% Now we may work with a chosen set of communities to define a 'catchment' subgraph - we define these as a set or list of nodes. Here there is just one catchment
Catchment=LPcomms[0].union(
#           LPcomms[1],
#           LPcomms[2],
#           LPcomms[9999],
           )
S = nx.subgraph(Network, Catchment)

#and from this we want the largest bi-connected component (i.e. no disconnected regions)
SUD = S.to_undirected()
SubNet = nx.subgraph(Network, max(nx.biconnected_components(SUD),key = len))
PlotSubNetworksEdges(Network, SubNet, lw = 0.5)
#%% for the catchment identify outlet nodes and their coords

# identify head nodes for the subgraph (nodes with <=1 predecessor nodes)
Head_nodes = [j for i,j in enumerate(SubNet.pred) if len(SubNet.pred[j])<=1]
Head_node_coords = [e["coords"] for u,e in Network.nodes(data=True) if u in(Head_nodes)]

# identify ouput nodes for the subgraph (nodes with <=1 successor nodes)
Out_nodes = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])<=1]

# trim further the outlet nodes by coordinates to focus only on the outlet edge
Xmax = 50.0 #meters from the model edge

ToRemove = []
for i in Out_nodes:
    if SubNet.nodes[i]['coords'][0] >= Xmax:
        ToRemove.append(i)

Out_nodes = [u for u in Out_nodes if u not in ToRemove]
Out_node_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Out_nodes)]

#get any moulin nodes with status == 4
node_status_arr = Network.nodes(data = 'node_status_arr')
Moulins = [i[0] for i in node_status_arr if np.any(i[1]==4)]
Moulins_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(Moulins)]

# New random seed node set, 100 nodes here
NetNotOut_nodes = [j for i,j in enumerate(SubNet.succ) if len(SubNet.succ[j])>0]
NetNotOut_nodes_k = random.choices(NetNotOut_nodes, k = 100)
NetNotOut_nodes_k_coords = [e["coords"] for u,e in SubNet.nodes(data=True) if u in(NetNotOut_nodes_k)]

#collate all 'input' nodes for graph analysis - removing duplicates. Priority is Moulins,Head then Random
Head_nodes = [j for i, j in enumerate(Head_nodes) if j not in Moulins]
NetNotOut_nodes_k = [j for i, j in enumerate(NetNotOut_nodes_k) if j not in Head_nodes]
In_nodes = Moulins+Head_nodes+NetNotOut_nodes_k

#remove ALWAYS 'floating' edges
edge_status_arr = Network.edges(data = 'status')
EdgeToRemove = [(i[0],i[1]) for i in edge_status_arr if np.all(i[2]<0)]
#careful here - you cant get these back!
Network.remove_edges_from(EdgeToRemove)

#Weighted path analyses for these in/out nodes (using the whole network) optionally with a maximum weight of x each way
SubNet = Dijkstra_SubNetSometoSome(Network,In_nodes[:],Out_nodes[:], Weight = 'weight')
#%% make a plot of the chosen flow catchment and add head/in nodes and outlet nodes
NodeSets = {'network in nodes': NetNotOut_nodes_k_coords,
            'moulin nodes': Moulins_coords,
            'head nodes': Head_node_coords,
            'outlet nodes': Out_node_coords, 
            }
PlotSubNetworksEdgesandNodes(Network, SubNet,NodeSets, lw = 0.1, ps = 0.2)
#%%rescale weight by a property as (1-prop/max_of_prop)**n...in this case we use channel area with n = 3
a = nx.get_edge_attributes(Network,'channel_area')
a_arr = [np.nanmin(a[key]) for key in a]
w = (1.0-a_arr/np.nanmax(a_arr))**3
new_weight = {key: w[i] for i,key in enumerate(a)}
nx.set_edge_attributes(Network, new_weight, 'weight2')

#%% If desired compute shortest paths or n-shortest paths on the SubNetwork as subgraphs
WS2SS = Dijkstra_SubNetSometoSomeShortest(SubNet,In_nodes[:],Out_nodes[:],Weight = 'weight2')
WS2SSn = Dijkstra_SubNetSometoSomeNShortest(SubNet,In_nodes[:],Out_nodes[:],10,Weight = 'weight2')
#%% compute edge betweenness centrality
SEBC = SomeEdgeBetweennessCentrality(SubNet, In_nodes, Out_nodes, weight = 'weight2', Norm = True)
#%% make plots of key edge properties for the graph
#status
PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'status', lw = 0.75, cmap = 'inferno_r')
#betweenness centrality
PlotNetworkEdgeProp(Network, SubNetworks = Network, prop = 'betweenness_sub', lw = 0.5, cmap = 'inferno_r')
#hydraulic potentisal gradient
PlotNetworkEdgeProp(Network, SubNetworks = Network, prop = 'hyd_pot_grad', lw = 0.5, cmap = 'inferno_r')
#channel flux
PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'channel_flux', lw = 0.75, cmap = 'inferno_r')
#channel area
PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'channel_area', lw = 0.75, cmap = 'inferno_r')
#weight
PlotNetworkEdgeProp(Network, SubNetworks = SubNet, prop = 'weight2', lw = 0.75, cmap = 'inferno_r')
#%% make plots of key node properties for the graph
#status
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'node_status', lw = 0.15, ps = 0.4, cmap = 'inferno_r')
#bed_elevation
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'bed_elevation', lw = 0.15, ps = 0.4, cmap = 'inferno_r')
#hydraulic potential
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'hydraulic_potential', lw = 0.15, ps = 0.4, cmap = 'inferno_r')
#effective pressure
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'effective_pressure', lw = 0.15, ps = 0.4, cmap = 'inferno_r')
#sheet flow thickness
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'h_sheet', lw = 0.15, ps = 0.4, cmap = 'inferno_r')
#basal velocity magnitude
PlotNetworkNodeProp(Network, SubNetworks = SubNet, prop = 'basal_velocity_magnitude', lw = 0.15, ps = 0.4, cmap = 'inferno_r')

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
            'outlet nodes': Out_node_coords, 
            }
#Shortest Paths
SubNets={
         'Shortest paths':WS2SS,
         'N Shortest paths':WS2SSn,
         }

PlotSubNetworksEdgesandNodes(Network, SubNets,NodeSets, lw = 0.2, ps = 0.4)

#centrality level 1 and 2
SubNets={
         'Centrality > 0.0':EBC0,
         'Centrality > 0.005':EBCMinCent
         }

PlotSubNetworksEdgesandNodes(Network, SubNets,NodeSets, lw = 0.2, ps = 0.4)
#%% Make copies of Subgraphs and pickle. Copies are needed as the subgraphs are views

#whole graph
with open('FinalNetwork_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(Network,file)

#Level 0 subgraph (catchment scale)
SubNetCopy = SubNet.copy()
with open('SubNetwork_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(SubNetCopy,file)
    
#Shortest Paths
WS2SSCopy = WS2SS.copy()
with open('ShortestNetwork_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(WS2SSCopy,file)

#N Shortest Paths
WS2SSnCopy = WS2SSn.copy()
with open('NShortestNetwork_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(WS2SSnCopy,file)

#Level 1 centrality subgraph
EBC0Copy = EBC0.copy()
with open('BC0.0Network_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(EBC0Copy,file)

#Level 2 centrality subgraph
EBCMinCentCopy = EBCMinCent.copy()
with open('BC0.005Network_SHMIP_C1_S3.pickle','wb') as file:
    pickle.dump(EBCMinCentCopy,file)