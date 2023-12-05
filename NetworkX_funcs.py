# -*- coding: utf-8 -*-
"""
Created on Tue Dec 5 2023

This file contains functions to conduct operations on NetworkX graphs to support the GraphSSet model.

It includes basic graph manipulation functions as well as anaytical functions to perform graph analysis and graph plotting functions

Plotting functions may need modification for graphs other than those supplied in the examples 

@author: Alan Aitkn alan.aitken@uwa.edu.au
"""
#import requird  python modules
import numpy as np
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from operator import itemgetter
import itertools

#%%Path analysis functions to make SubGraphs

#Simple paths
#Make a subgraph containing all simple paths to any node
def SubNetAlltoOne(Network,OutNode,MaxDepth):
    #make a reversed version for the predecessor algorithm
    krowteN = Network.reverse()
    #identify subnet and rank
    subnet_n = nx.predecessor(krowteN,OutNode, cutoff = MaxDepth)
    SubkrowteN = nx.subgraph(krowteN,subnet_n.keys())
    SubNetwork = SubkrowteN.reverse()
    return(SubNetwork)

#Make a subgraph containing all simple paths from any node
def SubNetOnetoAll(Network,InNode,MaxDepth):
    subnet_n = nx.predecessor(Network,InNode, cutoff = MaxDepth)
    SubNetwork = nx.subgraph(Network,subnet_n.keys())
    return(SubNetwork)

#Also with weightings using the Dijkstra algorithm

#As SubNetAlltoOne but weighted
def Dijkstra_SubNetAlltoOne(Network,OutNode,MaxWeight,Weight):
    #make a reversed version of the network
    krowteN = Network.reverse()
    #identify subnet and rank
    subnet_n = nx.dijkstra_predecessor_and_distance(krowteN,OutNode, cutoff = MaxWeight, weight = Weight)
    SubkrowteN = nx.subgraph(krowteN,subnet_n[0].keys())
    SubNetwork = SubkrowteN.reverse()
    return(SubNetwork)

#s SubNetOnetoAll but weighted
def Dijkstra_SubNetOnetoAll(Network,InNode,MaxWeight,Weight):
    subnet_n = nx.dijkstra_predecessor_and_distance(Network,InNode, cutoff = MaxWeight, weight = Weight)
    SubNetwork = nx.subgraph(Network,subnet_n[0].keys())
    return(SubNetwork)

#Now for all nodes to some selection of nodes - may be weighted or not
def Dijkstra_SubNetAlltoSome(Network,OutNodes,MaxWeight = None ,Weight = None):
    #make a reversed version of the network
    krowteN = Network.reverse()
    subnet_n = nx.multi_source_dijkstra_path(krowteN,OutNodes, cutoff = MaxWeight, weight = Weight)
    SubkrowteN = nx.subgraph(krowteN,subnet_n)
    SubNetwork= SubkrowteN.reverse()
    return(SubNetwork)

# and for some selection of nodes to all nodes - may be weighted or not
def Dijkstra_SubNetSometoAll(Network,InNodes,MaxWeight = None ,Weight = None):
    subnet_n = nx.multi_source_dijkstra_path(Network,InNodes, cutoff = MaxWeight, weight = Weight)
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#Get the shortst path one to one mode - unweighted or weighted
def SubNetOnetoOneShortest(Network,InNode,OutNode, Weight = None):
    subnet_n = nx.shortest_path(Network,InNode, target = OutNode, weight = Weight)
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#Get all shortest paths - weighted or unweighted
def SubNetOnetoOneNShortest(Network,InNode,OutNode,n,Weight = None):
    paths = nx.all_shortest_paths(Network,InNode, target = OutNode, weight = Weight)
    subnet_n= []
    i = 0
    while i<n:
        try:
            subnet_n.extend(next(paths))
        except StopIteration:    
            print('no more paths at i = {}'.format(i))
            i=n
        else:
            i+=1
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#Paths for some selection to some other selection of nodes - may be weighted or not
#It has not been tested exactly what happens if nodes are shared in In and Out sets
def Dijkstra_SubNetSometoSome(Network,InNodes,OutNodes,MaxWeight = None ,Weight = None):
    krowteN = Network.reverse()
    subnet_o = nx.multi_source_dijkstra_path(krowteN,OutNodes, cutoff = MaxWeight, weight = Weight)
    subnet_i = nx.multi_source_dijkstra_path(Network,InNodes, cutoff = MaxWeight, weight = Weight)
    subnet_n = {}
    for key in subnet_o:
        if key in subnet_i:
            path = subnet_i[key]+subnet_o[key]
            subnet_n[key] = path
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#some to some shortest paths - may be weighted or not
def Dijkstra_SubNetSometoSomeShortest(Network,InNodes,OutNodes,Weight = None):
    subnet_n = []
    for i,k in enumerate(InNodes):
        for m,n in enumerate(OutNodes):
            if nx.has_path(Network,source = k,target = n):
                subnet_n += nx.shortest_path(Network, source = k,target = n,weight = Weight)
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#some to some all shortest paths - may be weighted or not
def Dijkstra_SubNetSometoSomeNShortest(Network,InNodes,OutNodes, n, Weight = None):
    subnet_n = []
    for j,k in enumerate(InNodes):
        for p,q in enumerate(OutNodes):
            if nx.has_path(Network,source = k,target = q):
                paths = nx.all_shortest_paths(Network, source = k,target = q,weight = Weight)
                i = 0
                while i<n:
                    try:
                        subnet_n.extend(next(paths))
                    except StopIteration:    
                        print('node {} to {}: no more paths at i = {}'.format(k,q,i))
                        i=n
                    else:
                        i+=1
    SubNetwork = nx.subgraph(Network,subnet_n)
    return(SubNetwork)

#%% Identification of communtiis - mainly to identify 'catchment' structurs but can also define other things

#greedy modularity communities
def GreedyModularityN(Network, resolution = 1, weight = None, cutoff = 1, best_n = 1):
    c = community.greedy_modularity_communities(Network,resolution = resolution, weight = weight, cutoff = cutoff, best_n = best_n)
    d = [set(j) for i,j in enumerate(c)]
    return d

#Girvan-Newman communities (for all ranks)
def GirvanNewman(Network, Prop = None, MVE = None):
    #returns the first rank of communities
    if not MVE:
        if Prop:
            def MVE(G):
                u, v, w = max(G.edges(data=Prop), key=itemgetter(2))
                return (u, v)
        else:
            MVE = None
    comp = community.girvan_newman(Network,most_valuable_edge = MVE)
    t = tuple(sorted(c) for c in next(comp))
    return t

#Girvan-Newman communities (for a specific rank)    
def GirvanNewmanK(Network, k = 2, Prop = None, MVE = None):
    #returns only the k-th rank but tracks modularity
    if not MVE:
        if Prop:
            def MVE(G):
                u, v, w = max(Network.edges(data=Prop), key=itemgetter(2))
                return (u, v)
        else:
            MVE = None
    comp = community.girvan_newman(Network, most_valuable_edge = MVE)
    mod = []
    for communities in itertools.islice(comp, k):
        t = tuple(sorted(c) for c in communities)
        m = community.modularity(Network,t)
        mod.append(m)
    return t,mod

#Asynchronous label propagation communities
def AsynLPA(Network, n_max = 1000, min_nodes = 1, weight = None):
    c = community.asyn_lpa_communities(Network, weight = weight)
    n = 0
    cc_accumulator = []
    Communities = {}
    while n< n_max:
        try:
            cc = next(c)
            if len(cc)>=min_nodes:
                Communities[n]= cc
            else:
                cc_accumulator+=cc
        except StopIteration:    
            print('no more constructor items at n = {}'.format(n))
            n=n_max
        else:
            n+=1
    Communities[9999]= cc_accumulator
    return Communities

#K-clique communities
def KcliqueComms(Network, k = 3, cliques = None, n_max = 1000, min_nodes = 1):
    c = community.k_clique_communities(Network, k)
    n = 0
    cc_accumulator = []
    Communities = {}
    while n< n_max:
        try:
            cc = next(c)
            if len(cc)>=min_nodes:
                Communities[n]= cc
            else:
                cc_accumulator+=cc
        except StopIteration:    
            print('no more constructor items at n = {}'.format(n))
            n=n_max
        else:
            n+=1
    Communities[9999]= cc_accumulator
    return Communities

#Louvain communities
def Louvain(Network, min_nodes = 1, resolution = 1, weight = None):
    c = community.louvain_communities(Network, weight = weight, resolution = resolution, threshold = 1e-12)
    return c

#Asynchronous fluid communities
def AsynFluid(Network, k):
    c = community.asyn_fluidc(Network, k)
    Communities = {}
    for i in range(0,k):
        try:
            cc = next(c)
            Communities[i]= cc
        except StopIteration:    
            print('no more constructor items at n = {}'.format(i))
    return Communities

#%% Centrality measures - these identify key edges or nodes for flow

# Closeness Centrality (for all nodes)
def ClosenessCentrality(Network, weight = None, threshold = None):
    c = nx.closeness_centrality(Network, distance = weight)
    nx.set_node_attributes(Network, c, "closeness")
    return c

#Betweenness centrality for a set of nodes (source and target)
def SomeNodeBetweennessCentrality(Network, sources, targets, weight = None, threshold = None):
    c = nx.betweenness_centrality_subset(Network, sources, targets, weight = weight)
    return c

#Edge betweenness centrality for all nodes
def AllEdgeBetweennessCentrality(Network, k = None, weight = None, threshold = None):
    c = nx.edge_betweenness_centrality(Network, k = k, weight = weight)
    nx.set_edge_attributes(Network, c, "betweenness")
    if threshold:
        subnet_e = [[u,v] for u,v,e in Network.edges(data=True) if e['betweenness'] >= threshold]
        return subnet_e

#Edge betweenness centrality for a set of nodes (source and target)
def SomeEdgeBetweennessCentrality(Network, sources, targets, weight = None, threshold = None, Norm = True):
    c = nx.edge_betweenness_centrality_subset(Network, sources, targets, weight = weight)
    if Norm:
        NF = len(sources)*(len(targets)-1)
        c = {j:c[j]/NF for i,j in enumerate(c)}
    nx.set_edge_attributes(Network, c, "betweenness_sub")
    return c

#%%randomise edge and node attributes

#Generate a random edge attribute
def Randomise_Edge_Attribute(Network, prop = 'weight', var_prop = None, allow_zero = True, allow_neg = False):
    rng = np.random.default_rng()
    data = nx.get_edge_attributes(Network,prop)
    d = np.array([data[key] for key in data])
    if var_prop is None:
        #variability between -0.5 and +0.5
        v = rng.random(size = np.shape(d))-0.5
    else:
        var_d = nx.get_edge_attributes(Network,var_prop)
        var_a = np.array([var_d[key] for key in var_d])
        #the formulation here is that variability is +/- the property given
        v = rng.random(size = np.shape(var_a))*var_a-var_a
    new_d = d+v
    if not allow_neg:
        new_d = np.where(new_d < 0, 0.0, new_d)
    if not allow_zero:
        new_d = np.where(new_d == 0, np.nan, new_d)
    new_data = {key: new_d[i] for i,key in enumerate(data)}
    nx.set_edge_attributes(Network, new_data, prop)

#Generate a random node attribute
def Randomise_Node_Attribute(Network, prop = 'weight', var_prop = None, allow_zero = True, allow_neg = False):
    rng = np.random.default_rng()
    data = nx.get_node_attributes(Network,prop)
    d = np.array([data[key] for key in data])
    if var_prop is None:
        #variability between -0.5 and +0.5
        v = rng.random(size = np.shape(d))-0.5
    else:
        var_d = nx.get_node_attributes(Network,var_prop)
        var_a = np.array([var_d[key] for key in var_d])
        #the formulation here is that variability is +/- the property given
        v = rng.random(size = np.shape(var_a))*var_a-var_a
    new_d = d+v
    if not allow_neg:
        new_d = np.where(new_d < 0, 0.0, new_d)
    if not allow_zero:
        new_d = np.where(new_d == 0, np.nan, new_d)
    new_data = {key: new_d[i] for i,key in enumerate(data)}
    #nx.set_node_attributes(Network, new_data, prop)
    return new_data

#Generate a node attribute defined on a spatial graticule
def Graticule_Attribute(Network, coords = 'coords', num_x = 5, num_y = 5, prop = None, null = '-99'):
    locs = nx.get_node_attributes(Network, coords)
    x_coords = np.array([locs[key][0] for key in locs])
    x_class = np.array([null]*len(x_coords))
    min_x = np.nanmin(x_coords)
    max_x =np.nanmax(x_coords)
    dx = (max_x - min_x)/num_x
    y_coords = np.array([locs[key][1] for key in locs])
    y_class = np.array([null]*len(y_coords))
    min_y = np.nanmin(y_coords)
    max_y =np.nanmax(y_coords)
    dy = (max_y - min_y)/num_y
    
    for i in range(0,num_y):
        for m,n in enumerate(y_coords):
            if n>=dy*i and n<dy*(i+1):
                y_class[m] = str(i)
            elif i == num_y-1 and n == max_y:
                y_class[m] = str(i)
                
    for i in range(0,num_x):
        for m,n in enumerate(x_coords):
            if n>=dx*i and n<dx*(i+1):
                x_class[m] = str(i)
            elif i == num_x-1 and n == max_x:
                x_class[m] = str(i)
                
    GratClass =  [x_class[i] + y_class[i] for i,j in enumerate(x_class)]
    new_data = {key: GratClass[i] for i,key in enumerate(locs)}
    return new_data

#%% Data access and manipulation

#make a X,Y,prop dict from Edges. In general we now do this explicitly but this is sometimes used
def NetworkEdgestoArr(Network, prop = 'weight'):
    edgelist = Network.edges.keys()
    lines = {}
    for m,n in enumerate(edgelist):
        lines[m] = {}
        lines[m]['coords'] = Network.get_edge_data(*n)['coords']
        lines[m][prop] = Network.get_edge_data(*n)[prop]
    return lines 

#make a X,Y,prop dict from Nodes. In general we now do this explicitly but this is sometimes used
def NetworkNodestoArr(Network, prop = 'weight'):
    nodelist = Network.nodes.keys()
    nodes = {}
    for m,n in enumerate(nodelist):
        nodes[m] = {}
        nodes[m]['coords'] = Network.nodes[n]["coords"]
        nodes[m][prop] = Network.nodes[n][prop]
    return nodes

#%% Plotting Functions

#plot the edges of a subgraph over the edges of the main graph
def PlotSubNetworksEdges(Network, SubNetworks, lw = None, ordered = True, legend = True):
    fig,ax = plt.subplots()
    lines = NetworkEdgestoArr(Network)
    #reorder for drawing by property
    if ordered:
        lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
    else:
        lines = list(lines.items())
    lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = 'gray', zorder=1, label = 'main network')
    ax.set_aspect('equal')
    ax.add_collection(lc)
    n=0
    if type(SubNetworks) is dict:
        lw += 0.25
        for i,j in SubNetworks.items():
            col = 'C{}'.format(n)
            lines = NetworkEdgestoArr(SubNetworks[i])
            #reorder for drawing by property
            if ordered:
                lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
            else:
                lines = list(lines.items())
            lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = col, zorder=2, label = 'sub-network {}'.format(i))

            ax.add_collection(lc)
            n +=1
    else:        #assume it is a graph
        col = 'C0'
        lines = NetworkEdgestoArr(SubNetworks)
        #reorder for drawing by property
        if ordered:
            lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
        else:
            lines = list(lines.items())
        lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = col, zorder=2, label = 'sub-network')
        ax.add_collection(lc)
    ax.autoscale()
    if legend:
        ax.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol = len(SubNetworks)+1)

#Plot a graph and selected node sets.
def PlotNetworkandNodes(Network,NodeLocSets, lw = None, ps = None, ordered = True):
    fig,ax = plt.subplots()
    lines = NetworkEdgestoArr(Network)
    #reorder for drawing by property
    if ordered:
        lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
    else:
        lines = list(lines.items())
    lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = 'gray', zorder=1,label = 'main network')
    ax.set_aspect('equal')
    ax.add_collection(lc)
    n = 0
    if type(NodeLocSets) is dict:
        for k,l in NodeLocSets.items():
            col = 'C{}'.format(n)
            ax.scatter([item[0] for item in l],[item[1] for item in l], s = ps,color = col, zorder=3, label = '{}'.format(k))
            n+=1
    ax.autoscale()
    ax.legend(loc='upper left')

#Plot a graph and sub-graph and selected node sets.
def PlotSubNetworksEdgesandNodes(Network, SubNetworks,NodeLocSets, lw = None, ps = None, ordered = True):
    fig,ax = plt.subplots(layout = 'constrained')
    lines = NetworkEdgestoArr(Network)
    #reorder for drawing by property
    if ordered:
        lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
    lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = 'gray', zorder=1,label = 'main network')
    ax.set_aspect('equal')
    ax.add_collection(lc)
    if type(SubNetworks) is dict:
        n = 0
        for i,j in SubNetworks.items():
            lw += lw
            col = 'C{}'.format(n)
            lines = NetworkEdgestoArr(SubNetworks[i])
            #reorder for drawing by property
            if ordered:
                lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
            lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = col, zorder=2, label = 'sub-network {}'.format(i))
            ax.add_collection(lc)
            n +=1
    else:        #assume it is a graph
        n = 0
        col = 'C0'
        lines = NetworkEdgestoArr(SubNetworks)
        #reorder for drawing by property
        if ordered:
            lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
        lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = col, zorder=2, label = 'sub-network')
        ax.add_collection(lc)
    if type(NodeLocSets) is dict:
        for k,l in NodeLocSets.items():
            n+=1
            col = 'C{}'.format(n)
            ax.scatter([item[0] for item in l],[item[1] for item in l], s = ps,color = col, zorder=3, label = '{}'.format(k))
    else:
        col = 'C{}'.format(n+1)
        ax.scatter([item[0] for item in NodeLocSets],[item[1] for item in NodeLocSets], s = ps,color = col, zorder=3, label = 'nodes')
    ax.legend(bbox_to_anchor=(0.0,1.15,1.0,0.1),loc='upper center', ncol = 1+ len(SubNetworks)+len(NodeLocSets),mode='expand', handletextpad = 0.1, handlelength = 1)
    ax.autoscale()
    #plt.show

#Plot an edge property as coloured edges over the main graph. Drawing order may be controlled 
def PlotNetworkEdgeProp(Network, SubNetworks = None, prop = 'weight', lw = None, cmap = 'pink', maxprop = None, minprop = None, ordered = True, fig = None, ax = None):
    if not ax:
        fig,ax = plt.subplots()
    lines = NetworkEdgestoArr(Network)
    #reorder for drawing by property
    if ordered:
        lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
    else:
        lines = list(lines.items())
    lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = 'gray', zorder=1, label = 'main network')
    ax.set_aspect('equal')
    ax.add_collection(lc)
    n=0
    if SubNetworks:
        if type(SubNetworks) is dict:
            for i,j in SubNetworks.items():
                lines = NetworkEdgestoArr(SubNetworks[i], prop = prop)
                #reorder for drawing by property
                if ordered:
                    lines = sorted(lines.items(),key=lambda x:x[1][prop])
                else:
                    lines = list(lines.items())
                lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, clim = (minprop, maxprop), cmap = cmap, zorder=2, label = prop)
                lc.set_array([e[prop] for u,e in lines])
                ax.add_collection(lc)
                n +=1
        else:        #assume it is a graph
            lines = NetworkEdgestoArr(SubNetworks,prop = prop)
            #reorder for drawing by property
            if ordered:
                lines = sorted(lines.items(),key=lambda x:x[1][prop])
            else:
                lines = list(lines.items())
            lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, clim = (minprop,maxprop), cmap = cmap, zorder=2, label = prop)
            lc.set_array([e[prop] for u,e in lines])
            ax.add_collection(lc)
    fig.colorbar(lc, ax = ax, fraction=0.010, pad=0.02)
    ax.autoscale()
    ax.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol = 2)

#Plot a node property as coloured nodes over the main graph. Drawing order may be controlled     
def PlotNetworkNodeProp(Network,SubNetworks, lw = None, ps = None, prop = None, cmap = 'pink', norm = None, maxprop = None, minprop = None, ordered = True, fig = None, ax = None):
    if not ax:
        fig,ax = plt.subplots()
    lines = NetworkEdgestoArr(Network)
    #reorder for drawing by property
    if ordered:
        lines = sorted(lines.items(),key=lambda x:x[1]['weight'])
    else:
        lines = list(lines.items())
    lc = LineCollection([e['coords'] for u,e in lines],linewidth=lw, color = 'gray', zorder=1,label = 'main network')
    ax.set_aspect('equal')
    ax.add_collection(lc)
    n = 0
    if SubNetworks:
        if type(SubNetworks) is dict:
            for i,j in SubNetworks.items():
                nodes = NetworkNodestoArr(SubNetworks[i], prop = prop)
                #reorder for drawing by property
                if ordered:
                    nodes = sorted(nodes.items(),key=lambda x:x[1][prop])
                ax.scatter([e['coords'][0] for u,e in nodes],[e['coords'][1] for u,e in nodes], s = ps,c = [e[prop] for u,e in nodes], cmap = cmap, zorder=3, label = '{}'.format(prop))
                n +=1
        else:        #assume it is a graph
            nodes = NetworkNodestoArr(SubNetworks, prop = prop)
            #reorder for drawing by property
            if ordered:
                nodes = sorted(nodes.items(),key=lambda x:x[1][prop])
            else:
                nodes = list(nodes.items())
            pc = ax.scatter([e['coords'][0] for u,e in nodes],[e['coords'][1] for u,e in nodes], s = ps, c = [e[prop] for u,e in nodes], norm = norm, cmap = cmap, zorder=3, label = '{}'.format(prop))
    fig.colorbar(pc, ax = ax, fraction=0.010, pad=0.02)
    ax.autoscale()
    ax.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol = 2)
