# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:05:35 2023

@author: Alan Aitken
"""
import mat73
from scipy.io import loadmat
import numpy as np
import networkx as nx

#Load the matlab file to a mesh object (a dict with the same structure as the matlab file)
def LoadMesh(file):
    try:
        mesh = loadmat(file)
    except NotImplementedError:
        mesh = mat73.loadmat(file, use_attrdict=True)#, only_include = 'para')
    return mesh

#This function gets the geometry for the mesh from the mesh object as numpy arrays
def getMeshArrays(mesh):
    node_array = mesh.para.dmesh.tri.nodes
    node_ids = [int(i+1) for i,j in enumerate(node_array)] #+1 for matlab vs python indexing
    edge_array = mesh.para.dmesh.tri.connect_edge
    edge_array = edge_array.astype(int)
    edge_length = mesh.para.dmesh.tri.edge_length
    return(node_array,node_ids,edge_array,edge_length)

#This function gets the boundary conditions for the mesh edges and nodes
def getBCs(mesh):
    b_mark_nodes = mesh.para.dmesh.tri.bmark
    b_mark_edges = mesh.para.dmesh.tri.bmark_edge
    return (b_mark_nodes,b_mark_edges)

#Access a 'constant' edge property from the mesh   
def getEdgeProp(mesh,prop):
    Prop = mesh.fields.edges[prop]
    return Prop

#Access a 'constant' node property from the mesh     
def getNodeProp(mesh,prop):
    Prop = mesh.fields.nodes[prop]
    return Prop

#Access a 'variable' edge property from the mesh for a timestep n  
def getEdgeVariable(mesh,var,n):
    Var = mesh[var][:,n]
    return Var

#Access a 'variable' node property from the mesh for a timestep n     
def getNodeVariable(mesh,var,n):
    Var = mesh[var][:,n]
    return Var

#get the scale factor for a variable - GraphSSet uses raw SI
def getScale(mesh,con):
    Con = mesh.para.scale[con]
    return Con

#get a numerical constant
def getConstant(mesh,con):
    Con = mesh.para.physical[con]
    return Con

#get the time for a timestep
def getTime(mesh,con,n):
    Con = mesh.para.time[con][n]
    return Con


def GladstoNetworkX_one(mesh, step = -1, weight_by = 'length'):
    """A method to generates a directed and weighted NetworkX graph from ONE hydrology model timestep (by default, the last one)"""
    #get mesh geometry
    node_arr, node_ids,edge_arr, edge_l = getMeshArrays(mesh)
    #get BCs
    BC_nodes, BC_edges = getBCs(mesh)
    #get moulin input
    scale = getScale(mesh,'source_term_c')
    moulin_flux = getNodeProp(mesh,'source_c')*scale
    
    #hydraulic potential gradient for flow direction
    scale = getScale(mesh,'phi')
    node_phi = getNodeVariable(mesh,'phis',step)*scale
    node_phi0 = getNodeProp(mesh,'phi_0')*scale
    edge_phis = [[node_phi[j[0]-1],node_phi[j[1]-1]] for i, j in enumerate(edge_arr)]
    edge_coords = [[node_arr[j[0]-1],node_arr[j[1]-1]] for i, j in enumerate(edge_arr)]
    l = lambda a,b,c: (a-b)/c
    edge_phi_grad = np.array([l(j[0],j[1],edge_l[i]) for i, j in enumerate(edge_phis)])

    #Edge Status    
    edge_status = np.ones_like(edge_phi_grad)
    for i,j in enumerate(edge_arr):
        #identify if both edges are floating (flag = -1)
        if edge_phis[i][0] == 0.0 and edge_phis[i][1]== 0.0:
            edge_status[i] = -1
        elif edge_phis[i][0] == 0.0 or edge_phis[i][1]== 0.0:
            edge_status[i] = 0
        elif BC_edges[i]==2.0:
            edge_status[i] = 3
        elif BC_nodes[j[0]-1] == 2.0 or BC_nodes[j[1]-1]==2.0:
            edge_status[i] = 2
        else:
            edge_status[i] = 1
            
    #Node Status
    #flag = -1 - 'floating' nodes where hydraulic potential is zero
    node_status = np.where(node_phi == 0.0,-1,1)
    #flag 0 - outlet nodes
    node_status = np.where(BC_nodes == 1.0, 0, node_status)        
    #flag = 2 - edge nodes 
    node_status = np.where(BC_nodes == 2.0, 2, node_status)
    #flag = 4 - moulins
    node_status = np.where(moulin_flux > 0.0, 4, node_status)
    
    #edge channel area
    scale = getScale(mesh,'S')
    edge_S = getEdgeVariable(mesh,'S_channels', step)*scale
    
    #calculate Effective Pressure
    node_N = node_phi0-node_phi
    
    #Channel Flux Q
    k_c = getConstant(mesh,'cond_c')
    alpha_c = getConstant(mesh,'alpha_c')
    beta_c = getConstant(mesh,'beta_c')
    edge_Q = k_c*edge_S**alpha_c*np.abs(edge_phi_grad)**(beta_c-2)*np.abs(edge_phi_grad)
    
    #weightings - high weights are higher cost
    if weight_by == 'length': #simple distance weighting
        edge_weights = edge_l/np.nanmax(edge_l)
    elif weight_by == 'hpg': #hpg only - higher weight for low HPG
        edge_weights = 1.0-edge_phi_grad/np.nanmax(edge_phi_grad*edge_l)
    elif weight_by == 'area': # higher weight for low area
        edge_weights = 1.0-edge_S/np.nanmax(edge_S)
    elif weight_by == 'flux': #higher weight for low flux
        edge_weights = 1.0-edge_Q/np.nanmax(edge_Q)
    else:
        print( 'weight_by choice not supported, using length')
        edge_weights = edge_l/np.nanmax(edge_l)
        
    #assign maximum weight to boundary edges
    edge_weights = np.where(BC_edges>0,1,edge_weights)
    
    #flow direction may be either way
    edge_flow_dir = np.where(edge_phi_grad>0,1,0)
    firsts = edge_arr[:,0]
    lasts = edge_arr[:,1]
    edge_up_node = np.where(edge_flow_dir == 1, firsts,lasts)
    edge_down_node = np.where(edge_flow_dir == 1, lasts,firsts)
    
    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - perhaps adjacency matrix is faster
    #but this is more clear - edge-linked nodes will be added by default
    for m,n in enumerate(edge_arr):
        w = edge_weights[m]
        c = edge_coords[m]
        #coords (first,last) may be reversed in network (up,down)
        #this is indicated by flow dir so we include that too
        d = edge_flow_dir[m]
        s = edge_status[m]
        l = edge_l[m]
        bc = BC_edges[m]
        ca = edge_S[m]
        cf = edge_Q[m]
        hpg = np.abs(edge_phi_grad[m])
        #we add from up to down, giving direction
        DG.add_edge(edge_up_node[m],edge_down_node[m], weight = w, coords = c, status = s, length = l, direction = d, hyd_pot_grad = hpg, channel_area = ca, channel_flux  = cf,edge_bc = bc)

    #make and add node attributes
    bed_elevation = getNodeProp(mesh,'bed_ele')
    ice_thick = getNodeProp(mesh,'ice_thick')
    surface_elevation = bed_elevation + ice_thick
    scale = getScale(mesh,'u_bed')
    u_bed = getNodeProp(mesh,'u_bed')*scale
    h_sheets = getNodeVariable(mesh,'h_sheets', step)
    for j,k in enumerate(node_ids):
        DG.nodes[k]["node_status"]=node_status[j]
        DG.nodes[k]["BC_nodes"]=BC_nodes[j]
        DG.nodes[k]["coords"]=node_arr[j]
        DG.nodes[k]["bed_elevation"]=bed_elevation[j]
        DG.nodes[k]["surface_elevation"]=surface_elevation[j]
        DG.nodes[k]["hydraulic_potential"]=node_phi[j]
        DG.nodes[k]["effective_pressure"]=node_N[j]
        DG.nodes[k]["basal_velocity_magnitude"]=u_bed[j]
        DG.nodes[k]["h_sheet"]=h_sheets[j]
    return DG

def GladstoNetworkX_multi(mesh, min_step = 0, max_step = -1, step_size = 1, weight_by = 'length'):
    """A method to generate a directed and weighted NetworkX graph from a hydrology model timestep series"""
    #get size for arrays
    arr_length = np.shape(mesh.phis)[1]
    if max_step < 0:
        max_step = arr_length + max_step
    if max_step <= min_step:
        raise ValueError ('max_step must be greater than min_step')
    arr_length = max_step-min_step
    if step_size > 1:
        arr_length = arr_length//step_size
    
    step_times = np.empty(arr_length)
    #get step times
    for n in range(min_step,max_step,step_size):
        t = (n-min_step)//step_size
        step_times[t] = getTime(mesh,'out_t',n)
    
    #get mesh geometry
    node_arr, node_ids,edge_arr, edge_l = getMeshArrays(mesh)
    edge_coords = [(node_arr[j[0]-1],node_arr[j[1]-1]) for i, j in enumerate(edge_arr)]
    #get BCs
    BC_nodes, BC_edges = getBCs(mesh)
    #get moulin input - seems to be a constant here
    scale = getScale(mesh,'source_term_c')
    moulin_flux = getNodeProp(mesh,'source_c')*scale
    
    #hydraulic potential gradients for flow direction
    scale = getScale(mesh,'phi')
    node_phi0 = getNodeProp(mesh,'phi_0')*scale
    node_phi = np.empty((arr_length,len(node_arr)))
    edge_phi = np.empty((arr_length, len(edge_arr),2))
    edge_phi_grad = np.empty((arr_length, len(edge_arr)))
    for n in range(min_step,max_step,step_size):
        t = (n-min_step)//step_size
        node_phi[t] = getNodeVariable(mesh,'phis',n)*scale
        edge_phi[t] = [[node_phi[t][j[0]-1],node_phi[t][j[1]-1]] for i, j in enumerate(edge_arr)]
    l = lambda a,b,c: (a-b)/c
    for n in range(min_step,max_step,step_size):
        t = (n-min_step)//step_size
        edge_phi_grad[t] = np.array([l(j[0],j[1],edge_l[i]) for i, j in enumerate(edge_phi[t])])
        
    #Edge Status    
    edge_status = np.ones_like(edge_phi_grad)
    for n,m in enumerate(edge_status):
        for i,j in enumerate(edge_arr):
            #identify if both nodes are floating (flag = -1)
            if edge_phi[n][i][0] == 0.0 and edge_phi[n][i][1]== 0.0:
                edge_status[n][i] = -1
            elif edge_phi[n][i][0] == 0.0 or edge_phi[n][i][1]== 0.0:
                edge_status[n][i] = 0
            elif BC_edges[i]==2.0:
                edge_status[n][i] = 3
            elif BC_nodes[j[0]-1] == 2.0 or BC_nodes[j[1]-1]==2.0:
                edge_status[n][i] = 2
            else:
                edge_status[n][i] = 1
            
    #Node Status
    #flag = -1 - 'floating' nodes where hydraulic potential is zero
    node_status = np.where(node_phi == 0.0,-1,1)
    for n,m in enumerate(node_status):
        #each n is one step
        #flag 0 - outlet nodes
        node_status[n] = np.where(BC_nodes == 1.0, 0, m)        
        #flag = 2 - edge nodes 
        node_status[n] = np.where(BC_nodes == 2.0, 2, m)
        #flag = 4 - moulins
        node_status[n] = np.where(moulin_flux > 0.0, 4, m)
    
    #flow direction may be either way
    edge_flow_dir = np.where(edge_phi_grad>0,1,0)
    #transpose to get list by edge:
    efd = np.transpose(edge_flow_dir)
    #identify the dominant flow direction
    fd_dom = np.empty(len(efd))
    for m,n in enumerate(efd):
        fd_dval,fd_dcount = np.unique(n,return_counts = True)
        fd_dom[m] = fd_dval[np.argmax(fd_dcount)]
    firsts = edge_arr[:,0]
    lasts = edge_arr[:,1]
    edge_up_node = np.where(fd_dom == 1, firsts,lasts)
    edge_down_node = np.where(fd_dom == 1, lasts,firsts)
    
    #edge channel area
    scale = getScale(mesh,'S')
    edge_S = np.empty((arr_length, len(edge_arr)))
    for n in range(min_step,max_step,step_size):
        t = (n-min_step)//step_size
        edge_S[t] = getEdgeVariable(mesh,'S_channels', n)*scale
    
    #calculate Effective Pressure
    node_N = node_phi0-node_phi
    
    #Channel Flux Q
    k_c = getConstant(mesh,'cond_c')
    alpha_c = getConstant(mesh,'alpha_c')
    beta_c = getConstant(mesh,'beta_c')
    edge_Q = k_c*edge_S**alpha_c*np.abs(edge_phi_grad)**(beta_c-2)*np.abs(edge_phi_grad)
    
    #weightings - high weights are higher cost
    if weight_by == 'length': #simple distance weighting
        edge_weights = edge_l/np.nanmax(edge_l)
    elif weight_by == 'hpg': #hpg only - higher weight for low HPG
        edge_weights = 1.0-edge_phi_grad/np.nanmax(edge_phi_grad*edge_l)
    elif weight_by == 'area': # higher weight for low area
        edge_weights = 1.0-edge_S/np.nanmax(edge_S)
    elif weight_by == 'flux': #higher weight for low flux
        edge_weights = 1.0-edge_Q/np.nanmax(edge_Q)
    else:
        print( 'weight_by choice not supported, using length')
        edge_weights = edge_l/np.nanmax(edge_l)
        
    #assign maximum weight to boundary edges
    edge_weights = np.where(BC_edges>0,1,edge_weights)
    
    #transpose weights array and take the minimum
    
    ew = np.transpose(edge_weights)
    weight = np.ones_like(edge_l)
    for i,j in enumerate(ew):
        weight[i] = np.nanmin(j)
    
    #transpose arrays for edgewise input
    es = np.transpose(edge_status)
    cs = np.transpose(edge_S)
    eq = np.transpose(edge_Q)
    epg = np.transpose(edge_phi_grad)
    
    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - perhaps adjacency matrix is faster
    #but this is more clear - edge-linked nodes will be added by default
    for m,n in enumerate(edge_arr):
        #add values for data
        w = weight[m]
        c = edge_coords[m]
        bc = BC_edges[m]
        l = edge_l[m]
        fd = fd_dom[m]
        es_i = es[m][0]
        ca_i = cs[m][0]
        cf_i = eq[m][0]
        hpg_i = np.abs(epg[m][0])
        #times for each step...
        ts = step_times
        #things that change are also entered as arrays
        cds = efd[m]
        ess = es[m]
        cas = cs[m]
        cfs = eq[m]
        hpgs = np.abs(epg[m])
        
        #we add from up to down, giving direction
        DG.add_edge(edge_up_node[m],edge_down_node[m], weight = w, 
                    coords = c, length = l, edge_bc = bc, 
                    times_arr = ts, 
                    status = es_i, edge_status_arr = ess, 
                    direction = fd, direction_arr = cds, 
                    hyd_pot_grad = hpg_i, hyd_pot_grad_arr = hpgs, 
                    channel_area = ca_i, channel_area_arr = cas, 
                    channel_flux = cf_i, channel_flux_arr  = cfs,
                    )

    ns = np.transpose(node_status)
    nphi = np.transpose(node_phi)
    nN = np.transpose(node_N)

    #make and add node attributes
    scale = getScale(mesh,'z')
    bed_elevation = getNodeProp(mesh,'bed_ele')*scale
    ice_thick = getNodeProp(mesh,'ice_thick')*scale
    surface_elevation = bed_elevation + ice_thick
    scale = getScale(mesh,'u_bed')
    u_bed = getNodeProp(mesh,'u_bed')*scale
    h_sheets = np.empty_like(node_phi)
    for n in range(min_step,max_step,step_size):
        t = (n-min_step)//step_size
        h_sheets[t] = getNodeVariable(mesh,'h_sheets', n)
    hs = np.transpose(h_sheets)
    for j,k in enumerate(node_ids):
        #static fields as values
        DG.nodes[k]["coords"]=node_arr[j]
        DG.nodes[k]["BC_nodes"]=BC_nodes[j]
        DG.nodes[k]["bed_elevation"]=bed_elevation[j]
        DG.nodes[k]["surface_elevation"]=surface_elevation[j]
        DG.nodes[k]["basal_velocity_magnitude"]=u_bed[j]
        DG.nodes[k]["node_status"]=ns[j][0]
        DG.nodes[k]["hydraulic_potential"]=nphi[j][0]
        DG.nodes[k]["effective_pressure"]=nN[j][0]
        DG.nodes[k]["h_sheet"]=hs[j][0]
        #variable fields also as arrays
        DG.nodes[k]["node_status_arr"]=ns[j]
        DG.nodes[k]["hydraulic_potential_arr"]=nphi[j]
        DG.nodes[k]["effective_pressure_arr"]=nN[j]
        DG.nodes[k]["h_sheet_arr"]=hs[j]
    return DG