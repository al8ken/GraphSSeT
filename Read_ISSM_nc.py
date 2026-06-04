# -*- coding: utf-8 -*-
"""
This script reads in the netcdf data from and ISSM model as exported using this tool

https://github.com/ISSMteam/ISSM/tree/c8b08554d5f39232b25aa676ef6eaae742ab5c0e/src/m/contrib/musselman

Created on Fri Jan 24 09:05:35 2025

@author: Alan Aitken
"""
from netCDF4 import Dataset
import numpy as np
import networkx as nx

#Load the netCDF file
def LoadNetCDF(file):
    try:
        nc_data = Dataset(file, "r", format="NETCDF4")
    except NotImplementedError:
        print('netCDF file not read')
    return nc_data

#This function gets the geometry for the mesh from the mesh object as numpy arrays
def getModelGroup(Model, group):
    Group = Model.groups[group]
    return Group

def getSolutionSlice(Model, solution, step = -1):
    SS = Model.groups['results'][solution].groups
    stepkey = list(SS.keys())[step]
    SS_slice = SS[stepkey]
    return SS_slice

def getSolutionSlices(Model, solution, minstep = 0, maxstep = -1):
    SS = Model.groups['results'][solution].groups
    stepkeys = list(SS.keys())[minstep:maxstep]
    #return as a dict
    SS_slice = {key: SS[key] for key in stepkeys}
    return SS_slice

def getMeshArrays(mesh):  
    #get the nodes as i,x,y
    Xs = np.array(mesh['x'])
    Ys = np.array(mesh['y'])
    node_array = np.array([[i,j,Ys[i]] for i,j in enumerate(Xs)])
    #edges just connect nodes- we don't care about elements here
    edges = np.array(mesh['edges'])  #format [node1,node2,element1,element2]
    edge_array = np.array([[i,int(j[0]-1),int(j[1]-1)] for i,j in enumerate(edges)]) #ID is -1 to account for matlab v python counting...
    #this means the edge_ids won't line up exactly -- maybe this is different for python-based ISSM model?
    return(node_array,edge_array)

#This function gets the boundary conditions for the mesh edges and nodes
def getBCs(mesh,edge_array, hydro):
    b_mark_nodes = np.array(mesh['vertexonboundary']) #1 if on boundary, 0 otherwise
    #0 for internal nodes, 1 for boundary-contacting edges, 2 for boundary edges
    b_mark_edges = [b_mark_nodes[j[1]]+b_mark_nodes[j[2]] for i,j in enumerate(edge_array)]
    o_mark_nodes = np.array(hydro['spcphi']) #0 if on outlet, NaN otherwise
    return (b_mark_nodes,b_mark_edges, o_mark_nodes)
        
#Access a 'constant' edge property -- one that has only one entry for the mode duration
def getEdgeProp(data,prop):
    Prop = np.array(data[prop])
    return Prop

#Access a 'constant' node property from the mesh     
def getNodeProp(data,prop):
    Prop = np.array(data[prop])
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

def ISSMtoNetworkX_init(Model, weight_by = 'length'):
    """A method to generates a directed and weighted NetworkX graph from the initial hydrology model mesh"""
    #get mesh and data objects
    mesh = getModelGroup(Model, 'mesh')
    geom = getModelGroup(Model, 'geometry')
    hydro = getModelGroup(Model, 'hydrology')
    mask = getModelGroup(Model, 'mask')
    hydro = getModelGroup(Model, 'hydrology')
    
    #get mesh arrays
    node_array, edge_array = getMeshArrays(mesh)
    node_ids = np.array([j[0] for i,j in enumerate(node_array)])
    #node coordinates
    node_coords = [(j[1],j[2]) for i, j in enumerate(node_array)]
    # edge coordinates
    edge_coords = [(node_coords[j[1]],node_coords[j[2]]) for i, j in enumerate(edge_array)]
    #edge_length
    l = lambda x1,y1,x2,y2: np.sqrt((x2-x1)**2+(y2-y1)**2) 
    edge_l = [l(c[0][0],c[0][1],c[1][0],c[1][1]) for c in edge_coords]
    #get BCs
    BC_nodes, BC_edges = getBCs(mesh, edge_array)
    print('made arrays')
    #node_mask for ice and ocean status
    ocean = getNodeProp(mask, 'ocean_levelset') # presence of ocean if < 0, coastline/grounding line if = 0, no ocean if > 0
    ice = getNodeProp(mask, 'ice_levelset') #presence of ice if < 0, icefront position if = 0, no ice if > 0
    #make 2D array
    OI = np.vstack((ocean,ice)).T
    mask = np.where(OI==[-1,1],0,99) # open ocean, 99 for unclassified
    mask = np.where(OI==[-1,0],4,mask) # ice shelf front
    mask = np.where(OI==[-1,-1],1,mask) # ice shelf
    mask = np.where(OI==[0,1],5,mask) # ice free coast
    mask = np.where(OI==[0,0],6,mask) # icefront at coast (no shelf)
    mask = np.where(OI==[0,-1],7,mask) # grounding line
    mask = np.where(OI==[1,1],2,mask) # land
    mask = np.where(OI==[1,0],8,mask) # grounded ice front
    mask = np.where(OI==[1,-1],3,mask) # grounded ice
    print('masking complete')
    #get stable node properties
    try:
        moulin_flux = getNodeProp(hydro,'moulin_input') #moulin input
    except IndexError:
        print("moulin_input not found, assigning zero")
        moulin_flux = np.zeros_like(node_ids)
    surface_elevation = getNodeProp(geom,'surface') #surface elevation
    ice_thick = getNodeProp(geom,'thickness') #ice thickness
    bed_elevation = getNodeProp(geom,'bed') #bed elevation
    try:
        bump_height = getNodeProp(hydro,'bump_height')  #bedrock bump height
    except IndexError:
        print("bump_height not found, assigning zero")
        bump_height = np.zeros_like(node_ids)
    
    #calculate SHREVE hydraulic potential gradient for flow direction
    l = lambda h,b: 917*9.81*h + 1000*9.81*b
    node_phi = [l(j,bed_elevation[i]) for i,j in enumerate(ice_thick)]
    edge_phis = [[node_phi[j[1]],node_phi[j[2]]] for i, j in enumerate(edge_array)]
    l = lambda a,b,c: (a-b)/c
    edge_phi_grad = np.array([l(j[0],j[1],edge_l[i]) for i, j in enumerate(edge_phis)])  
    #Edge Status    
    edge_status = np.ones_like(edge_phi_grad)
    for i,j in enumerate(edge_array):
        #identify if both edges are floating (flag = -1)
        if edge_phis[i][0] == 0.0 and edge_phis[i][1]== 0.0:
            edge_status[i] = -1 #not part of the active model
        elif edge_phis[i][0] == 0.0 or edge_phis[i][1]== 0.0:
            edge_status[i] = 0 #outlet edge if one floating
        elif BC_edges[i]==2.0: 
            edge_status[i] = 3 #boundary edge
        elif BC_edges[i] == 1.0:
            edge_status[i] = 2 #boundary contacting edge (may be an outlet)
        else:
            edge_status[i] = 1 #normal edge
            
    #Node Status
    #flag = -1 - 'floating' nodes where hydraulic potential is zero
    node_status = np.where(node_phi == 0.0,-1,1)   
    #flag 0 - outlet nodes if specified in BCs
    #node_status = np.where(BC_nodes == 2.0, 0, node_status) 
    #flag = 2 - edge nodes (may be outlets) 
    node_status = np.where(BC_nodes == 1.0, 2, node_status)
    #flag = 4 - moulins
    node_status = np.where(moulin_flux > 0.0, 4, node_status)
      
    #weightings - high weights are higher cost
    if weight_by == 'length': #simple distance weighting
        edge_weights = edge_l/np.nanmax(edge_l)
    elif weight_by == 'hpg': #hpg only - higher weight for low HPG
        edge_weights = 1.0-edge_phi_grad/np.nanmax(edge_phi_grad*edge_l)
    else:
        print( 'weight_by choice not supported, using length')
        edge_weights = edge_l/np.nanmax(edge_l)
        
    #assign maximum weight to boundary edges
    edge_weights = np.where(BC_edges==2,1,edge_weights)
    
    #flow direction may be either way
    edge_flow_dir = np.where(edge_phi_grad>0,1,0)
    firsts = edge_array[:,1]
    lasts = edge_array[:,2]
    edge_up_node = np.where(edge_flow_dir == 1, firsts,lasts)
    edge_down_node = np.where(edge_flow_dir == 1, lasts,firsts)
    print('begin making graph with {} edges'.format(len(edge_array)))
    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - perhaps adjacency matrix is faster
    #but this is more clear - edge-linked nodes will be added by default
    for m,n in enumerate(edge_array):
        if m % 100000 == 0:
            print('edge number {}'.format(m))
        w = edge_weights[m]
        c = edge_coords[m]
        #coords (first,last) may be reversed in network (up,down)
        #this is indicated by flow dir so we include that too
        d = edge_flow_dir[m]
        s = edge_status[m]
        l = edge_l[m]
        bc = BC_edges[m]
        hpg = np.abs(edge_phi_grad[m])
        #we add from up to down, giving direction
        DG.add_edge(edge_up_node[m],edge_down_node[m], weight = w, coords = c, status = s, length = l, direction = d, hyd_pot_grad = hpg,edge_bc = bc)
    print('all edges added')
    #add node attributes - not all are essential
    for j,k in enumerate(node_ids):
        DG.nodes[k]["node_status"]=node_status[j]
        DG.nodes[k]["BC_nodes"]=BC_nodes[j]
        DG.nodes[k]["coords"]=node_coords[j]
        DG.nodes[k]["bed_elevation"]=bed_elevation[j]
        DG.nodes[k]["surface_elevation"]=surface_elevation[j]
        DG.nodes[k]["ice_thickness"]=ice_thick[j]
        DG.nodes[k]["hydraulic_potential"]=node_phi[j]
        DG.nodes[k]["moulin_flux"]=moulin_flux[j]
        DG.nodes[k]["bump_height"]=bump_height[j]
    print('node props added')
    return DG


def ISSMtoNetworkX_one(Model, step = -1, weight_by = 'length'):
    """A method to generates a directed and weighted NetworkX graph from ONE hydrology model timestep (by default, the last one)"""
    #get mesh and data objects
    mesh = getModelGroup(Model, 'mesh')
    geom = getModelGroup(Model, 'geometry')
    hydro = getModelGroup(Model, 'hydrology')
    mask = getModelGroup(Model, 'mask')
    hydro = getModelGroup(Model, 'hydrology')
    init = getModelGroup(Model, 'initialization')
    TS = getSolutionSlice(Model,'TransientSolution', step)
    #get mesh arrays
    node_array, edge_array = getMeshArrays(mesh)
    node_ids = np.array([j[0] for i,j in enumerate(node_array)])
    #node coordinates
    node_coords = [(j[1],j[2]) for i, j in enumerate(node_array)]
    # edge coordinates
    edge_coords = [(node_coords[j[1]],node_coords[j[2]]) for i, j in enumerate(edge_array)]
    #edge_length
    l = lambda x1,y1,x2,y2: np.sqrt((x2-x1)**2+(y2-y1)**2) 
    edge_l = [l(c[0][0],c[0][1],c[1][0],c[1][1]) for c in edge_coords]
    #get BCs
    BC_nodes, BC_edges, BC_outlet = getBCs(mesh, edge_array, hydro)
    
    #node_mask for ice and ocean status
    ocean = getNodeProp(mask, 'ocean_levelset') # presence of ocean if < 0, coastline/grounding line if = 0, no ocean if > 0
    ice = getNodeProp(mask, 'ice_levelset') #presence of ice if < 0, icefront position if = 0, no ice if > 0
    #make 2D array
    OI = np.vstack((ocean,ice)).T
    mask = np.where(OI==[-1,1],0,99) # open ocean, 99 for unclassified
    mask = np.where(OI==[-1,0],4,mask) # ice shelf front
    mask = np.where(OI==[-1,-1],1,mask) # ice shelf
    mask = np.where(OI==[0,1],5,mask) # ice free coast
    mask = np.where(OI==[0,0],6,mask) # icefront at coast (no shelf)
    mask = np.where(OI==[0,-1],7,mask) # grounding line
    mask = np.where(OI==[1,1],2,mask) # land
    mask = np.where(OI==[1,0],8,mask) # grounded ice front
    mask = np.where(OI==[1,-1],3,mask) # grounded ice
    
    #get stable node properties
    try:
        moulin_flux = getNodeProp(hydro,'moulin_input') #moulin input
    except IndexError:
        print("moulin_input not found, assigning zero")
        moulin_flux = np.zeros_like(node_ids)
    surface_elevation = getNodeProp(geom,'surface') #surface elevation
    ice_thick = getNodeProp(geom,'thickness') #ice thickness
    bed_elevation = getNodeProp(geom,'bed') #bed elevation
    basal_velocity_magnitude = getNodeProp(init,'vel')/365/24/3600 #basal velocity [m/s]
    try:
        bump_height = getNodeProp(hydro,'bump_height')  #bedrock bump height
    except IndexError:
        print("bump_height not found, assigning zero")
        bump_height = np.zeros_like(node_ids)
    #get variables from TS and SBS    
    #edge channel area and discharge
    edge_S = getEdgeProp(TS,'ChannelArea')
    edge_Q = np.abs(getEdgeProp(TS,'ChannelDischarge'))
    #node effective pressure, sheet thickness and basal velocity
    node_phi = getNodeProp(TS,'HydraulicPotential') #hydraulic potential
    node_N = getNodeProp(TS,'EffectivePressure')
    h_sheets = getNodeProp(TS,'HydrologySheetThickness')
    
    #calculate hydraulic potential gradient
    edge_phis = [[node_phi[j[1]],node_phi[j[2]]] for i, j in enumerate(edge_array)]
    l = lambda a,b,c: (a-b)/c
    edge_phi_grad = np.array([l(j[0],j[1],edge_l[i]) for i, j in enumerate(edge_phis)])  
    #Edge Status    
    edge_status = np.ones_like(edge_phi_grad)
    for i,j in enumerate(edge_array):
        #identify if both edges are floating (flag = -1)
        if edge_phis[i][0] == 0.0 and edge_phis[i][1]== 0.0:
            edge_status[i] = -1 #not part of the active model
        elif edge_phis[i][0] == 0.0 or edge_phis[i][1]== 0.0:
            edge_status[i] = 0 #outlet edge if one floating
        elif BC_edges[i]==2.0: 
            edge_status[i] = 3 #boundary edge
        elif BC_edges[i] == 1.0:
            edge_status[i] = 2 #boundary contacting edge (may be an outlet)
        else:
            edge_status[i] = 1 #normal edge
            
    #Node Status
    #flag = -1 - 'floating' nodes where hydraulic potential is zero
    node_status = np.where(node_phi == 0.0,-1,1)   
    #flag = 2 - edge nodes (may be outlets) 
    node_status = np.where(BC_nodes == 1.0, 2, node_status)
    #flag 0 - outlet nodes if specified in BCs
    node_status = np.where(BC_outlet == 0.0, 0, node_status) 
    #flag = 4 - moulins
    node_status = np.where(moulin_flux > 0.0, 4, node_status)
      
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
    edge_weights = np.where(BC_edges==2,1,edge_weights)
    
    #flow direction may be either way
    edge_flow_dir = np.where(edge_phi_grad>0,1,0)
    firsts = edge_array[:,1]
    lasts = edge_array[:,2]
    edge_up_node = np.where(edge_flow_dir == 1, firsts,lasts)
    edge_down_node = np.where(edge_flow_dir == 1, lasts,firsts)
        
    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - perhaps adjacency matrix is faster
    #but this is more clear - edge-linked nodes will be added by default
    for m,n in enumerate(edge_array):
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
    
    #add node attributes - not all are essential
    for j,k in enumerate(node_ids):
        DG.nodes[k]["node_status"]=node_status[j]
        DG.nodes[k]["BC_nodes"]=BC_nodes[j]
        DG.nodes[k]["coords"]=node_coords[j]
        DG.nodes[k]["bed_elevation"]=bed_elevation[j]
        DG.nodes[k]["basal_velocity_magnitude"]=basal_velocity_magnitude[j]
        DG.nodes[k]["surface_elevation"]=surface_elevation[j]
        DG.nodes[k]["ice_thickness"]=ice_thick[j]
        DG.nodes[k]["hydraulic_potential"]=node_phi[j]
        DG.nodes[k]["effective_pressure"]=node_N[j]
        DG.nodes[k]["h_sheet"]=h_sheets[j]
        DG.nodes[k]["moulin_flux"]=moulin_flux[j]
        DG.nodes[k]["bump_height"]=bump_height[j]
    return DG

def ISSMtoNetworkX_multi(Model, minstep = 0, maxstep = -1, stepsize = 1, mintime = None, maxtime = None, weight_by = 'length'):
    """A method to generates a directed and weighted NetworkX graph from many hydrology model timesteps (by default, all)"""
    #get mesh and data objects
    mesh = getModelGroup(Model, 'mesh')
    geom = getModelGroup(Model, 'geometry')
    hydro = getModelGroup(Model, 'hydrology')
    mask = getModelGroup(Model, 'mask')
    hydro = getModelGroup(Model, 'hydrology')
    stepkeys = list(Model.groups['results']['TransientSolution'].groups.keys())
    if stepsize > 1:
        try:
            stepkeys = stepkeys[0::stepsize]
        except TypeError:
            stepkeys = stepkeys[0::int(stepsize)]    
    #if times are given figure out the step; min and max are fully independent
    if mintime != None:
        for i,key in enumerate(stepkeys):   
            t = Model.groups['results']['TransientSolution'].groups[key]['time']
            if t > mintime:
                minstep = i
                break
    if maxtime != None:
        for i,key in enumerate(stepkeys):   
            t = Model.groups['results']['TransientSolution'].groups[key]['time']
            if t > maxtime:
                maxstep = i-1
                break
    if maxstep<=minstep and maxstep > 0:
        raise ValueError('maxstep is not less then minstep')
    TS = getSolutionSlices(Model,'TransientSolution', minstep, maxstep)#get size for arrays
    TSkeys = TS.keys()    
    times = np.array([TS[key]['time'] for key in TSkeys]).flatten()
   #steps = np.array([TS[key]['step'] for key in TSkeys])
    
    #get mesh arrays
    node_array, edge_array = getMeshArrays(mesh)
    node_ids = np.array([j[0] for i,j in enumerate(node_array)])
    #node coordinates
    node_coords = [(j[1],j[2]) for i, j in enumerate(node_array)]
    # edge coordinates
    edge_coords = [(node_coords[j[1]],node_coords[j[2]]) for i, j in enumerate(edge_array)]
    #edge_length
    l = lambda x1,y1,x2,y2: np.sqrt((x2-x1)**2+(y2-y1)**2) 
    edge_l = [l(c[0][0],c[0][1],c[1][0],c[1][1]) for c in edge_coords]
    #get BCs
    BC_nodes, BC_edges = getBCs(mesh, edge_array)
    
    #node_mask for ice and ocean status - for this model these do not change
    ocean = getNodeProp(mask, 'ocean_levelset') # presence of ocean if < 0, coastline/grounding line if = 0, no ocean if > 0
    ice = getNodeProp(mask, 'ice_levelset') #presence of ice if < 0, icefront position if = 0, no ice if > 0
    if np.nanmin(ocean) > 0:
        print('no ocean mask applied (all grounded), just ice mask')
        mask = np.ones_like(ice)*99 # 99 for unclassified
        m = np.where(ice < 0 )[0] # grounded ice
        mask[m] = 3
        m = np.where(ice == 0 )[0] # glacial terminus
        mask[m] = 8
        m = np.where(ice > 0 )[0] # exposed land
        mask[m] = 9
    elif np.nanmin(ice) > 0:
        #NB this should not happen but included for completeness
        print('no ice mask applied (no ice present), just ocean mask - this perhaps is an error?')
        mask = np.ones_like(ocean)*99 # 99 for unclassified
        m = np.where(ocean < 0 )[0] # open ocean
        mask[m] = 0
        m = np.where(ocean == 0 )[0] # coastline
        mask[m] = 4
        m = np.where(ocean > 0 )[0] # exposed land
        mask[m] = 9
    else:
        print('both ocean and ice masks applied')
        #fix this later it does not apply to Finland
        # mask = np.ones_like(ocean)*99 # 99 for unclassified
        # m = np.where(OI==[-1,1])[0] # 0 open ocean
        # mask[m] = 0                
        # m = np.where(OI==[-1,0])[0] # ice shelf front
        # mask[m] = 4
        # m = np.where(OI==[-1,-1])[0]
        # mask[m] = 1 # ice shelf
        # m = np.where(OI==[0,1])[0]
        # mask[m] = 5 # ice free coast
        # m = np.where(OI==[0,0])[0]
        # mask[m] = 6 # icefront at coast (no shelf)
        # m = np.where(OI==[0,-1])[0]
        # mask[m] = 7 # grounding line
        # m = np.where(OI==[1,1])[0]
        # mask[m] = 2 # land
        # m = np.where(OI==[1,0])[0]
        # mask[m] = 8# grounded ice front
        # m = np.where(OI==[1,-1])[0]
    
    #get stable node properties
    try:
        moulin_flux = getNodeProp(hydro,'moulin_input') #moulin input
    except IndexError:
        print("moulin_input not found, assigning zero")
        moulin_flux = np.zeros_like(node_ids)
    surface_elevation = getNodeProp(geom,'surface') #surface elevation
    ice_thick = getNodeProp(geom,'thickness') #ice thickness
    bed_elevation = getNodeProp(geom,'bed') #bed elevation
    try:
        bump_height = getNodeProp(hydro,'bump_height')  #bedrock bump height
    except IndexError:
        print("bump_height not found, assigning zero")
        bump_height = np.zeros_like(node_ids)
    
    #set up arrays for variable properties from TS and SBS
    #nodeproperties
    blank_nodes = np.zeros(shape = (len(TSkeys),len(node_ids)))
    #edgeProperties
    blank_edges = np.zeros(shape = (len(TSkeys),len(edge_l)))
    t0 = times[0]
    for i,key in enumerate(TSkeys):
        if i == 0:
            #set up blank arrays
            node_phi = np.zeros_like(blank_nodes)
            node_N = np.zeros_like(blank_nodes)
            node_status = np.zeros_like(blank_nodes)
            h_sheets = np.zeros_like(blank_nodes)
            edge_S = np.zeros_like(blank_edges)
            edge_Q = np.zeros_like(blank_edges)
            edge_phi_grad = np.zeros_like(blank_edges)
            edge_status = np.ones_like(blank_edges)
            edge_weights = np.ones_like(blank_edges)
            edge_flow_dir = np.ones_like(blank_edges)
        data = TS[key]
        #edge channel area and discharge array
        edge_S[i] = getEdgeProp(data,'ChannelArea')
        edge_Q[i] = np.abs(getEdgeProp(data,'ChannelDischarge')) #negative values are eliminated
        #node effective pressure, sheet thickness and basal velocity
        node_phi[i] = getNodeProp(data,'HydraulicPotential') #hydraulic potential
        node_N[i] = getNodeProp(data,'EffectivePressure')
        h_sheets[i] = getNodeProp(data,'HydrologySheetThickness')
    
        #calculate hydraulic potential gradient
        edge_phis = [[node_phi[i][n[1]],node_phi[i][n[2]]] for m, n in enumerate(edge_array)]
        l = lambda a,b,c: (a-b)/c
        edge_phi_grad[i] = np.array([l(n[0],n[1],edge_l[m]) for m,n in enumerate(edge_phis)])  
        
        #Edge Status    
        for k,l in enumerate(edge_array):
            #identify if both edges are floating (flag = -1)
            if edge_phis[k][0] == 0.0 and edge_phis[k][1]== 0.0:
                edge_status[i][k] = -1 #not part of the active model
            elif edge_phis[k][0] == 0.0 or edge_phis[k][1]== 0.0:
                edge_status[i][k] = 0 #outlet edge if one floating
            elif BC_edges[k]==2.0: 
                edge_status[i][k] = 3 #boundary edge
            elif BC_edges[k] == 1.0:
                edge_status[i][k] = 2 #boundary contacting edge (may be an outlet)
            else:
                edge_status[i][k] = 1 #normal edge
            
        #Node Status
        #flag = -1 - 'floating' nodes where hydraulic potential is zero
        node_status[i] = np.where(node_phi[i] == 0.0,-1,1)   
        #flag 0 - outlet nodes if specified in BCs
        #node_status = np.where(BC_nodes == 2.0, 0, node_status) 
        #flag = 2 - edge nodes (may be outlets) 
        node_status[i] = np.where(BC_nodes == 1.0, 2, node_status[i])
        #flag = 4 - moulins
        node_status[i] = np.where(moulin_flux > 0.0, 4, node_status[i])
          
        #weightings - high weights are higher cost
        if weight_by == 'length': #simple distance weighting
            edge_weights[i] = edge_l/np.nanmax(edge_l)
        elif weight_by == 'hpg': #hpg only - higher weight for low HPG
            edge_weights[i] = 1.0-edge_phi_grad[i]/np.nanmax(edge_phi_grad[i]*edge_l)
        elif weight_by == 'area': # higher weight for low area
            edge_weights[i] = 1.0-edge_S[i]/np.nanmax(edge_S[i])
        elif weight_by == 'flux': #higher weight for low flux
            edge_weights[i] = 1.0-edge_Q[i]/np.nanmax(edge_Q[i])
        else:
            print( 'weight_by choice not supported, using length')
            edge_weights[i] = edge_l/np.nanmax(edge_l)
            
        #assign maximum weight to boundary edges
        edge_weights[i] = np.where(BC_edges==2,1,edge_weights[i])
        
        #flow direction may be either way
        edge_flow_dir[i] = np.where(edge_phi_grad[i]>0,1,0)
        
        #swap the sign for hpg where flow direction is reversed 
        edge_phi_grad[i] = np.where(edge_flow_dir[i]==0,edge_phi_grad[i]*-1,edge_phi_grad[i])
    
    firsts = edge_array[:,1]
    lasts = edge_array[:,2]
    edge_up_node = np.where(edge_flow_dir[0] == 1, firsts,lasts)
    edge_down_node = np.where(edge_flow_dir[0] == 1, lasts,firsts)

    #transpose arrays for edgewise input
    ew = np.transpose(edge_weights)
    es = np.transpose(edge_status)
    cas = np.transpose(edge_S)
    cfs = np.transpose(edge_Q)
    epg = np.transpose(edge_phi_grad)    
    efd = np.transpose(edge_flow_dir)
    #nodes
    ns = np.transpose(node_status)
    ns = np.transpose(node_status)
    nphi = np.transpose(node_phi)
    nN = np.transpose(node_N)
    nh = np.transpose(h_sheets)
        
    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - edge-linked nodes will be added by default
    t = (times-t0)*365*24*3600.
    for m,n in enumerate(edge_array):
        #things that dont change
        l = edge_l[m]
        bc = BC_edges[m]     
        c = edge_coords[m]
        #properties that change
        w = ew[m]
        d= efd[m]
        s = es[m]
        ca = cas[m]
        cf = cfs[m]
        #hpg = np.abs(epg[m])# here we do not allow negatives - all rests on flow dir
        hpg = epg[m] # here we allow negative hpg to indicate changes in edge direction (should match changes in flow dir)
        #we add from up to down, giving direction
        DG.add_edge(edge_up_node[m],
                    edge_down_node[m], 
                    coords = c, 
                    length = l,
                    status = s[0],
                    status_arr = s,
                    weight = w[0],
                    weight_arr = w,
                    direction = d[0],
                    direction_arr = d, 
                    hyd_pot_grad = hpg[0], 
                    hyd_pot_grad_arr = hpg, 
                    channel_area = ca[0], 
                    channel_area_arr = ca, 
                    channel_flux  = cf[0],
                    channel_flux_arr = cf,
                    edge_bc = bc, 
                    time = t,
                    last_time = t[0],
                    )              
 
    #add node attributes - not all are essential
    for j,k in enumerate(node_ids):
        if k in DG.nodes():
            #static props
            DG.nodes[k]["mask"]=mask[j]
            DG.nodes[k]["coords"]=node_coords[j]
            DG.nodes[k]["bed_elevation"]=bed_elevation[j]
            DG.nodes[k]["surface_elevation"]=surface_elevation[j]
            DG.nodes[k]["ice_thickness"]=ice_thick[j]
            DG.nodes[k]["BC_nodes"]=BC_nodes[j]
            DG.nodes[k]["moulin_flux"]=moulin_flux[j]
            DG.nodes[k]["bump_height"]=bump_height[j]
            #changing props
            DG.nodes[k]["node_status"]=ns[j][0]
            DG.nodes[k]["node_status_arr"]=ns[j]
            DG.nodes[k]["hydraulic_potential"]=nphi[j][0]
            DG.nodes[k]["hydraulic_potential_arr"]=nphi[j]
            DG.nodes[k]["effective_pressure"]=nN[j][0]
            DG.nodes[k]["effective_pressure_arr"]=nN[j]
            DG.nodes[k]["h_sheet"]=nh[j][0]
            DG.nodes[k]["h_sheet_arr"]=nh[j]
    return DG