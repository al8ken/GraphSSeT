# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:50:04 2023

@author: 00075859
"""
import meshio
import numpy as np
import networkx as nx
from scipy import stats, interpolate

def ReadVTUtoMeshio(Path):
    mesh = meshio.read(Path)
    return(mesh)

def InterpolateISMtoMesh(mesh, ism, prop, D = '3D', nanval = np.nan):
    if D =='3D':
        meshpoints = mesh.points.copy()
        meshpoints[:,2] = mesh.point_data['zb']
        points = ism.points.copy()
        points_data = ism.point_data[prop]
    elif D =='2D':
        meshpoints = np.delete(mesh.points,2,1)
        XYpoints = np.delete(ism.points,2,1)
        points, unq = np.unique(XYpoints, axis = 0, return_index = True)
        points_data = ism.point_data[prop][unq]
    elif D =='Basal':
        print('coming soon?')
    else: 
        raise ValueError('D is not supported')
    interp =  interpolate.LinearNDInterpolator(points,points_data, fill_value = nanval)
    prop_on_mesh = interp(meshpoints)
    return(prop_on_mesh)

def UnitstoSI(mesh,channels,n):
    y2s = 31556926.0
    #mesh properties
    #MPa to Pa
    mesh.point_data['hydraulic potential']=mesh.point_data['hydraulic potential']*1e6
    mesh.point_data['normalstress']=mesh.point_data['normalstress']*1e6
    mesh.point_data['effective pressure']=mesh.point_data['effective pressure']*1e6
    mesh.point_data['water pressure']=mesh.point_data['water pressure']*1e6
    #m^x/a to m^x/s
    mesh.point_data['sheet discharge']=mesh.point_data['sheet discharge']/y2s #m^2/a to m^2/s
    mesh.point_data['velocity']=mesh.point_data['velocity']/y2s #m/a to m/s
    mesh.point_data['beta']=mesh.point_data['beta']* 1e4/y2s #Mpa * a/m to Pa*s/m
    #channels properties
    #m^x/a to m^x/s
    channels.cell_data['Channel Flux'][n]=channels.cell_data['Channel Flux'][n]/y2s #m^3/a to m^2/s

def TrimOutliers(array, MinZscore = -3.5, MaxZscore = 3.5):
    #here we identify outliers and trim based on modified z score
    median = np.median(array)
    print(median)
    MAD = stats.median_abs_deviation(array)
    print(MAD)
    high = MaxZscore*MAD/0.6745 + median
    print(high)
    low = MinZscore*MAD/0.6745 + median
    print(low)
    array  = np.where(array > high,high,array)
    array = np.where(array < low,low,array)
    return array

def MeshiotoNetworkX_one(mesh,channels, variability = None):
    
    # check if channels points and mesh points have the same locations(within tolerance)
    # if this is true we can directly transfer between mesh and cells points.
    # if not the script below won't work
    plocs = np.isclose(mesh.points,channels.points, atol = 1)[0]
    sanity = np.where(plocs==False)
    print('mislocated points on axis {}'.format(sanity[0]))
    
    #add required new data fields to channels
    c_shape = np.shape(channels.cell_data["Channel Flux"][0])
    channels.cell_data["HP_diff"] = np.zeros(c_shape)
    channels.cell_data["HP_grad_abs"] = np.zeros(c_shape)
    channels.cell_data["flow_dir"] = np.zeros(c_shape, dtype = np.int64)
    channels.cell_data["segment_status"] = np.ones(c_shape, dtype = np.int64)
    channels.cell_data["segment_length"] = np.zeros(c_shape)
    if variability is not None:
        channels.cell_data["HP_diff"+variability] = np.zeros(c_shape)
        channels.cell_data["HP_grad_abs"+variability] = np.zeros(c_shape)
        
    #add coordinates to channels as [[X0,Y0],[X1,Y1]]
    channels.cell_data["coords"] = np.zeros((c_shape[0],2,2))
    firsts = channels.cells_dict['line'][:,0]
    lasts = channels.cells_dict['line'][:,1]
    
    for m,n in enumerate(firsts):
        channels.cell_data["coords"][m,0,0] = channels.points[n][0]
        channels.cell_data["coords"][m,0,1] = channels.points[n][1]
    
    for m,n in enumerate(lasts):
        channels.cell_data["coords"][m,1,0] = channels.points[n][0]
        channels.cell_data["coords"][m,1,1] = channels.points[n][1]
    
    channels.cell_data["segment_length"] = np.sqrt((channels.cell_data["coords"][:,0,0]-channels.cell_data["coords"][:,1,0])**2+(channels.cell_data["coords"][:,0,1]-channels.cell_data["coords"][:,1,1])**2)

    #make node id a data item for each in case we slice or reorder, add some node info
    p_shape = np.shape(mesh.point_data["hydraulic potential"])
    mesh.point_data["node_id"] = np.reshape(np.arange(0, len(mesh.points)),p_shape)
    channels.point_data["node_id"] = np.reshape(np.arange(0, len(channels.points)),p_shape)
    
    #from here on we work on the channels array as 'active' and the mesh as 'static'
    mesh.point_data["node_type"] = np.zeros(p_shape, dtype = np.int64)
    channels.point_data["node_status"] = np.ones(p_shape, dtype = np.int64)
       
    #copy over the required data from mesh to channels
    channels.point_data['hydraulic potential']= mesh.point_data["hydraulic potential"]
    channels.point_data['bed elevation']= mesh.point_data["zb"]
    channels.point_data['surface elevation']= mesh.point_data["zs"]
    channels.point_data['basal velocity magnitude']= mesh.point_data['basal velocity magnitude']
    channels.point_data['basal tau magnitude']= mesh.point_data['basal tau magnitude']
    channels.point_data['effective pressure']= mesh.point_data['effective pressure']
    channels.point_data['sheet thickness']= mesh.point_data['sheet thickness']

    if variability is not None:
        channels.point_data['hydraulic potential'+variability]= mesh.point_data["hydraulic potential"+variability]
        channels.point_data['bed elevation'+variability]= mesh.point_data["zb"+variability]
        channels.point_data['surface elevation'+variability]= mesh.point_data["zs"+variability]
        channels.point_data['basal velocity magnitude'+variability]= mesh.point_data['basal velocity magnitude'+variability]
        channels.point_data['basal tau magnitude'+variability] = mesh.point_data['basal tau magnitude'+variability]
        
    #here we assign the following node types on the mesh:
    # internal_node == 0 - default where node is inside the domain
    # external_boundary_node == 1 - assigned where node is on the external domain boundary
    external_boundary_nodes = np.unique(mesh.cells_dict['line'].flatten())
    nodearr = np.isin(mesh.point_data["node_id"],[external_boundary_nodes])
    mesh.point_data["node_type"][nodearr] = 1
    # join node == 2 - assigned at duplicate nodes in subdomain joins, 
    # here we label the joins and later we will reassign edges that use the duplicate ids
    # we do not deal with triangles or lines in the mesh
    unq,unq_idx,unq_cnt = np.unique(mesh.points, axis=0,return_index=True,return_counts = True)
    unq_nodes = mesh.point_data["node_id"][unq_idx]
    not_unq = np.isin(mesh.point_data["node_id"],unq_nodes, invert = True)
    mesh.point_data["node_type"][not_unq] = 2

    #make a dict with the replacement ID for not_unq - can we do this without loop?
    NodeTranslate = {}
    for m,n in enumerate(mesh.point_data["node_id"][not_unq]):
            loc = mesh.points[n]
            close = np.isclose(mesh.points,loc, atol = 1)
            IDs=[]
            for i,j in enumerate(close):
                if j[0] and j[1]:
                    IDs.append(mesh.point_data["node_id"][i])
            for p,q in enumerate(IDs):
                NodeTranslate[q]=IDs[0]

    #search for lines that reference duplicated nodes and replace node_id
    cells = channels.cells_dict['line']
    for m, n in enumerate(cells):
        if n[0] in NodeTranslate:
            cells[m][0] = NodeTranslate[n[0]]
        if n[1] in NodeTranslate:
            cells[m][1] = NodeTranslate[n[1]]    
    #new dict
    new_channels={}
    new_channels['line']=cells
    firsts = new_channels['line'][:,0]
    lasts = new_channels['line'][:,1]
    
    #here we identify the status of nodes and segments
    
    # first work with segments
    
    for m,n in enumerate(new_channels['line']):
        begin = np.where(mesh.point_data["node_id"] == n[0])[0]
        end = np.where(mesh.point_data["node_id"]==n[1])[0]
        #get hydraulic potential difference on channel segment
        channels.cell_data["HP_diff"][m] = mesh.point_data["hydraulic potential"][end]-mesh.point_data["hydraulic potential"][begin]
        #and error
        if variability is not None:
            channels.cell_data["HP_diff"+variability][m] = np.sqrt(mesh.point_data["hydraulic potential"+variability][end]**2+mesh.point_data["hydraulic potential"+variability][begin]**2)
        #absolute gradient
        channels.cell_data["HP_grad_abs"][m] = np.abs(channels.cell_data["HP_diff"][m]/channels.cell_data["segment_length"][m])
        if variability is not None:
            channels.cell_data["HP_grad_abs"+variability][m] = np.abs(channels.cell_data["HP_diff"+variability][m]/channels.cell_data["segment_length"][m])
        #identify if entirely floating (flag = -1)
        if mesh.point_data["hydraulic potential"][end]<=0.0 and mesh.point_data["hydraulic potential"][begin]<=0.0:
            channels.cell_data["segment_status"][m] = -1
        #identify if outlet segment (flag = 0)
        elif mesh.point_data["hydraulic potential"][end]<=0.0 or mesh.point_data["hydraulic potential"][begin]<=0.0:
            channels.cell_data["segment_status"][m] = 0
        #identify if edge segment (flag = 3)
        elif mesh.point_data["node_type"][end]==1 and mesh.point_data["node_type"][begin]==1:
            channels.cell_data["segment_status"][m] = 3
        #identify if edge-contacting segments (flag = 2)
        elif mesh.point_data["node_type"][end]==1 or mesh.point_data["node_type"][begin]==1:
            channels.cell_data["segment_status"][m] = 2
        #other segments have flag = 1
        else:
            channels.cell_data["segment_status"][m] = 1
        
    # for all channel segments flag the flow direction and make arrays of up-potential and down-potential nodes
    channels.cell_data["flow_dir"]=np.where(channels.cell_data["HP_diff"]>0,1,0)
    up_nodes = np.where(channels.cell_data["flow_dir"] == 1, lasts,firsts)
    down_nodes = np.where(channels.cell_data["flow_dir"] == 0, lasts,firsts)
    
    #flag 3 - head nodes where all connected nodes are down-potential
    never_down = np.isin(channels.point_data["node_id"],down_nodes,invert = True)
    channels.point_data["node_status"][never_down] = 3
    
    #flag 2 - sink nodes where all connected nodes are up-potential
    never_up = np.isin(channels.point_data["node_id"],up_nodes,invert = True)
    channels.point_data["node_status"][never_up] = 2
    
    #flag 0 - outlet nodes - the up-potential ends of outlet segments
    outsegments = np.where(channels.cell_data["segment_status"] == 0)
    outnodes = up_nodes[outsegments]
    channels.point_data["node_status"][outnodes] = 0
    
    #exclude non-network nodes
        
    #flag = -1 - 'floating' nodes where hydraulic potential is zero
    NGI = np.where(channels.point_data['hydraulic potential']<=0.0)[0]
    channels.point_data["node_status"][NGI] = -1

    #flag = -2 - nodes not connected to any channel segment
    NON = np.isin(channels.point_data["node_id"],new_channels['line'],invert = True)
    channels.point_data["node_status"][NON] = -2

    #now make a network X graph and add the key data
    DG = nx.DiGraph()
    #add edges one by one - perhaps adjacency matrix is faster
    #get weights according to channel volume
    CVol = channels.cell_data["channel area"][0]*channels.cell_data["segment_length"]
    Weights = 1.0-CVol/np.nanmax(CVol)
    #deal with nans as maximum weight
    Weights = np.nan_to_num(Weights,nan=1.0, posinf=1.0, neginf=1.0)
    #but this is more clear - edge-linked nodes will be added by default
    for m,n in enumerate(new_channels['line']):
        w = Weights[m]
        c = channels.cell_data["coords"][m]
        #coords (first,last) may be reversed in network (up,down)
        #this is indicated by flow dir so we include that too
        d = channels.cell_data["flow_dir"][m]
        s = channels.cell_data["segment_status"][m]
        l = channels.cell_data["segment_length"][m]
        cf = channels.cell_data["Channel Flux"][0][m]
        ca = channels.cell_data["channel area"][0][m]
        hpg = channels.cell_data["HP_grad_abs"][m]
        if variability is not None:
            cf_v = channels.cell_data["Channel Flux"+variability][m]
            ca_v = channels.cell_data["channel area"+variability][m]
            hpg_v = channels.cell_data["HP_grad_abs"+variability][m]
        #we add from up to down, giving direction
        if variability is None:
            DG.add_edge(up_nodes[m],down_nodes[m], weight = w, coords = c, status = s, length = l, channel_flux = cf, channel_area = ca, direction = d, hyd_pot_grad = hpg)
        else:
            DG.add_edge(up_nodes[m],down_nodes[m], weight = w, coords = c, status = s, length = l, direction = d, channel_flux = cf, channel_flux_var = cf_v, channel_area = ca, channel_area_var = ca_v, hyd_pot_grad = hpg, hyd_pot_grad_var = hpg_v)
    #add node attributes
    for j,k in enumerate(channels.point_data['node_id']):
        if channels.point_data['node_status'][j] != -2:
            DG.nodes[k]["node_status"]=channels.point_data['node_status'][j]
            DG.nodes[k]["coords"]=channels.points[j]
            DG.nodes[k]["bed_elevation"]=channels.point_data['bed elevation'][j]
            DG.nodes[k]["surface_elevation"]=channels.point_data['surface elevation'][j]
            DG.nodes[k]["hydraulic_potential"]=channels.point_data['hydraulic potential'][j]
            DG.nodes[k]["basal_velocity_magnitude"]=channels.point_data['basal velocity magnitude'][j]
            DG.nodes[k]["basal_tau_magnitude"]=channels.point_data['basal tau magnitude'][j]
            DG.nodes[k]["h_sheet"]=channels.point_data['sheet thickness'][j]
            DG.nodes[k]["effective_pressure"]=channels.point_data['effective pressure'][j]
            if variability is not None:
                DG.nodes[k]["bed_elevation"+variability]=channels.point_data['bed elevation'+variability][j]
                DG.nodes[k]["surface_elevation"+variability]=channels.point_data['surface elevation'+variability][j]
                DG.nodes[k]["hydraulic_potential"+variability]=channels.point_data['hydraulic potential'+variability][j]
                DG.nodes[k]["basal_velocity_magnitude"+variability]=channels.point_data['basal velocity magnitude'+variability][j]
                DG.nodes[k]["basal_tau_magnitude"+variability]=channels.point_data['basal tau magnitude'+variability][j]
    return DG