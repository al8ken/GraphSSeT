#!/usr/bin/env python
"""
Redesigned SubglacialErosionandSedimentFlux functions with pure array operations.
All functions work with numpy/cupy arrays only, no NetworkX graph operations.

Author: Redesigned from original GraphSSeT model
"""

import sys
from typing import Tuple, List, Dict, Optional, Union

# Import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.stats as cpx_stats
    print("CuPy imported successfully. Using GPU acceleration.")
except ImportError:
    print("CuPy not found. Falling back to NumPy (CPU-only).")
    import numpy as cp
    import scipy.stats as cpx_stats

# Import helper functions from main module
# try:
#     from GraphSSeT_V1 import (  # probably we avoid this circularity
#         NewDdist, VelScaled, VelTauScaled, SHREVE_potential,
#         DarcyWeisbach, WaterShearStress, VirtualVelocity_1,
#         GetEdgeVolumes, CombineDdistsArray, MixVols, ExtractDdistsArray
#     )
# except ImportError:
#     print("Warning: Helper functions not available. Some functions may not work.")


def calculate_edge_d_and_rhos(
    n_edges: int,
    d_median: Optional[cp.ndarray] = None,
    d_distribution: Optional[List] = None,
    sed_d_distribution: Optional[List] = None,
    mean_d: float = 2.2,
    std_d: float = 1.5,
    samp_n: int = 100
) -> Tuple[cp.ndarray, List, List]:
    """
    Calculate grain size distributions for edges.
    
    Args:
        n_edges: Number of edges
        d_median: Existing median grain size array (optional)
        d_distribution: Existing grain size distribution list (optional)
        sed_d_distribution: Existing sediment grain size distribution list (optional)
        mean_d: Mean grain size (phi units)
        std_d: Standard deviation of grain size (phi units)
        samp_n: Number of samples for distribution
    
    Returns:
        d_median: median grain size array
        d_dist: grain size distribution list
        sed_d_dist: sediment grain size distribution list
    """
    if d_median is not None:
        result_d_median = d_median.copy()
        if d_distribution is not None:
            result_d_dist = d_distribution.copy()
        else:
            result_d_dist = [(mean_d, std_d) for _ in range(n_edges)]
    else:
        # Generate new distributions
        new_ds = [NewDdist(mean_d, std_d, samp_n) for _ in range(n_edges)]
        result_d_median = cp.array([d[0] for d in new_ds])
        result_d_dist = [d[1] for d in new_ds]
    
    if sed_d_distribution is not None:
        result_sed_d_dist = sed_d_distribution.copy()
    else:
        new_ds = [NewDdist(mean_d, std_d, samp_n) for _ in range(n_edges)]
        result_sed_d_dist = [d[1] for d in new_ds]
    
    return result_d_median, result_d_dist, result_sed_d_dist


def calculate_time_step(
    last_time: cp.ndarray,
    current_time: float
) -> cp.ndarray:
    """
    Calculate time step array for each edge.
    
    Args:
        last_time: Array of last calculation times for each edge
        current_time: Current simulation time
    
    Returns:
        dt: time step array
    """
    return current_time - last_time


def calculate_hydraulic_potential_gradient(
    method: str,
    length: cp.ndarray,
    hyd_pot_grad: Optional[cp.ndarray] = None,
    hydraulic_potential_upstream: Optional[cp.ndarray] = None,
    hydraulic_potential_downstream: Optional[cp.ndarray] = None,
    ice_thickness_upstream: Optional[cp.ndarray] = None,
    ice_thickness_downstream: Optional[cp.ndarray] = None,
    bed_elevation_upstream: Optional[cp.ndarray] = None,
    bed_elevation_downstream: Optional[cp.ndarray] = None,
    surface_elevation_upstream: Optional[cp.ndarray] = None,
    surface_elevation_downstream: Optional[cp.ndarray] = None,
    ice_density: float = 910.0,
    fluid_density: float = 1000.0,
    g: float = 9.81
) -> cp.ndarray:
    """
    Calculate hydraulic potential gradient.
    
    Args:
        method: Calculation method ('Direct', 'Potential', 'IceThickness', 'SurfaceElevation')
        length: Edge length array
        hyd_pot_grad: Direct hydraulic potential gradient (for 'Direct' method)
        hydraulic_potential_upstream/downstream: Hydraulic potential at nodes
        ice_thickness_upstream/downstream: Ice thickness at nodes
        bed_elevation_upstream/downstream: Bed elevation at nodes
        surface_elevation_upstream/downstream: Surface elevation at nodes
        ice_density: Ice density (kg/m³)
        fluid_density: Fluid density (kg/m³)
        g: Gravitational acceleration (m/s²)
    
    Returns:
        DPhi: hydraulic potential gradient array
    """
    if method == "Direct":
        if hyd_pot_grad is None:
            raise ValueError("hyd_pot_grad required for Direct method")
        return hyd_pot_grad.copy()
    
    elif method == "Potential":
        if hydraulic_potential_upstream is None or hydraulic_potential_downstream is None:
            raise ValueError("Hydraulic potential arrays required for Potential method")
        return (hydraulic_potential_upstream - hydraulic_potential_downstream) / length
    
    elif method == "IceThickness":
        if (ice_thickness_upstream is None or ice_thickness_downstream is None or
            bed_elevation_upstream is None or bed_elevation_downstream is None):
            raise ValueError("Ice thickness and bed elevation arrays required for IceThickness method")
        
        # Calculate Shreve potential at upstream and downstream nodes
        shreve_upstream = SHREVE_potential(
            ice_thickness_upstream, bed_elevation_upstream,
            ice_density, fluid_density, g, s_isThick=True
        )
        shreve_downstream = SHREVE_potential(
            ice_thickness_downstream, bed_elevation_downstream,
            ice_density, fluid_density, g, s_isThick=True
        )
        return (shreve_upstream - shreve_downstream) / length
    
    elif method == "SurfaceElevation":
        if (surface_elevation_upstream is None or surface_elevation_downstream is None or
            bed_elevation_upstream is None or bed_elevation_downstream is None):
            raise ValueError("Surface and bed elevation arrays required for SurfaceElevation method")
        
        # Calculate Shreve potential at upstream and downstream nodes
        shreve_upstream = SHREVE_potential(
            surface_elevation_upstream, bed_elevation_upstream,
            ice_density, fluid_density, g, s_isThick=False
        )
        shreve_downstream = SHREVE_potential(
            surface_elevation_downstream, bed_elevation_downstream,
            ice_density, fluid_density, g, s_isThick=False
        )
        return (shreve_upstream - shreve_downstream) / length
    
    else:
        raise ValueError(f"Unknown potential gradient method: {method}")


def calculate_erosion_rate(
    method: str,
    length: cp.ndarray,
    erosion_rate_upstream: Optional[cp.ndarray] = None,
    erosion_rate_downstream: Optional[cp.ndarray] = None,
    basal_velocity_upstream: Optional[cp.ndarray] = None,
    basal_velocity_downstream: Optional[cp.ndarray] = None,
    basal_tau_upstream: Optional[cp.ndarray] = None,
    basal_tau_downstream: Optional[cp.ndarray] = None,
    k: float = 1e-4,
    l: float = 1,
    w: float = 2e-10
) -> cp.ndarray:
    """
    Calculate erosion rate for edges.
    
    Args:
        method: Erosion calculation method
        length: Edge length array
        erosion_rate_upstream/downstream: Erosion rates at nodes
        basal_velocity_upstream/downstream: Basal velocity at nodes
        basal_tau_upstream/downstream: Basal shear stress at nodes
        k, l, w: Erosion law parameters
    
    Returns:
        erod: erosion rate array
    """
    if method == "InfiniteTill":
        return cp.zeros_like(length)
    
    elif method == "Direct":
        if erosion_rate_upstream is None or erosion_rate_downstream is None:
            raise ValueError("Erosion rate arrays required for Direct method")
        return (erosion_rate_upstream + erosion_rate_downstream) / 2
    
    elif method == "Vel":
        if basal_velocity_upstream is None or basal_velocity_downstream is None:
            raise ValueError("Basal velocity arrays required for Vel method")
        v = (basal_velocity_upstream + basal_velocity_downstream) / 2
        return VelScaled(v, k, l, units='m s-1')
    
    elif method == "VelTau":
        if (basal_velocity_upstream is None or basal_velocity_downstream is None or
            basal_tau_upstream is None or basal_tau_downstream is None):
            raise ValueError("Basal velocity and tau arrays required for VelTau method")
        v = (basal_velocity_upstream + basal_velocity_downstream) / 2
        tau = (basal_tau_upstream + basal_tau_downstream) / 2
        return VelTauScaled(tau, v, w)
    
    elif method == "MixedBed":
        if basal_velocity_upstream is None or basal_velocity_downstream is None:
            raise ValueError("Basal velocity arrays required for MixedBed method")
        v = (basal_velocity_upstream + basal_velocity_downstream) / 2
        return VelScaled(v, k, l, units='m s-1')
    
    else:
        raise ValueError(f"Unknown erosion method: {method}")


def calculate_channel_flux_on_edge(
    method: str,
    length: cp.ndarray,
    channel_flux: cp.ndarray,
    qw_in: cp.ndarray,
    edge_input_flux: Optional[cp.ndarray] = None,
    node_input_flux_weighted: Optional[cp.ndarray] = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Calculate channel flux on edges.
    
    Args:
        method: Flux calculation method
        length: Edge length array
        channel_flux: Current channel flux array
        qw_in: Water input flux array
        edge_input_flux: Input flux along edges
        node_input_flux_weighted: Weighted node input flux
    
    Returns:
        cflux: channel flux array
        outflux: output flux array
    """
    if method in ["Flux", "FluxArea"]:
        cflux = channel_flux.copy()
        outflux = channel_flux.copy()
    
    elif method == "InputEdge":
        if edge_input_flux is None:
            raise ValueError("edge_input_flux required for InputEdge method")
        cflux = qw_in + edge_input_flux * length / 2
        outflux = channel_flux + edge_input_flux * length / 2
    
    elif method == "InputNode":
        if node_input_flux_weighted is None:
            raise ValueError("node_input_flux_weighted required for InputNode method")
        cflux = qw_in + node_input_flux_weighted
        outflux = channel_flux.copy()
    
    elif method == "InputBoth":
        if edge_input_flux is None or node_input_flux_weighted is None:
            raise ValueError("Both edge and node input flux required for InputBoth method")
        cflux = qw_in + edge_input_flux * length / 2 + node_input_flux_weighted
        outflux = channel_flux + edge_input_flux * length / 2
    
    else:
        raise ValueError(f"Unknown channel flux method: {method}")
    
    return cflux, outflux


def calculate_transport_capacity_eh(
    cflux: cp.ndarray,
    qw_in: cp.ndarray,
    dphi: cp.ndarray,
    d_median: cp.ndarray,
    length: cp.ndarray,
    channel_area: Optional[cp.ndarray] = None,
    beta: float = cp.pi,
    fr: float = 0.15,
    fluid_density: float = 1000.0,
    rhos_active: float = 2650.0,
    dhmin: float = 0.21,
    g: float = 9.81,
    mean_d: float = 2.2
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Calculate sediment transport capacity using Engelund-Hansen.
    
    Args:
        cflux: Channel flux array
        qw_in: Water input flux array
        dphi: Hydraulic potential gradient array
        d_median: Median grain size array
        length: Edge length array
        channel_area: Channel area array (optional)
        beta: Hooke angle (radians)
        fr: Darcy-Weisbach friction factor
        fluid_density: Fluid density (kg/m³)
        rhos_active: Active sediment density (kg/m³)
        dhmin: Minimum hydraulic diameter (m)
        g: Gravitational acceleration (m/s²)
        mean_d: Mean grain size (phi units)
    
    Returns:
        QSc: transport capacity array
        Uv: virtual velocity array
    """
    # Total flux
    flux = cflux + qw_in
    
    # Get hydraulic parameters - channel area and width
    if channel_area is not None:
        S = channel_area.copy()
        Wc = 2 * cp.sin(beta/2) * cp.sqrt(2*S/(beta-cp.sin(beta)))
    else:
        Dh, S, Wc = DarcyWeisbach(dphi, flux, beta=beta, fr=fr, rw=fluid_density, dhmin=dhmin)
    
    # Calculate basal shear stress
    Tau = WaterShearStress(cflux, S, fr=fr, rw=fluid_density)
    
    # Effective density of sediment in water
    R_mean_active = (rhos_active - fluid_density) / fluid_density
    
    # Get sediment flux capacity
    QSc = (0.4/fr * 1/(d_median*R_mean_active**2*g**2) * 
           (Tau/fluid_density)**(5.0/2.0) * Wc)
    
    # Handle NaN values
    min_qsc = cp.nanmin(QSc)
    QSc = cp.where(cp.isnan(QSc), min_qsc, QSc)
    
    # Calculate virtual velocity
    Uv = VirtualVelocity_1(Tau, d_median, rhos_active, 
                          rw=fluid_density, D50=(2**-mean_d)/1000, g=g)
    
    return QSc, Uv


def calculate_till_mobilisation(
    method: str,
    length: cp.ndarray,
    edgewidth: cp.ndarray,
    dt: cp.ndarray,
    qsc: cp.ndarray,
    qs_in: cp.ndarray,
    till_thickness: cp.ndarray,
    erod: cp.ndarray,
    sedl_factor: float = 1.0,
    bed_porosity: float = 0.3,
    h_lim: float = 1.0,
    hg: float = 0.75,
    dsig: float = 1e-3
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Calculate till mobilisation.
    
    Args:
        method: Erosion method
        length: Edge length array
        edgewidth: Edge width array
        dt: Time step array
        qsc: Transport capacity array
        qs_in: Sediment input flux array
        till_thickness: Till thickness array
        erod: Erosion rate array
        sedl_factor: Sediment uptake length factor
        bed_porosity: Bed porosity
        h_lim: Maximum till thickness
        hg: Hero delimitation thickness
        dsig: Sigma parameter for till mobilisation
    
    Returns:
        dQSdx: sediment flux gradient array
        mt: erosion source term array
        till_thickness: updated till thickness array
    """
    sedl = length * sedl_factor
    
    # Till mobilisation for transport limited case
    mob = (qsc - qs_in) / sedl
    
    # Sediment needed to reach maximum possible till deposition
    if method == "MixedBed":
        min_mob = -h_lim * (1 - bed_porosity) * edgewidth / dt
    else:
        min_mob = (till_thickness - h_lim) * (1 - bed_porosity) * edgewidth / dt
    
    # Don't over-fill the edge
    mob_r = cp.where(mob < min_mob, min_mob, mob)
    
    if method == "InfiniteTill":
        dQSdx = mob_r.copy()
        mt = cp.zeros_like(erod)
    else:
        # Sigma function (corrected from Delaney et al 2019)
        sigmaH = (1 + cp.exp(10 - 5*till_thickness/dsig))**-1
        
        # Till source term from bedrock erosion
        mt = erod * (1 - (till_thickness/hg)) * edgewidth
        mt = cp.where(mt > 0, mt, 0.0)
        
        # Maximum till available to mobilise
        max_mob = mt + till_thickness * (1 - bed_porosity) * edgewidth / dt
        
        # Channel can only mobilise existing till
        mob_r = cp.where(mob_r > max_mob, max_mob, mob_r)
        
        # Mobilisation for mixed supply/transport limited case
        mob2 = mob_r * sigmaH + mt * (1 - sigmaH)
        
        # Select which rule to apply
        conA = cp.logical_and(mob_r <= 0, till_thickness >= h_lim)
        conB = cp.logical_and(conA == False, mob_r <= mt)
        conC = cp.logical_and(conA == False, conB == False)
        
        # Sediment flux gradient on edge
        dQSdx = (conA * cp.zeros_like(length) + 
                conB * mob_r + 
                conC * mob2)
    
    return dQSdx, mt, till_thickness

def calculate_till_transport(
    flux_density: cp.ndarray,
    length: cp.ndarray,
    uv: cp.ndarray,
    qsc: cp.ndarray,
    qs_in: cp.ndarray,
    dqsdx: cp.ndarray,
    mt: cp.ndarray,
    vt: cp.ndarray,
    dt: cp.ndarray,
    d_dist: List,
    sed_d_dist: List,
    dod_d_dist: Dict,
    dod_sed_d_dist: Dict,
    G_adj: cp.ndarray,
    node_status: cp.ndarray,
    edge_keys: List,
    node_keys: List,
    rn_array: cp.ndarray,
    samp_n: int = 100,
    mean_d: float = 2.2,
    std_d: float = 1.5
) -> Tuple[List, List, List]:
    """
    Calculate till transport using kinematic wave approach.
    
    Args:
        flux_density: Current flux density array
        length: Edge length array
        uv: Virtual velocity array
        qsc: Transport capacity array
        qs_in: Sediment input flux array
        dqsdx: Sediment flux gradient array
        mt: Erosion source term array
        dt: Time step array
        d_dist: Grain size distribution list
        sed_d_dist: Sediment grain size distribution list
        VSout_adj: Volume output adjacency matrix
        VScap_adj: Volume capacity adjacency matrix
        jammed_adj: Jammed status adjacency matrix
        node_status: Node status array (0 for outlets)
        edge_keys: Edge key list
        node_keys: Node key list
        rn_array: Random number array
        samp_n: Number of samples
        mean_d: Mean grain size
        std_d: Standard deviation of grain size
    
    Returns:
        flux_density: updated flux density array
        d_median: updated median grain size array
        d_dist: updated grain size distribution list
        volume_arrays: volume calculation arrays
    """
    # Minimum distance to avoid gridlock
    lmin = length * 0.01
    umin = lmin / dt
    
    u = cp.where(uv > umin, uv, umin)
    l = u * dt
    
    # Critical distance for material exit
    xcrit = cp.where(l > length, length, l)
    
    # Maximum flux density
    kmax = qsc * dt / length
    vs_cap = cp.where(cp.isfinite(qsc),qsc*dt,0.0)
    
    # Jam conditions
    jammed = cp.greater(flux_density, kmax)
    constricted = cp.greater(qs_in, qsc)
    nonfree = cp.logical_or(jammed, constricted)
    
    # Excess flow handling
    XQcon = cp.where(constricted, qs_in - qsc, 0.0)
    XQ = cp.where(jammed, qs_in, XQcon)
    
    # Only allow deposition in non-free edges
    deposition_only = cp.where(dqsdx < 0.0, dqsdx, 0.0)
    dqsdx_modified = cp.where(nonfree, deposition_only, dqsdx)
    
    # Get volume elements
    vol_arrays = GetEdgeVolumes(flux_density, kmax, qs_in, XQ, 
                               dqsdx_modified, length, xcrit, mt, dt)
    vs_out = vol_arrays[5]
    
    ## Node volume calculations using adjacency matrices ##
    #for succs we must apply numpy functions along the rows: sum_succs = np.sum(adj[[i],:])
    #for preds we can apply numpy functions along the cols: sum_preds = np.sum(adj[:,[i]])
    #we can place edge values into the adj matrix as adj[u,v] where u, v are the keys
    vs_out_adj = G_adj.copy()
    is_pred_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # downstream node index
        vs_out_adj[u_idx][v_idx] = vs_out[i]  #volumes
        is_pred_adj[u_idx][v_idx] = 1 # is an actual pred
        
    # Calculate volume to and from nodes
    vs_to_node = cp.sum(vs_out_adj, axis=0) # sum across all predecessors
    #very hacky here from GPT but see if it works first
    #denom = cp.where(vs_to_node > 0, vs_to_node, 1)
    #P_vs_to_node = vs_out_adj / denom
    #node_index_map = {node: idx for idx, node in enumerate(node_keys)}
    P_vs_to_node = [vs_out_adj[:, i][is_pred_adj[:, i]==1]/vs_to_node[i] for i,j in enumerate(node_keys)]
    
    # combine incoming sediment distributions
    d_dist_to_node = [CombineDdistsArray(dod_d_dist[j], #here as a dict tied to node key
                                       #P_vs_to_node[[node_index_map[p] for p in dod_d_dist[j].keys()], i], # the predeccesor volume values
                                       P_vs_to_node[i], 
                                       RArray = rn_array[i], #a slice of the RNA
                                       def_mean = mean_d, 
                                       def_std = std_d)[1] for i,j in enumerate(node_keys)]
    
    vsc_from_node = cp.zeros_like(vs_to_node)
    # Calculate capacity from nodes (considering jammed edges)    
    jammed_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # upstream node index
        jammed_adj[u_idx][v_idx] = jammed[i]    
 
    vs_cap_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # upstream node index
        vs_cap_adj[u_idx][v_idx] = vs_cap[i]    

    for i in range(len(node_keys)):
        # Sum capacity of non-jammed successors
        succ_capacities = vs_cap_adj[i, :] * (1 - jammed_adj[i, :])  # zero output to jammed edges
        vsc_from_node[i] = cp.sum(succ_capacities)
    
    # Calculate backflow and outflow
    Backflow = cp.where(vs_to_node > vsc_from_node, vs_to_node - vsc_from_node, 0.0)
    # No backflow for outlet nodes
    Backflow = cp.where(node_status == 0, 0.0, Backflow)
    Outflow = vs_to_node - Backflow
    
    # Calculate backflow ratios and volume split ratios
    BFR = cp.where(vs_to_node > 0, Backflow / vs_to_node, 0.0)
    VSR = cp.where(vsc_from_node > 0, Outflow / vsc_from_node, 0.0)
    
    # Update volume flows using adjacency matrices
    # Calculate VSback for each edge (backflow to predecessor edges)
    vs_back = cp.zeros_like(vs_out)
    for i, (u, v) in enumerate(edge_keys):
        u_idx = node_keys.index(v)  # downstream node index
        vs_back[i] = vs_out[i] * BFR[u_idx]
    
    # Update VSout (adjusted outflow from edges)
    vs_out -= vs_back
    
    # Calculate inflow to successor edges for next iteration
    in_vs = cp.zeros_like(vs_out)
    for i, (u, v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        # Only to unjammed edges
        if not jammed[i]:
            in_vs[i] = vs_cap_adj[u_idx, node_keys.index(v)] * VSR[u_idx]
    
    QSin = in_vs/dt
    
    # Update arrays
    vol_arrays[5] = vs_out
    vol_arrays[6] = vs_back
    
    # Update VSfinal and flux density at end of timestep
    vol_arrays[7] += vs_back
    flux_density = vol_arrays[7]/length
    
    # Update grain size distributions
    # new samples of global distribution with sample size n_samp
    newDs = [NewDdist(mean_d, std_d, samp_n)[1] for i in edge_keys]
    
    #input distributions from upstream nodes
    d_dist_in = [d_dist_to_node[node_keys.index(u)] for u,v in edge_keys] # Access d_dists by node key index
    
    #volumetric proportions defind using function MixVols
    PVolArrays = MixVols(vol_arrays)
    P_as_edge = PVolArrays[0]
    P_as_node = PVolArrays[1]
    P_basal = PVolArrays[2]
    P_basement = PVolArrays[3]
    
    #volume proportions for each volume element
    volPs = [[P_as_edge[i],P_as_node[i],P_basal[i],P_basement[i]] for i,j in enumerate(edge_keys)]
    
    #dist for each volume element
    try:
        dists = [[d_dist[i],d_dist_in[i],sed_d_dist[i], newDs[i]] for i,j in enumerate(edge_keys)]
    except IndexError:
        raise("Index error accessing arrays")
    #get new edge distributions with function CombineDdists
    edge_dists = [CombineDdistsArray(dists[i], #lists here
                                     volPs[i], 
                                     RArray = rn_array[i],
                                     def_mean = mean_d, 
                                     def_std = std_d) for i,j in enumerate(edge_keys)]
        
    #get new grainsize distribution for the basal sediment layer
    
    # volume deposited/mobilised - one should be zero
    vs_dep = vol_arrays[2]
    vs_mob = vol_arrays[4]

    # as a proportion
    Pdep = cp.where(cp.isfinite(vs_dep/(vt+vs_mob+vs_dep)),vs_dep/(vt+vs_mob+vs_dep),0.0)
    Pmob = cp.where(cp.isfinite(vs_mob/(vt+vs_mob+vs_dep)),vs_mob/(vt+vs_mob+vs_dep),0.0)
    Pbas = cp.where(cp.isfinite(vt/(vt+vs_mob+vs_dep)),vt/(vt+vs_mob+vs_dep),0.0)
    
    #first we need to add VSdep to VT
    volPs = [[Pdep[i],Pbas[i]] for i,j in enumerate(edge_keys)]
    dists = [[edge_dists[i][1], sed_d_dist[i]] for i,j in enumerate(edge_keys)]
    
    basal_dists = [CombineDdistsArray(dists[i],
                                      volPs[i],
                                      RArray = rn_array[i],
                                      def_mean = mean_d, 
                                      def_std = std_d) for i,j in enumerate(edge_keys)]
    
    #then we need to remove VSmob from the combination
    volPs = [[Pbas[i]+Pdep[i],Pmob[i]] for i,j in enumerate(edge_keys)]
    dists = [[basal_dists[i][1],dists[i][0]] for i,j in enumerate(edge_keys)]
    basal_dists = [ExtractDdistsArray(dists[i],
                                      volPs[i],
                                      RArray = rn_array[i],
                                      def_mean = mean_d, 
                                      def_std = std_d) for i,j in enumerate(edge_keys)]
    
    #report to graph >>> what do we return instead?
    #edge
    d_median = cp.array([edge_dists[i][0] for i in range(len(edge_keys))])
    d_dist = [edge_dists[i][1] for i in range(len(edge_keys))]
    #basal
    sed_d_median = cp.array([basal_dists[i][0] for i in range(len(edge_keys))])
    sed_d_dist = [basal_dists[i][1] for i in range(len(edge_keys))]
    
    G_props = [jammed, Outflow, QSin, flux_density, d_dist_to_node, d_median, d_dist, sed_d_median, sed_d_dist]
        
    return G_props, vol_arrays, PVolArrays

def calculate_till_transport_lite(
    flux_density: cp.ndarray,
    length: cp.ndarray,
    uv: cp.ndarray,
    qsc: cp.ndarray,
    qs_in: cp.ndarray,
    dqsdx: cp.ndarray,
    dt: cp.ndarray,
    node_connectivity: Dict,
    edge_keys: List,
    node_keys: List
) -> Tuple[cp.ndarray, List]:
    """
    Calculate till transport (simplified version).
    
    Args:
        flux_density: Current flux density array
        length: Edge length array
        uv: Virtual velocity array
        qsc: Transport capacity array
        qs_in: Sediment input flux array
        dqsdx: Sediment flux gradient array
        dt: Time step array
        node_connectivity: Node connectivity information
        edge_keys: Edge key list
        node_keys: Node key list
    
    Returns:
        flux_density: updated flux density array
        volume_arrays: volume calculation arrays
    """
    # Similar to full transport but without grain size tracking
    lmin = length * 0.01
    umin = lmin / dt
    
    u = cp.where(uv > umin, uv, umin)
    l = u * dt
    xcrit = cp.where(l > length, length, l)
    
    kmax = qsc * dt / length
    jammed = cp.greater(flux_density, kmax)
    constricted = cp.greater(qs_in, qsc)
    nonfree = cp.logical_or(jammed, constricted)
    
    XQcon = cp.where(constricted, qs_in - qsc, 0.0)
    XQ = cp.where(jammed, qs_in, XQcon)
    
    deposition_only = cp.where(dqsdx < 0.0, dqsdx, 0.0)
    dqsdx_modified = cp.where(nonfree, deposition_only, dqsdx)
    
    # Simplified volume calculation
    vol_arrays = GetEdgeVolumes(flux_density, kmax, qs_in, XQ, 
                               dqsdx_modified, length, xcrit, 
                               cp.zeros_like(length), dt)
    
    VSfinal = vol_arrays[7]
    new_flux_density = VSfinal / length
    
    return new_flux_density, vol_arrays


def calculate_exner_equation(
    dqsdx: cp.ndarray,
    mt: cp.ndarray,
    length: cp.ndarray,
    edgewidth: cp.ndarray,
    dt: cp.ndarray,
    till_thickness: cp.ndarray,
    bed_elevation: cp.ndarray,
    bedrock_elevation: cp.ndarray,
    G_adj:cp.ndarray,
    edge_keys: List,
    node_keys: List,
    bed_porosity: float = 0.3,
    vol_arrays: Optional[List] = None
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Calculate evolving till thickness using Exner equation.
    
    Args:
        dqsdx: Sediment flux gradient array
        mt: Erosion source term array
        length: Edge length array
        edgewidth: Edge width array
        dt: Time step array
        till_thickness: Current till thickness array
        bed_elevation: Current bed elevation array
        bedrock_elevation: Current bedrock elevation array
        node_connectivity: Node connectivity information
        edge_keys: Edge key list
        node_keys: Node key list
        bed_porosity: Bed porosity
        vol_arrays: Volume arrays from transport calculation
    
    Returns:
        till_thickness: updated till thickness array
        bed_elevation: updated bed elevation array
        bedrock_elevation: updated bedrock elevation array
    """
    # Calculate till thickness change rate
    dHdt = (-dqsdx/(1-bed_porosity) + mt/(1-bed_porosity)) / edgewidth
    
    # Excess volume as height (if available)
    if vol_arrays is not None and len(vol_arrays) > 8:
        XVS = vol_arrays[8]
        XVasH = XVS / (length * edgewidth)
    else:
        XVasH = cp.zeros_like(till_thickness)
    
    # Bedrock lowering from erosion
    dBRE = -mt * dt / edgewidth
    
    dBRE_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # upstream node index
        dBRE_adj[u_idx][v_idx] = dBRE[i]    
 
    
    # New till thickness
    NewTill = dt * dHdt + XVasH / (1 - bed_porosity)
    till_thickness += NewTill
    
    TT_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # upstream node index
        TT_adj[u_idx][v_idx] = till_thickness[i]    
    
    # Node averaging for bed elevation updates
    NTT = cp.zeros(len(node_keys))
    NBE = cp.zeros(len(node_keys))
    
    for i in range(len(node_keys)):
        # Get predecessor values (sum along columns)
        NTT[i] = cp.mean(TT_adj[:, i])  # mean till thickness from predecessors
        NBE[i] = cp.mean(dBRE_adj[:, i])  # bedrock change from erosion from predecessors
    
    # Update bedrock elevation with erosion
    bedrock_elevation += NBE
    
    # Set new bed elevation based on till thickness
    # For positive NTT: bed elevation = bedrock + till
    # For negative NTT: bedrock elevation = bedrock + till (lowering)
    bed_elevation = cp.where(NTT >= 0.0, 
                                bedrock_elevation + NTT, 
                                bedrock_elevation)
    bedrock_elevation = cp.where(NTT < 0.0, 
                                    bedrock_elevation + NTT, 
                                    bedrock_elevation)
    
    # Ensure non-negative till thickness
    till_thickness = cp.where(till_thickness < 0, 0.0, till_thickness)        
    
    return dHdt, till_thickness, bed_elevation, bedrock_elevation


def calculate_channel_flux_to_node(
    QW_adj:cp.ndarray,
    LT_adj:cp.ndarray,
    EW_adj:cp.ndarray,
    current_time: float,
    edge_weight:cp.ndarray,
    cflux: cp.ndarray,
    edge_keys: List,
    node_keys: List,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Calculate flux into nodes from upstream edges.
    
    Args:
        QW_adj: Water output flux from edges (adjacency matrix)
        LT_adj: Last calculation time for edges (adjacency matrix)
        EW_adj: Edge weights for edges (adjacency matrix)
        current_time: Current simulation time
        cflux: Channel flux array
        edge_keys: Edge key list
        node_keys: Node key list
    
    Returns:
        qw_node: water flux at nodes array
        vw_node: water volume at nodes array
        qw_in: water input to edges array
    """
    # Calculate time step size
    dt_adj = cp.where(current_time - LT_adj > 0,current_time - LT_adj, 1)
    v_adj = QW_adj*dt_adj
    
    ## working with the adjacency matrix ##
    #for succs we must apply numpy functions along the rows: sum_succs = np.sum(adj[[i],:])
    #for preds we can apply numpy functions along the cols: sum_preds = np.sum(adj[:,[i]])
    
    # Accumulate flux at nodes
    qw_node_dict = {j:cp.sum(QW_adj[:,i]) for i,j in enumerate(node_keys)} #sum QW across preds as a dict
    qw_node = [qw_node_dict[key] for key in qw_node_dict] #array for output
    vw_node = [cp.sum(v_adj[:,i]) for i,j in enumerate(node_keys)] #sum QW across preds
    
    # Distribute flux to downstream edges
    ew_succs = {j:cp.sum(1-EW_adj[i,:]) for i,j in enumerate(node_keys)} #summed weights of all succs as a dict
    flux_node = [qw_node_dict[u] * edge_weight[i]/ew_succs[u] for i,(u,v) in enumerate(edge_keys)]
    qw_influx = flux_node-cflux
    return qw_node, vw_node, qw_influx


def calculate_detritus_tracking(
    method: str,
    p_vol_arrays: List,
    detritus_edge: Dict,
    det_dod: Dict,
    G_adj: cp.ndarray,
    detritus_node: Dict,
    sed_detritus: Dict,
    detritus_classes: List,
    edge_keys: List,
    node_keys: List,
    Pdep: cp.ndarray,
    VSout: List,
    detritus_prop: Optional[Dict] = None,
) -> Tuple[List, List, List]:
    """
    Track detritus as passive tracer properties.
    
    Args:
        method: Detritus tracking method
        p_vol_arrays: Volume proportion arrays
        detritus_edge: Current edge detritus list
        detritus_node: Current node detritus list
        sed_detritus: Current sediment detritus list
        detritus_classes: Detritus classification list
        edge_keys: Edge key list
        node_keys: Node key list
        detritus_prop: Detritus properties at nodes
    
    Returns:
        detritus_edge: updated edge detritus list
        detritus_node: updated node detritus list
        sed_detritus: updated sediment detritus list
    """
    if method == "None":
        return detritus_edge, detritus_node, sed_detritus
    
    # Extract volume proportions
    P_as_edge = p_vol_arrays[0]
    P_as_node = p_vol_arrays[1]
    P_basal = p_vol_arrays[2]
    P_basement = p_vol_arrays[3]
    
    #total proportion should be 1
    TP = P_as_edge+P_as_node+P_basal+P_basement
    
    det_ps={}
    if method == "SedErod":
        # Separate mobilisation of till vs fresh erosion
        for i, key in enumerate(edge_keys):
                if cp.abs(1-TP[i]) > 0.01:
                    print('WARNING: edge {} has total probability {} not adding to 1, using init, but check for NANs or negative volumes'.format(key,TP[i]))
                    ps = cp.array([1,0,0])
                else:
                    Basal = cp.array([0,1,0])* P_basal[i]  # mobilised from the bed
                    Basement = cp.array([0,0,1])* P_basement[i]
                    try:
                        As_Edge = cp.array([detritus_edge[key][Dclass] for Dclass in detritus_classes])*P_as_edge[i]
                        As_Node = cp.array([detritus_node[key[0]][Dclass] for Dclass in detritus_classes])*P_as_node[i]
                    except KeyError:
                        #here we consider missing keys as 'init' because we don't want to keep the details
                        As_Edge = cp.array([1,0,0]) * P_as_edge[i]
                        As_Node =  cp.array([1,0,0]) * P_as_node[i]
                    ps = Basal + Basement + As_Edge + As_Node
                det_ps[key]= {detritus_classes[0]:ps[0],detritus_classes[1]:ps[1],detritus_classes[2]:ps[2]}
 
    elif method == "NodeProp":
        # work out detritus on the edge
        for i,key in enumerate(edge_keys):
            #add any unrepresented classes to Dclasses
            classlist = list(sed_detritus[key].keys())
            newclasses = [item for item in classlist if item not in detritus_classes]
            if len(newclasses) > 0:
                detritus_classes += newclasses
                #print('added Det classes from basal {}'.format(newclasses))
            classlist = list(detritus_edge[key].keys())
            newclasses = [item for item in classlist if item not in detritus_classes]
            if len(newclasses) > 0:
                detritus_classes += newclasses
                #print('added Det classes from edge {}'.format(Dclasses))
            classlist = list(detritus_node[key[0]].keys())
            newclasses = [item for item in classlist if item not in detritus_classes]
            if len(newclasses) > 0:
                detritus_classes += newclasses
                #print('added Det classes from node {}'.format(newclasses))
            #get detritus prop at upstream and downstream nodes
            DPu, DPv = detritus_prop[key[0]],detritus_prop[key[1]] 
            #are these the same?
            if DPu == DPv:
                l = cp.array([1 if b == DPu else 0 for b in detritus_classes])
                Basement = l * P_basement[i]
            else:
                l = cp.array([0.5 if b == DPu or b == DPv else 0 for b in detritus_classes])
                Basement = l * P_basement[i]
            try: #sediment remobilisation uses the proportions already in the till layer
                Basal = cp.array([sed_detritus[key][Dclass] for Dclass in detritus_classes])*P_basal[i]
            except KeyError:
                Basal = cp.array([1 if b == 'basal' else 0 for b in detritus_classes])*P_basal[i]
            try:
                As_Edge = cp.array([detritus_edge[key][Dclass] for Dclass in detritus_classes])*P_as_edge[i]
            except KeyError:
                As_Edge = cp.array([1 if b == 'init' else 0 for b in detritus_classes])*P_as_edge[i]
            try:
                As_Node = cp.array([detritus_node[key[0]][Dclass] for Dclass in detritus_classes])*P_as_node[i]   
            except KeyError:
                As_Node =  cp.array([1 if b == 'init' else 0 for b in detritus_classes])*P_as_node[i]
            ps = Basal + Basement + As_Edge + As_Node
            #new local dict for data
            det_ps[key] = {n: ps[m] for m,n in enumerate(detritus_classes)}
    
    sed_det_ps = {}   
    #detritus of basal sediment layer            
    for i,key in enumerate(edge_keys):
        sed_det_ps[key] = {}
        ToSed = cp.array([det_ps[key][Dclass] for Dclass in detritus_classes])*Pdep[i]
        try:
            Sed = cp.array([sed_detritus[key][Dclass] for Dclass in detritus_classes])*(1-Pdep[i])
        except KeyError:
            Sed = cp.array([1 if b == 'basal' else 0 for b in detritus_classes])*(1-Pdep[i])
        sedps = ToSed+Sed
        for k,l in enumerate(detritus_classes):
            sed_det_ps[key][l] = sedps[k]
    
    #now we need to assign values for outgoing sediment to the nodes        
    
    # Volumes from preceding edges        
    VSo_adj = G_adj.copy()
    for i,(u,v) in enumerate(edge_keys):
        u_idx = node_keys.index(u)  # upstream node index
        v_idx = node_keys.index(v)  # downstream node index
        VSo_adj[u_idx][v_idx] = VSout[i]
    
    #update det to node with new values
    det_node = MixDetritusbyVolume(det_dod, #original values as a dict of dicts
                                      VSo_adj, # the predeccessor volume values
                                      det_ps, #detritus on edges
                                      node_keys, #node keys
                                      )    
    return det_ps, sed_det_ps, det_node, 

#### Helper Methods called in the preceding methods ####   
def SHREVE_potential(s,b,ri=920.0,rw=1000.0,g=9.81,k=1.0, s_isThick = False):
    """method to calculate hydraulic potential in Pa with equation of Shreve"""
    if k > 1.0:
        msg = "k must be between 0 and 1 setting to 1"
        k=1.0
        print(msg)
    if k < 0.0:
        msg = "k must be between 0 and 1 setting to 0"
        k=0.0
        print(msg)
    if s_isThick:
        Phi = k*ri*g*(s) + rw*g*b
    else:
        Phi = k*ri*g*(s-b) + rw*g*b
    return Phi

def DarcyWeisbach(DPhi,Qw,beta=cp.pi/6,fr=0.015,rw=1000.0,dhmin=0.0):
    """method to calculate channel geometries with Darcy-Weisbach equation"""
    #Darcy-Weisbach formula factor
    s = 2*(beta-cp.sin(beta))/(beta/2+cp.sin(beta/2))**2
    P = s*fr*rw
    #hydraulic diameter
    Dh = (P*Qw**2/cp.abs(DPhi))**0.2
    Dh = cp.where(Dh < dhmin,dhmin,Dh)
    #channel section area
    S = Dh**2/2*(beta/2+cp.sin(beta/2))**2/(beta-cp.sin(beta))
    #channel width
    wc = 2*cp.sin(beta/2)*cp.sqrt(2*S/(beta-cp.sin(beta)))
    return Dh,S,wc

def WaterShearStress(Qw,S,fr=0.015,rw=1000.0):
    """method to calculate basal shear stress from water flow"""
    Uw = Qw/S #water velocity
    Tauw=1/8*fr*rw*Uw**2 #basal shear stress
    return Tauw

def VirtualVelocity_1(Tau,d,rs,rw=1000.0, a = 2.30, D50 = 0.11, g = 9.81, Vmin = 0.0):
    """method to calculate virtual velocity using equation 14 from Kloesch and Habersack 2018"""
    Scale = (rs-rw)*g*d
    Tau_star = Tau/Scale
    Tau_c_star = 0.052*(d/D50)**-0.82
    Vu = cp.where(Tau_star >= Tau_c_star,a*((rs-rw)*g*d/rw)**0.5*(Tau_star-Tau_c_star)*((Tau_star)**0.5-(Tau_c_star)**0.5),Vmin)
    return Vu

def VirtualVelocity_2(Tau,d,rs,rw=1000.0, a = 0.96, b =1.5, D50 = 0.11, g = 9.81, Vmin = 0.0):
    """method to calculate virtual velocity using equation 17 from Kloesch and Habersack 2018"""
    Scale = (rs-rw)*g*d
    Tau_star = Tau/Scale
    Tau_c_star = 0.055*(d/D50)**-0.83
    Vu = cp.where(Tau_star >= Tau_c_star,a*((rs-rw)*g*d/rw)**0.5*(Tau_star-Tau_c_star)**b,Vmin)
    return Vu

def VelScaled(ub,k,l,units = 'm a-1'):
    """method to calculate erosion rate from basal velocity"""
    y2s = 31556926.0
    if units =='m s-1':
        ub = ub*y2s
        e = k*ub**l/y2s #m s-1
    elif units =='m a-1':
        e = k*ub**l #m/a
    else:
        e = k*ub**l #m/a
        print('units {} are not m a-1 or m s-1, applied erosion assuming m a-1.'.format(units))
    return e

def VelTauScaled(tb,ub,w):
    """method to calculate erosion rate from basal velocity and Tau"""
    e = w*tb*ub #ms-1
    return e

# volume balance of numerous components for mixing calculations
def GetEdgeVolumes(k, kmax, QSin, XQ, dQSdx, length, xcrit, mt, dt):
    """method to calculate volumme components on edges"""
    #At beginning of timestep
    VSinit = k*length #sediment remaining on the edge from last timestep
    VSin = (QSin-XQ)*dt #input from upstream node
    XVSin = XQ*dt # excess input sediment will be deposited
    #At middle of timestep
    VSdep = cp.where(dQSdx<0.0, -dQSdx*length*dt,0.0) #deposition to basal sediment layer
    VSnew = dQSdx*length*dt+VSdep #mobilisation of sediment
    #now the balance
    VS = VSinit+VSin+VSnew-VSdep
    #active sediment must be greater than zero
    VS = cp.where(VS>0,VS,0.0)
    VS = cp.where(VS<kmax*length,VS,kmax*length) #limit by the maximum capacity of edge to have active sediment
    XVS = cp.where(VS>kmax*length,VS-kmax*length,0.0) + XVSin
    VSdep += XVS  #excess volume deposited to basal sediment layer
    # for detritus and grain size we want to distinguish erosion and remobilisation
    conB = mt-dQSdx #con B
    VSerod = cp.where(conB > 0, VSnew, mt*length*dt)
    # mobilised till from the basal sediment layer
    VSmob = VSnew-VSerod # will be zero except where con B is < 0 where it should be positive
    if cp.nanmin(VSmob) < 0:
        print ('negative volume found {} in VSmob, assigning excess to VSdep'.format(cp.nanmin(VSnew)))
        VSdep = cp.where(VSmob < 0, -VSmob+VSdep,VSdep)
        VSmob = cp.where(VSmob < 0, 0.0,VSmob)
    #At end of the timestep
    VSout = VS*xcrit/length #sediment leaving the edge
    VSback = cp.zeros_like(VS)
    VSfinal = cp.where(VS-VSout>0,VS-VSout,0.0) #sediment left on the edge
    return [VSinit,VSin,VSdep,VSerod,VSmob,VSout,VSback,VSfinal, XVS]

def MixVols(VolArrays):
    """method to calculate volumetric mixtures on edges"""
    VSinit = VolArrays[0] #as_edge
    VSin= VolArrays[1] #as_node
    VSerod= VolArrays[3] #basement
    VSmob= VolArrays[4] #basal
    #new arrays for volume elements
    V_as_edge = cp.zeros_like(VSinit)
    V_as_node = cp.zeros_like(VSinit)
    V_basal = cp.zeros_like(VSinit)
    V_basement = cp.zeros_like(VSinit)
    #begin timestep
    V_as_edge += VSinit
    V_as_node += VSin
    #middle timestep
    V_basal += VSmob
    V_basement += VSerod
    V_Total = V_as_edge + V_as_node +V_basal + V_basement
    #V proportions -- we assume if V is 0 at this stage that nothing is happening
    P_as_edge = cp.where(V_Total > 0,V_as_edge/V_Total,1.0)
    P_as_node = cp.where(V_Total > 0,V_as_node/V_Total,0)
    P_basal = cp.where(V_Total > 0,V_basal/V_Total,0)
    P_basement = cp.where(V_Total > 0,V_basement/V_Total,0)
    return [P_as_edge, P_as_node, P_basal, P_basement]

# grain-size distributions
def NewDdist(mean,sigma,n):
    """method to draw a sample from a distribution"""
    if sigma == 0:
        #convert from phi to mm
        median = (2**-mean)/1000.
        dist = (mean,0)
    else:
        rng = cp.random.default_rng()
        # cupy only has standard_normal as at writing so we use the older implementation if this fails
        try:
            D_arr = rng.normal(mean,sigma,n)
        except AttributeError:
            D_arr = cp.random.normal(mean,sigma,n)
        medianPhi = cp.nanmedian(D_arr)
        median = (2**-medianPhi)/1000
        dist = (cp.nanmean(D_arr),cp.nanstd(D_arr))
    return median,dist

def CombineDdists(dists,vPs,n, def_mean = 0, def_std = 1):
    """method to combine several distribution samples by volume"""
    if def_std == 0:
        median = (2**-def_mean)/1000
        dist = (def_mean,def_std)
    else:    
        rng = cp.random.default_rng()
        n_els = [int(i*n) for i in vPs]
        if cp.nansum(cp.array(n_els)) > n-len(vPs):
            for i,j in enumerate(dists):
                if i == 0:
                    try:
                        D_arr =  rng.normal(j[0],j[1],n_els[i])
                    except AttributeError:
                        D_arr =  cp.random.normal(j[0],j[1],n_els[i])
                else:
                    try:
                        arr =  rng.normal(j[0],j[1],n_els[i])
                    except AttributeError:
                        arr =  cp.random.normal(j[0],j[1],n_els[i])
                    D_arr = cp.concatenate((D_arr,arr))
        else:
            try:
                D_arr = rng.normal(def_mean,def_std,n)
            except AttributeError:
                D_arr = cp.random.normal(def_mean,def_std,n)
        medianPhi = cp.median(D_arr)
        median = (2**-medianPhi)/1000.
        dist = (cp.nanmean(D_arr),cp.nanstd(D_arr))
    return median,dist

def CombineDdistsArray(Ddists, DvPs, RArray = None, n = None, def_mean = 0, def_std = 1):
    """method to combine several distribution samples by volume"""
    #DvPs = DvPs[DvPs>0] # keep only positive values
    if def_std == 0: # single value option
        median = (2**-def_mean)/1000.
        dist = (def_mean,def_std)
    else:    
        if RArray is not None:
            n = len(RArray)
        elif n is not None:
            #make a 1D array of random numbers
            rng = cp.random.default_rng()
            RArray = rng.standard_normal(size = n)
        else:
            raise ValueError('Both RArray and samp_n are None, you must provide one of these')
        #get the number of elements for each volume input
        n_els = cp.array([cp.rint(i*n) for i in DvPs])
        if type(Ddists) == dict: #works for a dict
            if len(Ddists.items())!= len(n_els):
                print ('Warning: Ddist.items() length ({}) is not the same as valid volume elements({})'.format(len(Ddists.items()),len(n_els)))
            
        #if there are not NaN issues, we can continue
        if cp.nansum(n_els) > n-len(DvPs):
            begins = [int(cp.nansum(n_els[:i])) for i,j in enumerate(n_els)]
            ends = [int(cp.nansum(n_els[:i+1])) for i,j in enumerate(n_els)]
            #initialise an array for mu
            MuArray = cp.ones(n)*def_mean
            #and sigma
            SigArray = cp.ones(n)*def_std
            #Get each shift and scale from Ddists if both are finite
            if type(Ddists) == dict: #works for a dict
                for i,(k,j) in enumerate(Ddists.items()):
                    if cp.isfinite(j[0]) and cp.isfinite(j[1]):
                        MuArray[begins[i]:ends[i]] = j[0] 
                        SigArray[begins[i]:ends[i]] = j[1]
            else:                 
                for i, j in enumerate(Ddists):
                    if cp.isfinite(j[0]) and cp.isfinite(j[1]):
                        MuArray[begins[i]:ends[i]] = j[0] 
                        SigArray[begins[i]:ends[i]] = j[1]
            #shift and scale RandArray
            DArray = RArray*SigArray+MuArray
            medianPhi = cp.nanmedian(DArray)
            dist = (cp.nanmean(DArray),cp.nanstd(DArray))
            if medianPhi < def_mean-2*def_std:
                #median phi is outside 2-sigma limits of population - limiting phi to def_mean-2*def_std)
                medianPhi = def_mean-2*def_std
                dist = (def_mean-2*def_std,def_std)
            elif medianPhi > def_mean+2*def_std:
                #median phi is outside 2-sigma limits of population - limiting phi to def_mean+2*def_std)
                medianPhi = def_mean+2*def_std
                dist = (def_mean+2*def_std,def_std)
            median = (2**-medianPhi)/1000.    
        else:
            median = (2**-def_mean)/1000.
            dist = (def_mean,def_std)
    return median,dist

def ExtractDdistsArray(Ddists, DvPs, RArray = None, n = None, def_mean = 0, def_std = 1):
    """method to extract one or more distributions from another by volume"""
    if def_std == 0:
        median = (2**-def_mean)/1000.
        dist = (def_mean,def_std)
    else:    
        # the first is the base distribution from which the others will be removed
        BaseDist = Ddists[0]
        if RArray is not None:
            n = len(RArray)
        elif n is not None:
            #make a 1D array of random numbers
            rng = cp.random.default_rng()
            RArray = rng.standard_normal(size = n)
        else:
            raise ValueError('Both RArray and samp_n are None, you must provide one of these')
        #get the number of elements for each volume input (except base)
        n_els = cp.array([cp.rint(i*n) for i in DvPs])
        #if there are not NaN issues, we can continue
        if cp.nansum(n_els) > n-len(DvPs):
            #initialise Mu array
            MuArray = cp.ones(n)*BaseDist[0]
            #and sigma array
            SigArray = cp.ones(n)*BaseDist[1]
            #Make the Base Array
            BaseArray = RArray*SigArray+MuArray
            n_remaining = BaseArray.shape[0]
            #For other dists make elements from a Boolean array False in line with their probability, or do nothing
            for i, j in enumerate(Ddists):
                if i != 0 and n_els[i] > 0 and n_remaining > 0:
                    if cp.isfinite(j[0]) and cp.isfinite(j[1]):
                        rng = cp.random.default_rng()
                        #make the Boolean array
                        Bool = cp.full(BaseArray.shape, True, dtype = bool)
                        #make probability array for BaseArray given the distribution
                        try: #here a function that is not in cupy
                            ProbArray = cpx_stats.norm.pdf(BaseArray, loc=j[0], scale=j[1])
                        except AttributeError:
                            def normal_pdf(x, mu, sigma):
                                return (1.0 / (cp.sqrt(2 * cp.pi) * sigma)) * cp.exp(-0.5 * ((x - mu) / sigma) ** 2)
                            ProbArray = normal_pdf(BaseArray, mu=j[0], sigma=j[1])
                        try:
                                scale = 1.0/cp.nansum(ProbArray)
                                ProbArray = ProbArray*scale
                        except ZeroDivisionError:
                                print('ProbArray summed to zero {}. Adding to all {}'.format(cp.nansum(ProbArray),1/len(ProbArray)))
                                ProbArray = ProbArray+1/len(ProbArray) #the case if we have all zeros
                        try:
                            Bool[rng.choice(BaseArray.shape[0], size = int(n_els[i]), replace = False,p = ProbArray)] = False
                        except AttributeError: #p= is apparently not implemented in cupy
                            Bool[cp.random.choice(BaseArray.shape[0], size = int(n_els[i]), replace = False)] = False
                        except ValueError:
                            Bool[rng.choice(BaseArray.shape[0], size = int(n_els[i]), replace = True)] = False
                        BaseArray = BaseArray[Bool]
                        n_remaining = BaseArray.shape[0]
            medianPhi = cp.nanmedian(BaseArray)
            median = (2**-medianPhi)/1000.
            dist = (cp.nanmean(BaseArray),cp.nanstd(BaseArray))
        else:
            median = (2**-def_mean)/1000.
            dist = (def_mean,def_std)
    return median,dist

def MixDetritusbyVolume(det_dod,vol_adj,dets, node_keys):
    #downstream node index - a column in the vol_adj
    v_idx = [node_keys.index(j) for i,j in enumerate(det_dod.keys())]
    #upstream node indices - the rows in the vol_adj
    u_idxs = [[node_keys.index(key) for key in det_dod[j].keys()] for i,j in enumerate(det_dod.keys())]
    # total incoming volumes
    T_vs = [cp.sum(vol_adj[:,i]) for i in v_idx]
    #proportional volumes for preds
    P_vs = [[vol_adj[row,col]/T_vs[col] for row in u_idxs[col]] for col in v_idx]
    #list of dets for preds
    dlist = [
    [dets[node_keys[ui], node_keys[vi]] for ui in u_idxs[i]]
    for i, vi in enumerate(v_idx)
    ]
    #for each node make and populate the new dict
    det_nodes = {}
    for i,node in enumerate(dlist):
        node_id = node_keys[i]
        if len(node) == 0: #i.e. no preds
            det_final = {'init': 1.0}
        elif len(node) == 1: #i.e. one pred, no need to do anything
            det_final = node[0]     
        else:
            P_vols = P_vs[i]  
            weighted_vals = [{Dclass:node[m][Dclass]*P_vols[m] for Dclass in n.keys()} for m,n in enumerate(node)] 
            det_final = {k: cp.sum(d.get(k, 0) for d in weighted_vals) for k in {k for d in weighted_vals for k in d}}
        det_nodes[node_id] = det_final
    return det_nodes
    
    
    
    
    
    
    
    
    
    
    