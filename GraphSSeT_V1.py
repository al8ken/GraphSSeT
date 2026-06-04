#!/usr/bin/env python
"""
Redesigned SubglacialErosionandSedimentFlux class with separated computational functions.

This redesigned version separates graph operations from computational logic:
- Pure computational functions work only with arrays
- Class methods handle graph data extraction and result storage
- Improved modularity, testability, and performance

Author: Redesigned from original GraphSSeT model
"""

import networkx as nx
import os
import itertools
import random
import ray
from NetworkX_funcs import AsynFluid
import sys
from typing import Tuple, List, Dict, Optional, Union

# Import the computational functions
from GraphSSeT_V1_functions import (
    calculate_edge_d_and_rhos,
    calculate_time_step,
    calculate_hydraulic_potential_gradient,
    calculate_erosion_rate,
    calculate_channel_flux_on_edge,
    calculate_transport_capacity_eh,
    calculate_till_mobilisation,
    calculate_till_transport,
    calculate_till_transport_lite,
    calculate_exner_equation,
    calculate_channel_flux_to_node,
    calculate_detritus_tracking
)

# Import CuPy for GPU acceleration
try:
    import cupy as cp
    import cupyx.scipy.stats as cpx_stats
    GPUs = True
    print("CuPy imported successfully. Using GPU acceleration.")
except ImportError:
    print("CuPy not found. Falling back to NumPy (CPU-only).")
    import numpy as cp
    import scipy.stats as cpx_stats
    GPUs = False

if GPUs:
    num_gpus = cp.cuda.runtime.getDeviceCount()
else:
    num_gpus = 0

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Supported methods (unchanged)
_SUPPORTED_POTGRAD_METHODS = ["Direct", "Potential","IceThickness","SurfaceElevation"]
_SUPPORTED_CFLUX_METHODS = ["FluxArea","Flux","InputEdge", "InputNode", "InputBoth"]
_SUPPORTED_TRANSPORT_METHODS = ["EngelundHansen","MeyerParkerMuller"]
_SUPPORTED_EROSION_METHODS = ["Direct","Vel", "VelTau","MixedBed","InfiniteTill"]
_SUPPORTED_DETRITUS_METHODS = ["SedErod","NodeProp","None"]


@ray.remote(max_restarts=0, num_gpus=num_gpus)
class SubglacialErosionandSedimentFlux():
    """
    Redesigned SubglacialErosionandSedimentFlux class with separated computational logic.
    
    This class now acts as a coordinator between NetworkX graph operations and 
    pure computational functions that work with arrays only.
    """
    
    _name = "SubglacialErosionandSedimentFlux"
    __version__ = "2.0"

    def __init__(
        self,
        graph,
        potgrad_method="Direct",
        cflux_method="Flux",
        erosion_method="Direct",
        transport_method="EngelundHansen",
        detritus_method="SedErod",
        bed_porosity=0.3,
        g=9.81,
        fluid_density=1000.0,
        ice_density=910.0,
        HookeAngle=180,
        DarchWeisbachFrictionFactor=0.15,
        Herodelim=0.75,
        MaxTillH=1.0,
        InitTillH=0.00,
        SedimentUptakeLengthFactor=1.,
        Dsig=1e-3,
        Dhmin=0.21,
        meanD=2.2,
        stdD=1.5,
        rhog=2650.0,
        K=1e-4,
        L=1,
        W=2e-10,
        samp_n=100,
        RNarray=None
    ):
        """Initialize the redesigned SubglacialErosionandSedimentFlux class."""
        
        # Validate graph input
        if not isinstance(graph, nx.classes.digraph.DiGraph):
            msg = f"SubglacialErosionandSedimentTransporter: graph must be a networkx directed graph but is {type(graph)}"
            raise TypeError(msg)
        else:
            self._graph = ray.put(graph)
            self._partitions = None

        # Validate and save parameters (unchanged from original)
        if not 0 <= bed_porosity < 1:
            msg = "SubglacialErosionandSedimentTransporter: bed_porosity must be between 0 and 1"
            raise ValueError(msg)
        self._bed_porosity = bed_porosity

        # Save physical parameters
        self._g = g
        self._fluid_density = fluid_density
        self._ice_density = ice_density
        self._time_idx = 0
        self._time = 0.0
        self._beta = cp.radians(HookeAngle)
        self._fr = DarchWeisbachFrictionFactor
        self._Hg = Herodelim
        self._Hlim = MaxTillH
        self._sedl_factor = SedimentUptakeLengthFactor if SedimentUptakeLengthFactor >= 1.0 else 1.0
        self._InitTill = InitTillH
        self._Dsig = Dsig
        self._dhmin = Dhmin
        self._meanD = meanD
        self._stdD = stdD
        self._rhos_grain = rhog
        self._rhos_bulk = rhog*(1-self._bed_porosity)+self._fluid_density*self._bed_porosity
        self._k = K
        self._l = L
        self._w = W
        self._samp_n = samp_n
        
        if RNarray is not None:
            self._RNarray_len = len(RNarray)
            self._RNarray = ray.put(RNarray)
        else: 
            self._RNarray = None
        
        # Validate and save method parameters
        if potgrad_method in _SUPPORTED_POTGRAD_METHODS:
            self._potgrad_method = potgrad_method
        else:
            raise ValueError("SubglacialErosionandSedimentFlux: invalid potential gradient method not supported.")

        if cflux_method in _SUPPORTED_CFLUX_METHODS:
            self._cflux_method = cflux_method
        else:
            raise ValueError("SubglacialErosionandSedimentFlux: invalid channel flux method not supported.")

        if erosion_method in _SUPPORTED_EROSION_METHODS:
            self._erosion_method = erosion_method
        else:
            raise ValueError("SubglacialErosionandSedimentFlux: invalid erosion method not supported.")

        if transport_method in _SUPPORTED_TRANSPORT_METHODS:
            self._transport_method = transport_method
        else:
            raise ValueError("SubglacialErosionandSedimentFlux: invalid transport method not supported.")
        
        if detritus_method in _SUPPORTED_DETRITUS_METHODS:
            self._detritus_method = detritus_method
        else:
            raise ValueError("SubglacialErosionandSedimentFlux: invalid detritus method not supported.")
            
        # Set transport function based on method
        if self._transport_method == "EngelundHansen":
            self._transport_function = self._calc_transport_cap_EH
        elif self._transport_method == "MeyerParkerMuller":
            # Would implement MPM function
            raise NotImplementedError("MeyerParkerMuller not yet implemented in redesigned version")
        
        # Initialize graph attributes and arrays
        self._initialize_graph_attributes()

    def _initialize_graph_attributes(self):
        """Initialize graph attributes and extract key arrays."""
        Graph = ray.get(self._graph)
        
        # Get attribute lists
        self._node_attributes = list(list(Graph.nodes(data=True))[0][-1].keys())
        self._edge_attributes = list(list(Graph.edges(data=True))[0][-1].keys())
        self._edge_keys = list(Graph.edges)
        self._node_keys = list(Graph.nodes)
        
        # Extract edge lengths
        if "length" in self._edge_attributes:
            self._length = self._get_edge_array(Graph, 'length')
        elif "coords" in self._edge_attributes:
            coords = nx.get_edge_attributes(Graph, 'coords')
            self._length = cp.array([
                cp.sqrt((c[1][0]-c[0][0])**2+(c[1][1]-c[0][1])**2) 
                for c in coords.values()
            ])
        else:
            raise ValueError('No way to determine edge length - requires either attribute "length" or coords')
        
        # Extract edge widths
        if "edge_width" in self._edge_attributes:
            self._edgewidth = self._get_edge_array(Graph, 'edge_width')
        else:
            self._edgewidth = cp.sqrt(3)/12 * self._length

    def _get_edge_array(self, graph, attribute_name):
        """Extract edge attribute as array."""
        attr_dict = nx.get_edge_attributes(graph, attribute_name)
        return cp.array([attr_dict[key] for key in self._edge_keys])
    
    def _get_node_array(self, graph, attribute_name):
        """Extract node attribute as array."""
        attr_dict = nx.get_node_attributes(graph, attribute_name)
        return cp.array([attr_dict[key] for key in self._node_keys])
    
    def _set_edge_array(self, graph, array, attribute_name):
        """Set edge attribute from array."""
        attr_dict = {key: array[i] for i, key in enumerate(self._edge_keys)}
        nx.set_edge_attributes(graph, attr_dict, attribute_name)
    
    def _set_node_array(self, graph, array, attribute_name):
        """Set node attribute from array."""
        attr_dict = {key: array[i] for i, key in enumerate(self._node_keys)}
        nx.set_node_attributes(graph, attr_dict, attribute_name)

    def _extract_node_values_for_edges(self, graph, attribute_name):
        """Extract node values for upstream and downstream nodes of each edge."""
        attr_dict = nx.get_node_attributes(graph, attribute_name)
        upstream = cp.array([attr_dict[u] for u, v in self._edge_keys])
        downstream = cp.array([attr_dict[v] for u, v in self._edge_keys])
        return upstream, downstream

    def _calculate_edge_d_and_rhos(self, graph=None):
        """Calculate grain size and grain density using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract existing arrays if available
        d_median = self._get_edge_array(graph, 'd_median') if 'd_median' in self._edge_attributes else None
        d_distribution = [nx.get_edge_attributes(graph, 'd_distribution')[key] 
                         for key in self._edge_keys] if 'd_distribution' in self._edge_attributes else None
        sed_d_distribution = [nx.get_edge_attributes(graph, 'sed_d_distribution')[key] 
                             for key in self._edge_keys] if 'sed_d_distribution' in self._edge_attributes else None
        
        # Call redesigned function
        d_median_result, d_dist_result, sed_d_dist_result = calculate_edge_d_and_rhos(
            n_edges=len(self._edge_keys),
            d_median=d_median,
            d_distribution=d_distribution,
            sed_d_distribution=sed_d_distribution,
            mean_d=self._meanD,
            std_d=self._stdD,
            samp_n=self._samp_n
        )
        
        # Store results in graph
        self._set_edge_array(graph, d_median_result, 'd_median')
        d_dist_dict = {key: d_dist_result[i] for i, key in enumerate(self._edge_keys)}
        nx.set_edge_attributes(graph, d_dist_dict, 'd_distribution')
        sed_d_dist_dict = {key: sed_d_dist_result[i] for i, key in enumerate(self._edge_keys)}
        nx.set_edge_attributes(graph, sed_d_dist_dict, 'sed_d_distribution')
        
        # Store arrays as instance variables
        self._d_median = d_median_result
        self._d_dist = d_dist_result
        self._sed_d_dist = sed_d_dist_result
        self._rhos_active = self._rhos_grain
        
        return graph

    def _get_graph_atts(self, graph=None):
        """Get graph attributes and calculate time step using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Update attribute lists
        self._node_attributes = list(list(graph.nodes(data=True))[0][-1].keys())
        self._edge_attributes = list(list(graph.edges(data=True))[0][-1].keys())
        
        # Extract last time array
        last_time = (self._get_edge_array(graph, 'last_time') 
                    if 'last_time' in self._edge_attributes 
                    else cp.zeros_like(self._length))
        
        # Calculate time step using redesigned function
        dt = calculate_time_step(last_time, self._time)
        self._dt = dt
        
        return graph, dt

    def _calc_HydraulicPotentialGradient(self, graph=None):
        """Calculate hydraulic potential gradient using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract method-specific parameters
        kwargs = {'method': self._potgrad_method, 'length': self._length}
        
        if self._potgrad_method == "Direct":
            if "hyd_pot_grad" not in self._edge_attributes:
                raise ValueError("hyd_pot_grad required for Direct method")
            kwargs['hyd_pot_grad'] = self._get_edge_array(graph, 'hyd_pot_grad')
        
        elif self._potgrad_method == "Potential":
            if "hydraulic_potential" not in self._node_attributes:
                raise ValueError("hydraulic_potential required for Potential method")
            upstream, downstream = self._extract_node_values_for_edges(graph, 'hydraulic_potential')
            kwargs['hydraulic_potential_upstream'] = upstream
            kwargs['hydraulic_potential_downstream'] = downstream
        
        elif self._potgrad_method == "IceThickness":
            if "ice_thickness" not in self._node_attributes or "bed_elevation" not in self._node_attributes:
                raise ValueError("ice_thickness and bed_elevation required for IceThickness method")
            ice_up, ice_down = self._extract_node_values_for_edges(graph, 'ice_thickness')
            bed_up, bed_down = self._extract_node_values_for_edges(graph, 'bed_elevation')
            kwargs.update({
                'ice_thickness_upstream': ice_up,
                'ice_thickness_downstream': ice_down,
                'bed_elevation_upstream': bed_up,
                'bed_elevation_downstream': bed_down,
                'ice_density': self._ice_density,
                'fluid_density': self._fluid_density,
                'g': self._g
            })
        
        elif self._potgrad_method == "SurfaceElevation":
            if "surface_elevation" not in self._node_attributes or "bed_elevation" not in self._node_attributes:
                raise ValueError("surface_elevation and bed_elevation required for SurfaceElevation method")
            surf_up, surf_down = self._extract_node_values_for_edges(graph, 'surface_elevation')
            bed_up, bed_down = self._extract_node_values_for_edges(graph, 'bed_elevation')
            kwargs.update({
                'surface_elevation_upstream': surf_up,
                'surface_elevation_downstream': surf_down,
                'bed_elevation_upstream': bed_up,
                'bed_elevation_downstream': bed_down,
                'ice_density': self._ice_density,
                'fluid_density': self._fluid_density,
                'g': self._g
            })
        
        # Call redesigned function
        DPhi = calculate_hydraulic_potential_gradient(**kwargs)
        
        # Store result
        self._set_edge_array(graph, DPhi, 'DPhi')
        self._DPhi = DPhi
        
        return graph


    def _calc_erosion_rate(self, graph=None):
        """Calculate erosion rate using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract method-specific parameters
        kwargs = {'method': self._erosion_method, 'length': self._length}
        
        if self._erosion_method == "Direct":
            if "erosion_potential" not in self._node_attributes:
                raise ValueError("erosion_potential required for Direct method")
            upstream, downstream = self._extract_node_values_for_edges(graph, 'erosion_potential')
            kwargs['erosion_potential_upstream'] = upstream
            kwargs['erosion_potential_downstream'] = downstream
        
        elif self._erosion_method in ["Vel", "MixedBed"]:
            if "basal_velocity_magnitude" not in self._node_attributes:
                raise ValueError("basal_velocity_magnitude required for Vel/MixedBed method")
            upstream, downstream = self._extract_node_values_for_edges(graph, 'basal_velocity_magnitude')
            kwargs['basal_velocity_upstream'] = upstream
            kwargs['basal_velocity_downstream'] = downstream
            kwargs.update({'k': self._k, 'l': self._l})
        
        elif self._erosion_method == "VelTau":
            if ("basal_velocity_magnitude" not in self._node_attributes or 
                "basal_tau_magnitude" not in self._node_attributes):
                raise ValueError("basal_velocity_magnitude and basal_tau_magnitude required for VelTau method")
            vel_up, vel_down = self._extract_node_values_for_edges(graph, 'basal_velocity_magnitude')
            tau_up, tau_down = self._extract_node_values_for_edges(graph, 'basal_tau_magnitude')
            kwargs.update({
                'basal_velocity_upstream': vel_up,
                'basal_velocity_downstream': vel_down,
                'basal_tau_upstream': tau_up,
                'basal_tau_downstream': tau_down,
                'w': self._w
            })
        
        # Call redesigned function
        erod = calculate_erosion_rate(**kwargs)
        
        # Store result
        self._set_edge_array(graph, erod, 'erosion_potential')
        self._erod = erod
        
        return graph

    def _calc_channel_flux_on_edge(self, graph=None):
        """Calculate channel flux on edges using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays
        channel_flux = (self._get_edge_array(graph, 'channel_flux') 
                       if 'channel_flux' in self._edge_attributes 
                       else cp.zeros_like(self._length))
        
        qw_in = (self._get_edge_array(graph, 'QWin') 
                if 'QWin' in self._edge_attributes 
                else cp.zeros_like(self._length))
        
        # Extract method-specific parameters
        kwargs = {
            'method': self._cflux_method,
            'length': self._length,
            'channel_flux': channel_flux,
            'qw_in': qw_in
        }
        
        if self._cflux_method in ["InputEdge", "InputBoth"]:
            if "edge_input_flux" not in self._edge_attributes:
                raise ValueError("edge_input_flux required for InputEdge/InputBoth method")
            kwargs['edge_input_flux'] = self._get_edge_array(graph, 'edge_input_flux')
        
        if self._cflux_method in ["InputNode", "InputBoth"]:
            if "node_input_flux" not in self._node_attributes:
                raise ValueError("node_input_flux required for InputNode/InputBoth method")
            
            # We have the graph here so can calculate weighted node input flux
            influxnode = cp.zeros_like(self._length)
            for i,(u,v) in enumerate(self._edge_keys):
                    weights = {}
                    aw = 0
                    for key in graph.succ[u]:
                        weights[key] = graph.succ[u][key]['weight']
                        aw += weights[key]
                    influxnode[i] = graph.nodes(data = "node_input_flux")[u] * weights[v]/aw
            
        # Call redesigned function
        cflux, outflux = calculate_channel_flux_on_edge(**kwargs)
        
        # Store results
        self._set_edge_array(graph, cflux, 'channel_flux')
        self._set_edge_array(graph, outflux, 'QWout')
        self._cflux = cflux
        
        return graph

    def _calc_transport_cap_EH(self, graph=None):
        """Calculate transport capacity using Engelund-Hansen method."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays
        qw_in = (self._get_edge_array(graph, 'QWin') 
                if 'QWin' in self._edge_attributes 
                else cp.zeros_like(self._length))
        
        channel_area = self._get_edge_array(graph, 'channel_area') 
                
        try:
            self._cflux = self._cflux #just catches the missing variable
        except AttributeError:
            self._cflux = self._get_edge_array(graph, 'channel_flux')
        
        try: 
           self._DPhi = self._DPhi 
        except AttributeError:
            self._DPhi = self._get_edge_array(graph, 'DPhi')
        
        # Call redesigned function
        QSc, Uv = calculate_transport_capacity_eh(
            cflux=self._cflux,
            qw_in=qw_in,
            dphi=self._DPhi,
            d_median=self._d_median,
            length=self._length,
            channel_area=channel_area,
            beta=self._beta,
            fr=self._fr,
            fluid_density=self._fluid_density,
            rhos_active=self._rhos_active,
            dhmin=self._dhmin,
            g=self._g,
            mean_d=self._meanD
        )
        
        # Store results
        self._set_edge_array(graph, QSc, 'QSc')
        self._set_edge_array(graph, Uv, 'Uv')
        self._QSc = QSc
        self._Uv = Uv
        
        return graph

    def _calc_till_mobilisation(self, graph=None):
        """Calculate till mobilisation using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays
        till_thickness = (self._get_edge_array(graph, 'till_thickness') 
                         if 'till_thickness' in self._edge_attributes 
                         else self._InitTill/2 + cp.random.rand(len(self._edge_keys)) * self._InitTill)
        
        erod = (self._get_edge_array(graph, 'erosion_potential') 
                if 'erosion_potential' in self._edge_attributes 
                else cp.zeros_like(self._length))
        
        qs_in = (self._get_edge_array(graph, 'QSin') 
                if 'QSin' in self._edge_attributes 
                else cp.zeros_like(self._length))
        
        # Call redesigned function
        dQSdx, mt, updated_till_thickness = calculate_till_mobilisation(
            method=self._erosion_method,
            length=self._length,
            edgewidth=self._edgewidth,
            dt=self._dt,
            qsc=self._QSc,
            qs_in=qs_in,
            till_thickness=till_thickness,
            erod=erod,
            sedl_factor=self._sedl_factor,
            bed_porosity=self._bed_porosity,
            h_lim=self._Hlim,
            hg=self._Hg,
            dsig=self._Dsig
        )
        
        # Store results
        self._set_edge_array(graph, dQSdx, 'dQSdx')
        self._set_edge_array(graph, mt, 'mt')
        self._set_edge_array(graph, updated_till_thickness, 'till_thickness')
        self._dQSdx = dQSdx
        self._mt = mt
        self._QSin = qs_in
        
        return graph

    def _calc_till_transport(self, graph=None):
        """Calculate till transport using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays
        flux_density = (self._get_edge_array(graph, 'flux_density') 
                       if 'flux_density' in self._edge_attributes 
                       else cp.zeros_like(self._length))
        
        G_adj = nx.adjacency_matrix(graph, nodelist=self._node_keys) # we pass in a blank adjacency matrix
        
        # Get node status array
        node_status = (self._get_node_array(graph, 'node_status') 
                      if 'node_status' in self._node_attributes 
                      else cp.ones(len(self._node_keys)))  # Default to 'normal'
        
        # Get random number array
        rng = cp.random.default_rng()
        if self._RNarray is not None:
            RNarray = ray.get(self._RNarray)
            if self._time_idx < self._RNarray_len:
                RNA = RNarray[self._time_idx]
            else:
                idx = rng.integers(low=0, high=len(RNarray))
                RNA = RNarray[idx]
        else:
            RNA = rng.standard_normal(size=(graph.number_of_edges(), self._samp_n))
        
        #get upstream edge grain size distributions (as dicts of dicts)
        dist_dod = {
            u: {v: data.get("d_distribution") for v, data in nbrs.items()}
            for u, nbrs in graph.pred.items()
        }
        
        sed_dist_dod = {
            u: {v: data.get("sed_d_distribution") for v, data in nbrs.items()}
            for u, nbrs in graph.pred.items()
        }
        
        tt_edge = GetGraphAttributeToArray(graph, 'till_thickness') #need to bring in these variables
        vt_edge = tt_edge*(1-self._bed_porosity)*self._edgewidth #these too!
        
        # Call redesigned function
        G_props, self._VolArrays, self._PVolArrays = calculate_till_transport(
            flux_density=flux_density,
            length=self._length,
            uv=self._Uv,
            qsc=self._QSc,
            qs_in=self._QSin,
            dqsdx=self._dQSdx,
            mt=self._mt,
            vt = vt_edge,
            dt=self._dt,
            d_dist =self._d_dist,
            sed_d_dist =self._sed_d_dist,
            dod_d_dist=dist_dod,
            dod_sed_d_dist=sed_dist_dod,
            G_adj=G_adj.toarray(),
            node_status=node_status,
            edge_keys=self._edge_keys,
            node_keys=self._node_keys,
            rn_array=RNA,
            samp_n=self._samp_n,
            mean_d=self._meanD,
            std_d=self._stdD
        )
        
        
        #G_props = [jammed, Outflow, QSin, flux_density, d_dist_to_node, d_median, d_dist, sed_d_median, sed_d_dist]
        # Store results to carry over
        self._set_edge_array(graph, G_props[0], 'jammed')
        self._set_node_array(graph, G_props[1], 'VSo')
        self._set_edge_array(graph, G_props[2], 'QSin')
        self._set_edge_array(graph, G_props[3], 'flux_density')
        self._set_node_array(graph, G_props[4], 'd_dist_node')
        self._set_edge_array(graph, G_props[5], 'd_median')
        self._set_edge_array(graph, G_props[6], 'd_distribution')
        self._set_edge_array(graph, G_props[7], 'sed_d_median')
        self._set_edge_array(graph, G_props[8], 'sed_d_distribution')
        
        return graph

    def _calc_Exner_equation(self, graph=None):
        """Calculate Exner equation using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays
        till_thickness = self._get_edge_array(graph, 'till_thickness')
        bed_elevation = (self._get_node_array(graph, 'bed_elevation') 
                        if 'bed_elevation' in self._node_attributes 
                        else cp.zeros(len(self._node_keys)))
        bedrock_elevation = (self._get_node_array(graph, 'bedrock_elevation') 
                            if 'bedrock_elevation' in self._node_attributes 
                            else bed_elevation.copy())

        G_adj = nx.adjacency_matrix(graph, nodelist=self._node_keys) # we pass in a blank adjacency matrix
        
        # Call redesigned function
        dHdt, till_thickness, bed_elevation, bedrock_elevation = calculate_exner_equation(
            dqsdx=self._dQSdx,
            mt=self._mt,
            length = self._length,
            edgewidth=self._edgewidth,
            dt=self._dt,
            till_thickness=till_thickness,
            bed_elevation=bed_elevation,
            bedrock_elevation=bedrock_elevation,
            G_adj=G_adj.toarray(),
            edge_keys=self._edge_keys,
            node_keys=self._node_keys,
            bed_porosity=self._bed_porosity,
            vol_arrays=getattr(self, '_VolArrays', None)
        )
        
        # Store results
        self._set_edge_array(graph, dHdt, 'dHdt')
        self._set_edge_array(graph, till_thickness, 'till_thickness')
        self._set_node_array(graph, bed_elevation, 'bed_elevation')
        self._set_node_array(graph, bedrock_elevation, 'bedrock_elevation')
        
        return graph

    def _calc_channel_flux_to_node(self, graph=None):
        """Calculate channel flux to nodes using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        # Extract required arrays - here is our first time where we need the adjacency matrix
        QW_adj = nx.adjacency_matrix(graph, nodelist = self._node_keys, weight = 'QWout') #this gives all the QWout values
        EW_adj = nx.adjacency_matrix(graph, nodelist = self._node_keys, weight = 'weight') #edge_weights
        
        if 'last_time' in self._edge_attributes:
            LT_adj = nx.adjacency_matrix(graph, nodelist = self._node_keys, weight = 'last_time') #this gives all the last-time values
        else:
            LT_adj = nx.adjacency_matrix(graph, nodelist = self._node_keys, weight = None) #values will be 1, so we will check dt != 0
        
        ## working with the adjacency matrix ##
        #for succs we must apply numpy functions along the rows: sum_succs = np.sum(adj[[i],:])
        #for preds we can apply numpy functions along the cols: sum_preds = np.sum(adj[:,[i]])
        
        edge_weight = self._get_edge_array(graph,'weight')
        
        # Call redesigned function
        qw_node, vw_node, qw_in = calculate_channel_flux_to_node(
            QW_adj=QW_adj.toarray(),
            LT_adj=LT_adj.toarray(),
            EW_adj=EW_adj.toarray(),
            current_time=self._time,
            edge_weight=edge_weight,
            cflux=self._cflux,
            edge_keys=self._edge_keys,
            node_keys=self._node_keys,
        )
        
        # Store results
        self._set_node_array(graph, qw_node, 'QW')
        self._set_node_array(graph, vw_node, 'VW')
        
        # Handle QWin based on flux method
        if self._cflux_method in ["Flux", "FluxArea"]:
            self._set_edge_array(graph, cp.zeros_like(self._length), 'QWin')
        else:
            self._set_edge_array(graph, qw_in, 'QWin')
        
        return graph

    def _detritus_tracking(self, graph=None):
        """Track detritus using redesigned function."""
        if graph is None:
            graph = ray.get(self._graph)
        
        if self._detritus_method == "None":
            return graph
        #print('detritus_1', list(graph.edges())[0], flush = True)
        
        # Extract detritus information as a dict
        detritus = nx.get_edge_attributes(graph, 'detritus', default = {'init': 1})
        sed_detritus = nx.get_edge_attributes(graph, 'sed_detritus', default = {'basal': 1})
        detritus_node = nx.get_node_attributes(graph, 'detritus_node', default = {'init': 1})
        
        Dclasses = ['init','basal','basement']        
        # Extract detritus properties if needed
        if self._detritus_method == "NodeProp":
            if "detritus_prop" in self._node_attributes:
                detritus_prop = nx.get_node_attributes(graph, 'detritus_prop', default = 'basement')
            else:
                raise ValueError('property "detritus_prop" is not a node attribute')
            Dclasses = ['init','basal','basement']+list(set([detritus_prop[n] for n in detritus_prop]))

        #get upstream edge detritus (as dicts of dicts)
        det_dod = {
            u: {v: data.get("detritus") for v, data in nbrs.items()}
            for u, nbrs in graph.pred.items()
        }

        #extract extra data        
        till_thickness = self._get_edge_array(graph, 'till_thickness')
        VSout = self._get_edge_array(graph, 'VSout') if 'VSout' in self._edge_attributes else cp.zeros_like(till_thickness)
        VT = till_thickness*(1-self._bed_porosity)*self._edgewidth
        VSdep = self._VolArrays[2]
        Pdep = cp.where(cp.isfinite(VSdep/(VT+VSdep)),VSdep/(VT+VSdep),0.0)

        # adjacency matrix
        G_adj = nx.adjacency_matrix(graph, nodelist = self._node_keys) 
        
        # Call redesigned function
        detritus, sed_detritus, detritus_node,  = calculate_detritus_tracking(
            method=self._detritus_method,
            p_vol_arrays=self._PVolArrays,
            detritus_edge=detritus,
            det_dod=det_dod,
            G_adj = G_adj.toarray(),
            detritus_node=detritus_node,
            sed_detritus=sed_detritus,
            detritus_classes=Dclasses,
            edge_keys=self._edge_keys,
            node_keys=self._node_keys,
            Pdep=Pdep,
            VSout=VSout,
            detritus_prop=detritus_prop,
        )
        
        # Store results
        nx.set_edge_attributes(graph, detritus, 'detritus')
        nx.set_edge_attributes(graph, sed_detritus, 'sed_detritus')
        nx.set_node_attributes(graph,detritus_node,'detritus_node')
        
        return graph

    def CheckAttributes(self, graph):
        """Check that required attributes exist in the graph."""
        # This would implement attribute checking logic
        pass

    # Additional utility methods for compatibility
    def _update_transport(self, *args, **kwargs):
        """Wrapper for transport capacity calculation."""
        return self._transport_function(*args, **kwargs)
    
    def _update_time(self, time):
        """Set the current simulation time and index."""
        self._time = time
        self._time_idx += 1
    
    def _reset_graph_time(self, graph):
        #record last_time on subgraph edges
        SetGraphAttributeFromArray(graph, self._time, self._edge_keys, 'last_time')
        return graph
    
    def get_graph(self):
        """Get the current graph."""
        return ray.get(self._graph)
    
    def set_graph(self, graph):
        """Update the stored graph."""
        self._graph = ray.put(graph)

    #for a 'steady' or timesliced setup we have steady state inputs for hydraulic potential (or relevant inputs), hydrology input and erosion rate 
    #thus we can calculate these once only

    def initialise_steady(self):
        #get graph attribute list
        G,self._dt = self._get_graph_atts()
        #recalculate updated Hydraulic Potential Gradient for network grid
        G = self._calc_HydraulicPotentialGradient(graph = G)
        #calculate hydrology flow for network grid
        G = self._calc_channel_flux_on_edge(graph = G)
        G = self._calc_channel_flux_to_node(graph = G)
        #calculate erosion rate
        G = self._calc_erosion_rate(graph = G)
        self._graph = ray.put(G)
    
    # for timesteps we resolve the downstream flow of water and sediment
    def run_one_step_steady(self, time, lite = False):
        if type(time) is not int:
            time = int(time)
        #get new time
        self._update_time(time)
        #get graph attribute list for each iteration
        G,self._dt = self._get_graph_atts()
        #change density and grain size information
        G = self._calculate_edge_d_and_rhos(graph = G)
        #calculate transport capacity
        G = self._update_transport(graph = G)
        #calculate the till mobilisation
        G = self._calc_till_mobilisation(graph = G)
        #calculate the till transport
        if lite:
            G = self._calc_till_transport_lite(graph = G)
        else:
            G = self._calc_till_transport(graph = G)
        #calculate Exner equation
        G = self._calc_Exner_equation(graph = G)
        #track detritus
        G = self._detritus_tracking(graph = G)
        #calculate hydrology flow accummulation for next timestep
        self._calc_channel_flux_to_node(graph = G)
        #reset graph time
        G = self._reset_graph_time(graph = G)
        return G
        
    # or for fully dynamic time-variable inputs we do everything every timestep
    def run_one_step_dynamic(self, time, lite = False):
        #get new time
        self._update_time(time)
        #get graph attribute list
        G,self._dt = self._get_graph_atts() #issue is here?
        #recalculate updated Hydraulic Potential Gradient for network grid
        G = self._calc_HydraulicPotentialGradient(graph = G)
        #calculate hydrology flow for network grid
        G = self._calc_channel_flux_on_edge(graph = G)
        G = self._calc_channel_flux_to_node(graph = G)
        #calculate erosion rate
        G = self._calc_erosion_rate(graph = G)
        #change density and grain size information
        G = self._calculate_edge_d_and_rhos(graph = G)
        #calculate transport capacity
        G = self._update_transport(graph = G)
        #calculate the till mobilisation
        G = self._calc_till_mobilisation(graph = G)
        #calculate the till transport
        if lite:
            G = self._calc_till_transport_lite(graph = G)
        else:
            G = self._calc_till_transport(graph = G)
        #calculate Exner equation
        G = self._calc_Exner_equation(graph = G)
        #track detritus
        G = self._DetritusTracking(graph = G)
        #calculate hydrology flow accummulation for next timestep
        self._calc_channel_flux_to_node(graph = G)
        #reset graph time
        G = self._reset_graph_time(graph = G)
        return G
        
@ray.remote(max_restarts = 0)#, num_gpus = num_gpus) #no gpus for now
class SGST_supervisor():
    def __init__(self, Graph, RNarray = None, n = None):
        self._graph = Graph
        if RNarray is None:
            self._hasRNA = False
        else:
            self._hasRNA = True
            self._RNA = RNarray
        self._n = n
    
    def MakePartitions(self, method = 'AsynFluid'):
        if self._n == None:
            self._n = os.cpu_count()
        if method == 'AsynFluid':
            G = self._graph.to_undirected()
            Comms = AsynFluid(G,self._n)
        SetGraphAttributeFromArray(self._graph, -1, self._graph.nodes(), "partition_ID", is_node_attribute=True)
        SetGraphAttributeFromArray(self._graph, -1, self._graph.edges(), "partition_ID")
        for key in Comms.keys():
            SG = self._graph.subgraph(Comms[key])    
            SetGraphAttributeFromArray(SG, key, SG.edges(), "partition_ID")
            SetGraphAttributeFromArray(SG, key, SG.nodes(), "partition_ID", is_node_attribute=True)

    def PartitionGraph(self, p_prop='partition_ID'):
        p_IDs = nx.get_node_attributes(self._graph,p_prop)
        ID_array = cp.array([p_IDs[key] for key in p_IDs])
        self._IDs = cp.unique(ID_array)
        def makeSubgraph(ID_dict,ID):
            nodes = {key for key in ID_dict.keys() if ID_dict[key] == ID}
            for n in nodes:
                p_nodes = set(self._graph.predecessors(n)) #first order predecessors of nodes in the partition
                s_nodes = set(self._graph.successors(n)) #first order successors of nodes in the partition
                nodes = nodes.union(p_nodes,s_nodes)
                for p in p_nodes:
                    pp_nodes = set(self._graph.predecessors(p)) #predecessors of predecessor nodes to the partition
                    ps_nodes = set(self._graph.successors(p)) #predecessors of successor nodes to the partition
                    nodes = nodes.union(pp_nodes,ps_nodes)
                for s in s_nodes:
                    sp_nodes = set(self._graph.predecessors(s)) #succcessors of predecessor nodes to the partition
                    ss_nodes = set(self._graph.successors(s)) #succcessors of successor nodes to the partition
                    nodes = nodes.union(sp_nodes,ss_nodes)        
            subG = self._graph.subgraph(nodes)  
            return subG
        HaloEdges = [(u,v) for u,v,e in self._graph.edges(data=True) if e[p_prop] == -1]
        self._partitions = {-1: self._graph.edge_subgraph(HaloEdges)}
        if self._hasRNA:
            rng = cp.random.default_rng()
            RNA_subset = {-1: rng.choice(self._RNA, size = len(HaloEdges), axis = 1, shuffle = False)}
        for ID in self._IDs:
            if ID != -1:
                SG = makeSubgraph(p_IDs,ID)
                print('part {}, num edges {}'.format(ID, SG.number_of_edges()))
                self._partitions[ID] = SG.copy()
                if self._hasRNA:
                    num_edges = SG.number_of_edges()
                    RNA_subset[ID] = rng.choice(self._RNA, size = num_edges, axis = 1, shuffle = False)
                    self._RNA_subset = ray.put(RNA_subset)
        return self._partitions

    def UpdateSubsfromMain(self):
        for ID in self._IDs:
            if ID != -1:
                G = self._partitions[ID]
                edges = [(u,v,e) for u,v,e in self._graph.edges(data=True) if (u,v) in G.edges()]
                nodes = [(u,e) for u,e in self._graph.nodes(data=True) if u in G.nodes()]
                G.update(edges = edges, nodes = nodes)
                self._partitions[ID] = G
                
    def UpdateSubfromMain(self, ID):
        G = self._partitions[ID]
        edges = [(u,v,e) for u,v,e in self._graph.edges(data=True) if (u,v) in G.edges()]
        nodes = [(u,e) for u,e in self._graph.nodes(data=True) if u in G.nodes()]
        G.update(edges = edges, nodes = nodes)
        self._partitions[ID] = G
    
    def HarmonisePartitions(self, p_prop='partition_ID'):
        #get the main graph from data store
        MainGraph = self._graph.copy()
        for key in self._partitions.keys():
            if key != -1:
                G_par = self._partitions[key]
                #Unambiguous edges come from their own partition
                p_edges =[(u,v,e) for u,v,e in G_par.edges(data=True) if e[p_prop] == key]
                #HaloEdges come if downstream node is in another partition (i.e. we are in the up_graph)
                h_edges = [(u,v,e) for u,v,e in G_par.edges(data=True) if G_par.nodes[v][p_prop] != key]
                #All nodes come from their own partition
                p_nodes = [(u,e) for u,e in G_par.nodes(data=True) if e[p_prop] == key]
                # and update the main graph
                MainGraph.update(edges = p_edges+h_edges, nodes = p_nodes)
        self._graph = MainGraph
    
    def HarmonisePartition(self, ID, p_prop='partition_ID'):
        #get the main graph from data store
        MainGraph = self._graph.copy()
        G_par = self._partitions[ID]
        #Unambiguous edges come from their own partition
        p_edges =[(u,v,e) for u,v,e in G_par.edges(data=True) if e[p_prop] == ID]
        #HaloEdges come if downstream node is in another partition (i.e. we are in the up_graph)
        h_edges = [(u,v,e) for u,v,e in G_par.edges(data=True) if G_par.nodes[v][p_prop] != ID]
        #All nodes come from their own partition
        p_nodes = [(u,e) for u,e in G_par.nodes(data=True) if e[p_prop] == ID]
        # and update the main graph
        MainGraph.update(edges = p_edges+h_edges, nodes = p_nodes)
        self._graph = MainGraph
        
    def SpawnWorkerActors(self, sgst_kwargs):
        #make worker actors for each partition
        self.UpdateSubsfromMain()
        if self._hasRNA:
            partition_sgsts = {ID: SubglacialErosionandSedimentFlux.remote(self._partitions[ID], RNarray = ray.get(self._RNA_subset)[ID], **sgst_kwargs) for ID in self._IDs if ID != -1}
        else:
            partition_sgsts = {ID: SubglacialErosionandSedimentFlux.remote(self._partitions[ID],**sgst_kwargs) for ID in self._IDs if ID != -1}
        return partition_sgsts
    
    def RunWorkerActorsStrict(self, actor_set, t, lite = False):
        #here we run all workers completely (blocking until done) and harmonise only once all are done
        #therefore nothing is synched 'out of order'
        def task(ID):
            graph = actor_set[ID].run_one_step_steady.remote(t, lite = lite)
            return graph
        workers = {ID: task(ID) for ID in self._IDs if ID != -1}
        resultIDs = [ID for ID in workers]
        partitions = ray.get([workers[ID] for ID in workers])
        for i,j in enumerate(resultIDs):
            self._partitions[j] = partitions[i]
        self.HarmonisePartitions()
        self.UpdateSubsfromMain()
        ray.get([actor_set[ID].set_graph.remote(self._partitions[ID]) for ID in self._IDs if ID != -1])
        
    def RunWorkerActorsSemiStrict(self, actor_set, t,lite = False):
       #here we run harmonise each worker when it is done
       #therefore graphs may be synched 'out of order' but this SHOULD save some time
       #it is still blocking with respect to the timestep
       def task(ID):
           graph = actor_set[ID].run_one_step_steady.remote(t, lite = lite)
           return graph
       workers = {ID: task(ID) for ID in self._IDs if ID != -1}
       tasklist = [workers[ID] for ID in workers]
       while tasklist:
           finished,tasklist = ray.wait(tasklist, num_returns = 1)
           resultID = [key for key in workers if workers[key] == finished[0]]
           self._partitions[resultID[0]] = ray.get(finished[0])
           self.HarmonisePartition(resultID[0])
           #update partition - while 'main' is set, here there is no guarantee that partitions == main
           ray.get(actor_set[resultID[0]].set_graph.remote(self._partitions[resultID[0]]))
       
    def RunWorkerActorsFlexi(self, actor_set, t, lite = False):
       #here we run each worker completely asynchronously and harmonise once each is done - useful where they are not overlapping
       #here it is NOT blocking with respect to the timestep and you'll need to control blocking outside this function (with ray.get())
       def task(ID):
           actor_set[ID].run_one_step_steady.remote(t, lite = lite)
           return ID
       workers = [task(ID) for ID in self._IDs if ID != -1] 
       while workers:
           finished,unfinished = ray.wait(workers, num_returns = 1)
           resultID = ray.get(finished[0])
           self.HarmonisePartition(resultID)
           actor_set[resultID].set_graph.remote(self._partitions[resultID]) 
    
    def get_graph(self):
        Graph = self._graph
        return Graph
    
    def set_graph(self, Graph):
            self._graph = Graph
            try:
                #if we have partitions already
                self.UpdateSubsfromMain()
            except AttributeError:
                #if we don't it does not matter
                pass

## functions for parallelisation ##
    
# divide into chunks #    
def chunks(l, n):
    """Divide a list of nodes or edges `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x
        
# do so with shuffled nodes
def randomchunks(l, n):
    """Divide a list of shuffled nodes or edges `l` in `n` chunks"""
    random.shuffle(l)
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

# this function is needed where we need a 'graph' but not a connected one to compute things on isolated nodes
# each node is included once and once only; edges are included ONLY where between included nodes
# therefore edges may be missing - use only for node-only oprerations
def SplitGraphbyNodes(G, node_list, processes=None):
    divisor = processes-1
    num_nodes = len(node_list)
    if num_nodes < divisor:
        divisor = num_nodes
    node_chunks = list(randomchunks(node_list, num_nodes // divisor))
    num_chunks = len(node_chunks)
    Graphs = [G.subgraph(node_chunks[i]) for i in range(num_chunks)]
    return(Graphs)


# this function is needed where we need a 'graph' but not a connected one to compute things on isolated edges
# each edge is included once and once only; nodes are included for all edges
# therefore nodes are duplicated - use only for edge-only operations
def SplitGraphbyEdges(G, edge_list, processes=None):
    divisor = processes-1
    num_edges = len(edge_list)
    if num_edges < divisor:
        divisor = num_edges
    edge_chunks = list(randomchunks(edge_list, num_edges // divisor))
    num_chunks = len(edge_chunks)
    Graphs = [G.edge_subgraph(edge_chunks[i]) for i in range(num_chunks)]
    Gascopy = [G.copy() for G in Graphs]
    #return(Graphs) # here you return a view - but I don't think this works
    return(Gascopy) # here you return a copy - faster

# this function is needed where we need a valid graph partition
# partitions will have two additional rings of edges

def SetGraphAttributeFromArray(Graph, array, keys, attribute, is_node_attribute=False):
    if is_node_attribute:
        for i, u in enumerate(keys):
            try:
                Graph.nodes[u][attribute] = array[i]
            except (TypeError, IndexError):
                Graph.nodes[u][attribute] = array
    else:
        for i, (u, v) in enumerate(keys):
            try:
                Graph.edges[u, v][attribute] = array[i]
            except (TypeError, IndexError):
                Graph.edges[u, v][attribute] = array

def GetGraphAttributeToArray(Graph, attribute, is_node_attribute=False):
    if is_node_attribute:
        a = nx.get_node_attributes(Graph, attribute)
        #keys = Graph.nodes()
    else:
        a = nx.get_edge_attributes(Graph, attribute)
        #keys = Graph.edges()
    arr = cp.array([a[key] for key in a])
    return arr