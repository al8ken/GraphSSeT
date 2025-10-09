#!/usr/env/python

"""
This is the GraphSSeT model that simulates the transport of sediment in the 
subglacial environment, while tracking the grain size, detritus, 
basal sediment and erosion of the bed

The fundamental approach is based on the SUGSET transport formulations of 

Delaney et al (2019) https://doi.org/10.1029/2019JF005004. 

The network transport uses a kinematic wave approach to control network-scale flow

For the full model dscriptions the paper of Aitken et al., (2024)

https://doi.org/10.xxxx.

codeauthor:: Alan Aitken

First Created on Fr July 22, 2022
Last edit was December 2023
"""

#import required python modules
import numpy as np
import networkx as nx
import os
import itertools
import random
import ray
from NetworkX_funcs import AsynFluid
from scipy.stats import norm
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#These are the methods that are or are intended be supported
#see details of methods below
#potential gradient calculation methods"
_SUPPORTED_POTGRAD_METHODS = ["Direct", "Potential","IceThickness","SurfaceElevation"]
#channel flux calculation method"
_SUPPORTED_CFLUX_METHODS = ["FluxArea","Flux","InputEdge", "InputNode", "InputBoth"]
#sediment transport capacity formulation - only EngelundHansen for now
_SUPPORTED_TRANSPORT_METHODS = ["EngelundHansen","MeyerParkerMuller"]
#Erosion law formulation
_SUPPORTED_EROSION_METHODS = ["Direct","Vel", "VelTau","MixedBed","InfiniteTill"]
#detritus tracking methods
_SUPPORTED_DETRITUS_METHODS = ["SedErod","NodeProp","None"]

@ray.remote(max_restarts = 0)
class SubglacialErosionandSedimentFlux():
    _name = "SubglacialErosionandSedimentFlux"

    __version__ = "1.0"

    def __init__(
        self,
        graph,
        potgrad_method = "Direct",
        cflux_method = "Flux",
        erosion_method = "Direct",
        transport_method="EngelundHansen",
        detritus_method="SedErod",
        bed_porosity=0.3,
        g=9.81, #m/s^2
        fluid_density=1000.0, #kg/m^3
        ice_density = 910.0, #kg/m^3
        HookeAngle=180, #degrees -- 'pi' is a semi-circle appropriate for GlaDS
        DarchWeisbachFrictionFactor=0.15,
        Herodelim = 0.75, #m
        MaxTillH=1.0, #m
        InitTillH = 0.00, #m
        SedimentUptakeLengthFactor = 1., #m default is to use edge length, values > 1 will damp the and make the models less sensitive; values < 1 are not permitted
        Dsig = 1e-3, #m
        Dhmin = 0.21, #m
        meanD = 2.2, # as Phi
        stdD = 1.5, # as Phi
        rhog = 2650.0, #kg m-3
        K = 1e-4,
        L = 1,
        W = 2e-10,
        samp_n = 100,
        RNarray = None
    ):
        if not isinstance(graph,nx.classes.digraph.DiGraph):
            msg = "SubglacialErosionandSedimentTransporter: graph must be a networkx directed graph but is {}".format(type(graph))
            raise TypeError(msg)
        else:
            self._graph = ray.put(graph)
            self._partitions = None

        # verify and save the bed porosity.
        if not 0 <= bed_porosity < 1:
            msg = "SubglacialErosionandSedimentTransporter: bed_porosity must be" "between 0 and 1"
            raise ValueError(msg)
        self._bed_porosity = bed_porosity

        # save or create other key properties.
        self._g = g
        self._fluid_density = fluid_density
        self._ice_density = ice_density
        self._time_idx = 0
        self._time = 0.0
        self._beta = np.radians(HookeAngle)
        self._fr = DarchWeisbachFrictionFactor
        self._Hg = Herodelim
        self._Hlim = MaxTillH
        if SedimentUptakeLengthFactor >= 1.0:
            self._sedl_factor = SedimentUptakeLengthFactor
        else:
            self._sedl_factor = 1.
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
        
        #check the potential gradient method is valid
        if potgrad_method in _SUPPORTED_POTGRAD_METHODS:
            self._potgrad_method = potgrad_method
        else:
            msg = "SubglacialErosionandSedimentFlux: invalid potential gradient method not supported."
            raise ValueError(msg)

        #check the channel flux method is valid
        if cflux_method in _SUPPORTED_CFLUX_METHODS:
            self._cflux_method = cflux_method
        else:
            msg = "SubglacialErosionandSedimentFlux: invalid channel flux method not supported."
            raise ValueError(msg)

        #check the erosion method is valid
        if erosion_method in _SUPPORTED_EROSION_METHODS:
            self._erosion_method = erosion_method
        else:
            msg = "SubglacialErosionandSedimentFlux: invalid erosion method not supported."
            raise ValueError(msg)

        # check the transport method is valid.
        if transport_method in _SUPPORTED_TRANSPORT_METHODS:
            self._transport_method = transport_method
        else:
            msg = "SubglacialErosionandSedimentFlux: invalid transport method not supported."
            raise ValueError(msg)
        
        #check the detritus method is valid    
        if detritus_method in _SUPPORTED_DETRITUS_METHODS:
            self._detritus_method = detritus_method
        else:
            msg = "SubglacialErosionandSedimentFlux: invalid detritus method not supported."
            raise ValueError(msg)
            
        # update the update_transport function to be the correct function for the chosen transport method.
        if self._transport_method == "EngelundHansen":
            self._update_transport = self._calc_transport_cap_EH
        elif self._transport_method == "MeyerParkerMuller":
            self._update_transport = self._calc_transport_cap_MPM
        
        Graph = ray.get(self._graph)
        # get a list of current node attributes from first node entry
        self._node_attributes = list(list(Graph.nodes(data = True))[0][-1].keys())
        # get a list of current edge attributes from first edge entry
        self._edge_attributes = list(list(Graph.edges(data = True))[0][-1].keys())
        
        # establish edge lengths from coords if not already specified as 'length'
        # length is used as the base for many later calculations - we define everything as np arrays in the form [u,v,data]
        if "length" in self._edge_attributes:
            self._length = GetGraphAttributeToArray(Graph, 'length')
        elif "coords" in self._edge_attributes:
            a = nx.get_edge_attributes(Graph,'coords')
            l = lambda c: np.sqrt((c[1][0]-c[0][0])**2+(c[1][1]-c[0][1])**2)
            self._length = np.array([l(a[key]) for key in a])
        else:
            msg = ('no way to determine edge length - requires either attribute "length" or coords as [[X0,Y0],[X1,Y1]]')
            raise ValueError(msg)      
        if "edge_width" in self._edge_attributes:
            self._edgewidth = GetGraphAttributeToArray(Graph, 'edge_width')
        else:
            l = lambda c : np.sqrt(3)/12*c
            self._edgewidth = np.array([l(c) for c in self._length])
        
    def _calculate_edge_d_and_rhos(self, Graph = None):
        """methods to calculate grain size and grain density"""
        if Graph == None:
            Graph = ray.get(self._graph)
        #if not included in graph, initialise variable grain size as samples from initial distribution.
        #we keep track of grain size as a log-normal distribution
        if "d_median" in self._edge_attributes:
            self._d_median = GetGraphAttributeToArray(Graph,'d_median')
            if 'd_distribution' in self._edge_attributes:
                self._d_dist = GetGraphAttributeToArray(Graph,'d_distribution')
            else: 
                #else we assume the default distribution
                self._d_dist = [(self._meanD, self._stdD) for key in self._edge_keys]
                SetGraphAttributeFromArray(Graph, self._d_dist, self._edge_keys, 'd_distribution')
        else: #define new values using a log normal distribution
            newDs = [NewDdist(self._meanD, self._stdD, self._samp_n) for i in self._edge_keys]
            self._d_median = np.array([i[0] for i in newDs])
            SetGraphAttributeFromArray(Graph, self._d_median, self._edge_keys, 'd_median')
            self._d_dist = [i[1] for i in newDs]
            SetGraphAttributeFromArray(Graph, self._d_dist, self._edge_keys, 'd_distribution')
        if 'sed_d_distribution' in self._edge_attributes:
            self._sed_d_dist = GetGraphAttributeToArray(Graph,'sed_d_distribution')
        else: 
            newDs = [NewDdist(self._meanD, self._stdD, self._samp_n) for i in self._edge_keys]
            self._sed_d_dist = [i[1] for i in newDs]
            SetGraphAttributeFromArray(Graph, self._sed_d_dist, self._edge_keys, 'sed_d_distribution')
        #sediment grain density is constant for now
        self._rhos_active = self._rhos_grain
        #self._rhos_active = np.array([self._rhos_grain for i in self._edge_keys])
        return Graph
    
    #at the start of the step, access the graph to define the key attributes
    def _get_graph_atts(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        # get a list of current node attributes from first node entry
        self._node_attributes = list(list(Graph.nodes(data = True))[0][-1].keys())
        # get a list of current edge attributes from first edge entry
        self._edge_attributes = list(list(Graph.edges(data = True))[0][-1].keys())
        self._edge_keys = list(Graph.edges)
        self._node_keys = list(Graph.nodes)
        # get last time for calculation on edge and so dt for each edge
        #NOTE: dt is not a constant!
        if "last_time" in self._edge_attributes:
            lt = GetGraphAttributeToArray(Graph, 'last_time')
        else:
            lt = np.zeros_like(self._length)
        dt = self._time - lt
        return Graph,dt
    
    #GraphSSeT may be calculated for steady state or dynamic runs. 
    #In steady state the following are calculated once only; in dynamic mode they are recaluculated every timestep
    
    def _calc_HydraulicPotentialGradient(self, Graph = None):
        """methods to calculate the hydraulic potential gradient"""
        if Graph == None:
            Graph = ray.get(self._graph)
        #check for required values and make array of gradient values for links
        msg = "SubglacialErosionandSedimentFlux: Data attributes needed for selected potential gradient method not found."
        if self._potgrad_method == "Direct": #reads it from an edge property the graph 
            if "hyd_pot_grad" in self._edge_attributes:
                self._DPhi = GetGraphAttributeToArray(Graph, 'hyd_pot_grad')
            else:
                raise ValueError(msg)
        elif self._potgrad_method == "Potential": #calculates the gradient from the potential (defined on the nodes) 
            if "hydraulic_potential" in self._node_attributes:
                #n1-n2 should give non-negative values if flow is not misdirected - we check later where needed
                l = lambda n1,n2,i: (Graph.nodes(data = "hydraulic_potential")[n1]-Graph.nodes(data = "hydraulic_potential")[n2])/self._length[i]
                self._DPhi = np.array([l(u,v,i) for i,(u,v) in enumerate(self._edge_keys)])
            else:
                raise ValueError(msg)
        elif self._potgrad_method == "IceThickness": #calculates the gradient from the ice thickness and bed elevation using SHREVE_potential function.
            if "ice_thickness" in self._node_attributes and "bed_elevation" in self._node_attributes:
                shreve = lambda n: SHREVE_potential(Graph.nodes(data = "ice_thickness")[n],
                                      Graph.nodes(data = "bed_elevation")[n],
                                      self._ice_density,
                                      self._fluid_density,
                                      self._g,
                                      s_isThick = True,
                                      )
                #n1-n2 should give non-negative values
                l = lambda n1,n2,i: (shreve(n1)-shreve(n2))/self._length[i]
                self._DPhi = np.array([l(u,v,i) for i,(u,v) in enumerate(self._edge_keys)])
            else:
                raise ValueError(msg)
        elif self._potgrad_method == "SurfaceElevation": #calculates the gradient from the surface and bed elevation using SHREVE_potential function.
            if "surface_elevation" in self._node_attributes and "bed_elevation" in self._node_attributes:
                shreve = lambda n: SHREVE_potential(Graph.nodes(data = "surface_elevation")[n],
                                      Graph.nodes(data = "bed_elevation")[n],
                                      self._ice_density,
                                      self._fluid_density,
                                      self._g,
                                      s_isThick = False,
                                      )
                #n1-n2 should give non-negative values
                l = lambda n1,n2,i: (shreve(n1)-shreve(n2))/self._length[i]
                self._DPhi = np.array([l(u,v,i) for i,(u,v) in enumerate(self._edge_keys)])
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)
        #assign values to graph
        SetGraphAttributeFromArray(Graph, self._DPhi, self._edge_keys, 'DPhi')
        return Graph
    
    def _calc_erosion_rate(self, Graph = None):
        if Graph == None:
            print('No graph provided, getting the base graph')
            Graph = ray.get(self._graph)
        """methods to calculate the erosion potential using one of the permitted methods in functions VelScaled or VelTauScaled"""
        msg = "SubglacialErosionandSedimentFlux: Data attributes needed for erosion method not found."
        if self._erosion_method == "InfiniteTill":
            self._erod = np.zeros_like(self._length)
        elif self._erosion_method == "Direct": # Get mean erosion potential for edge from nodes if provided
            if "erosion_rate" in self._node_attributes:
                l = lambda n1,n2: (Graph.nodes(data = "erosion_rate")[n1]+Graph.nodes(data = "erosion_rate")[n2])/2
                self._erod = np.array([l(u,v) for (u,v) in self._edge_keys])
            else:
                raise ValueError(msg)                    
        elif self._erosion_method == "Vel":  #calculate using velocity raised to a power
            if "basal_velocity_magnitude" in self._node_attributes:
                l = lambda n1,n2: (Graph.nodes(data = "basal_velocity_magnitude")[n1]+Graph.nodes(data = "basal_velocity_magnitude")[n2])/2
                v = np.array([l(u,v) for (u,v) in self._edge_keys])
                self._erod = VelScaled(v,self._k,self._l,units = 'm s-1')
            else:
                raise ValueError(msg)                    
        elif self._erosion_method == "VelTau":  #calculate using basal velocity and shear stress
            if "basal_velocity_magnitude" in self._node_attributes and "basal_tau_magnitude" in self._node_attributes:
                l = lambda n1,n2: (Graph.nodes(data = "basal_velocity_magnitude")[n1]+Graph.nodes(data = "basal_velocity_magnitude")[n2])/2
                v = np.array([l(u,v) for (u,v) in self._edge_keys])
                l = lambda n1,n2: (Graph.nodes(data = "basal_tau_magnitude")[n1]+Graph.nodes(data = "basal_tau_magnitude")[n2])/2
                tau = np.array([l(u,v) for (u,v) in self._edge_keys])
                self._erod = VelTauScaled(tau,v,self._w)
            else:
                raise ValueError(msg)                    
        elif self._erosion_method == "MixedBed":
            if "basal_velocity_magnitude" in self._node_attributes:
                l = lambda n1,n2: (Graph.nodes(data = "basal_velocity_magnitude")[n1]+Graph.nodes(data = "basal_velocity_magnitude")[n2])/2
                v = np.array([l(u,v) for (u,v) in self._edge_keys])
                self._erod = VelScaled(v,self._k,self._l,units = 'm s-1')
            else:
                raise ValueError(msg)                    
        else:
            raise ValueError("erosion mode not found")
        SetGraphAttributeFromArray(Graph, self._erod, self._edge_keys, 'erosion_potential')
        return Graph
    
    def _calc_channel_flux_on_edge(self, Graph = None):
        if Graph == None:
            print('No graph provided, getting the base graph')
            Graph = ray.get(self._graph)
        """methods to calculate channelised flux on edges and output to downstream node"""
        msg = "SubglacialErosionandSedimentFlux: Data attributes needed for selected flux method not found."
        #We don't do any flow accummulation in here - this is purely local.
        if "channel_flux" in self._edge_attributes: #get it from the graph
            chanflux = GetGraphAttributeToArray(Graph,'channel_flux')
        else:
            chanflux = np.zeros_like(self._length)
        if "QWin" in self._edge_attributes: #get it from the graph
                influx = GetGraphAttributeToArray(Graph,'QWin')
        else:
                influx = np.zeros_like(self._length)
        if self._cflux_method == "Flux" or self._cflux_method == "FluxArea":
                self._cflux = chanflux
                outflux = chanflux
                #in m^3/s already and ready to use
                #average and output node flux rate is c_flux
        elif self._cflux_method =="InputEdge": #get it from inputs along edges
            if "edge_input_flux" in self._edge_attributes:
                influxedge = GetGraphAttributeToArray(Graph, 'edge_input_flux')
                # in m^2/s - a linear input per metre along the edge so we progressively add to channel flux - it can be negative
                # average flux is (c_flux + influx*length/2)
                self._cflux = influx + influxedge*self._length/2
                outflux = chanflux + influxedge*self._length/2
            else:
                raise ValueError(msg)                    
        elif self._cflux_method =="InputNode": #get it from inputs at nodes
            if "node_input_flux" in self._node_attributes:
                #in m^3/s - i.e a simple volumetric flux. Here we just add this to the channel flux
                influxnode = np.zeros_like(chanflux)
                for i,(u,v) in enumerate(self._edge_keys):
                    #flux at the node may go to several edges so we reweight between those
                    weights = {}
                    aw = 0
                    for key in Graph.succ[u]:
                        weights[key] = Graph.succ[u][key]['weight']
                        aw += weights[key]
                    influxnode[i] = Graph.nodes(data = "node_input_flux")[u] * weights[v]/aw
                self._cflux = influx + influxnode
                outflux = chanflux
            else:
                raise ValueError(msg)
        elif self._cflux_method =="InputBoth":  #this is just the above modes put together
            if "node_input_flux" in self._node_attributes and "edge_input_flux" in self._edge_attributes:
                GetGraphAttributeToArray(Graph,'edge_input_flux')
                influxnode = np.empty_like(chanflux)
                for i,(u,v) in enumerate(self._edge_keys):
                    weights = {}
                    aw = 0
                    for key in Graph.succ[u]:
                        weights[key] = Graph.succ[u][key]['weight']
                        aw += weights[key]
                    influxnode[i] = Graph.nodes(data = "node_input_flux")[u] * weights[v]/aw
                self._cflux = influx + influx*self._length/2 + influxnode
                outflux = chanflux + influxedge*self._length/2
            else:
                raise ValueError(msg)    
        #write the new channel flux to the graph 
        SetGraphAttributeFromArray(Graph, self._cflux, self._edge_keys, 'channel_flux')
        # flux leaving the edge as QWout (an edge property)
        SetGraphAttributeFromArray(Graph, outflux, self._edge_keys, 'QWout')
        return Graph
    
    #The following are calculated each timestep regardless of steady state or dynamic
    
    def _calc_transport_cap_EH(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        #check attributes exist
        self.CheckAttributes(Graph)
        
        """method to calculate sediment transport capacity using the formulation of "Engelund and Hansen, 1967"""
        #first establish if we have QWin from a prior timestep - otherwise initialise as zero
        if "QWin" in self._edge_attributes:
            QWin = GetGraphAttributeToArray(Graph, 'QWin')
        else:
            QWin = np.zeros_like(self._length)            
        # add to channel flux
        try:
            flux = self._cflux+QWin
        except(ValueError):
            flux = QWin
        #get hydraulic parameters on the network
        if self._cflux_method =="FluxArea": #read channel area from the graph
            S = GetGraphAttributeToArray(Graph,'channel_area')
            Wc = 2*np.sin(self._beta/2)*np.sqrt(2*S/(self._beta-np.sin(self._beta)))
        else: #calculate channel area based on Darcy-Weisbach function
            Dh,S,Wc=DarcyWeisbach(self._DPhi,flux,beta=self._beta,fr=self._fr,rw=self._fluid_density,dhmin=self._dhmin)
        
        #calculate basal shear stress using function WaterShearStress
        Tau = WaterShearStress(self._cflux,S,fr=self._fr,rw=self._fluid_density)
        #effective density of sediment in water
        R_mean_active = (self._rhos_active-self._fluid_density)/self._fluid_density
        
        # Get sediment flux capacity for grain density and grain size
        self._QSc = 0.4/self._fr * 1/(self._d_median*R_mean_active**2*self._g**2)*(Tau/self._fluid_density)**(5.0/2.0) * Wc
        
        #where nan (e.g. due to divide by zero) assign the lowest value
        minQSc = np.nanmin(self._QSc)
        self._QSc = np.where(np.isnan(self._QSc),minQSc,self._QSc)
        
        # establish the virtual velocity using either eq 14 or 17 of Kloesch and Habersack
        #eq 14
        self._Uv = VirtualVelocity_1(Tau,self._d_median,self._rhos_active,rw = self._fluid_density, D50 = (2**-self._meanD)/1000, g = self._g)
        #eq 17
        #self._Uv = VirtualVelocity_2(Tau,self._d_median,self._rhos_active,rw = self._fluid_density, D50 = self._medianD, g = self._g)
        
        #assign these to the graph
        SetGraphAttributeFromArray(Graph, self._QSc, self._edge_keys, 'QSc')
        SetGraphAttributeFromArray(Graph, self._Uv, self._edge_keys, 'Uv')
        #SetGraphAttributeFromArray(Graph, S, self._edge_keys, 'S')
        return Graph
   
    def _calc_transport_cap_MPM(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
        """method to calculate sediment transport capacity using the formulation of "Meyer-Peter and Mueller, 1948"""
        #out[i] = 3.97* dir *sqrt(R*g*Dm^3*excess_shear^3)
        return Graph

    def _calc_till_mobilisation(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
       
        """method to calculate sediment mobilisation using the Sugset formulation of Delaney et al, 2019"""
        
        self._sedl = self._length*self._sedl_factor
        
        #read till thickness from graph or initialise as random value or zero
        if "till_thickness" in self._edge_attributes:
            TTedge = GetGraphAttributeToArray(Graph, 'till_thickness')
        else:
            l = lambda c: self._InitTill/2+np.random.rand()*self._InitTill
            TTedge = np.array([l(c) for c in self._length])

        #update till thickness on graph
        SetGraphAttributeFromArray(Graph, TTedge, self._edge_keys, 'till_thickness')

        # get 1D sediment flux into link from upstream node (from last timestep)
        if 'QSin' in self._edge_attributes:
            self._QSin = GetGraphAttributeToArray(Graph, 'QSin')
        else: #zero if first timestep
            self._QSin = np.zeros_like(self._length)

        #till mobilisation (from eq. 10 of Delaney et al 2019) 
        
        #for transport limited case
        mob = (self._QSc-self._QSin)/self._sedl
        
        #sediment needed to reach maximum possible till deposition
        if self._erosion_method == "MixedBed":
            #with mixed bed we cannot deposit more than the limit at once, but we can allow a thick layer to exist
            min_mob = -self._Hlim*(1-self._bed_porosity)*self._edgewidth/self._dt
        else:
            min_mob = (TTedge-self._Hlim)*(1-self._bed_porosity)*self._edgewidth/self._dt
        
        #We do not permit to over-fill the edge
        mob_r = np.where(mob < min_mob, min_mob, mob)
        
        if self._erosion_method == "InfiniteTill":
            #here we set to mobilisation with erosion 0
            self._dQSdx = mob_r
            self._mt = self._erod*0
        else:
            #sigmaH eq is incorrect in Delaney et al 2019 paper -- multiply by 5 (not divide) gives the correct form
            #see Delaney et al, 2023 for this formulation (with m not m^-1)
            sigmaH = (1+np.exp(10-5*TTedge/self._Dsig))**-1
        
            #conditional for till source term (eq 15 of Delaney) - new eroded sediment is from bedrock only; 
            #bedrock porosity is assumed zero
            self._mt = self._erod*(1-(TTedge/self._Hg))*self._edgewidth
            self._mt = np.where(self._mt > 0, self._mt,0.0) #we may in places get thickness greater that the limit so enforce positive mt
            SetGraphAttributeFromArray(Graph, self._mt, self._edge_keys, 'mt')
            
            #maximum till available to mobilise from new erosion and existing till
            #here we include porosity as we want a volume of grain only not pores
            max_mob = self._mt + TTedge*(1-self._bed_porosity)*self._edgewidth/self._dt
            
            # the channel can only mobilise till that exists and also the water cannot erode into bedrock
            mob_r = np.where(mob_r > max_mob, max_mob, mob_r)
            
            #mobilisation for mixed supply/transport limited case
            mob2 = mob_r*sigmaH + self._mt*(1-sigmaH)
            
            #select which rule to apply for flux gradient (per eq 10 Delaney 2019)
            conA = np.logical_and(mob_r <= 0,TTedge >= self._Hlim) #supply-in exceeds transport capacity AND no space for till
            conB = np.logical_and(conA == False, mob_r<= self._mt) #clearly transport limited case
            conC = np.logical_and(conA == False, conB == False) #transport or supply limited case
                    
            #report the con to the graph for QC
            whichcon1 = np.where(conC,'C','B')
            con = np.where(conA,'A',whichcon1)
            #SetGraphAttributeFromArray(Graph, con, self._edge_keys, 'con')
            
            #sediment flux gradient on edge
            self._dQSdx = conA*np.zeros_like(self._length) + conB*mob_r + conC*mob2
        
        #report to graph
        SetGraphAttributeFromArray(Graph, self._dQSdx, self._edge_keys, 'dQSdx')
        return Graph

    def _calc_till_transport(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
                
        """method to calculate transport of till on the network using a kinematic wave approach"""
        # A kinematic-wave transport model conserving volume is used see Newell 1993 Transport Research B vol 27B part I-III
        # active sediment flux is stored as a transient flux density k, in m^3/m
        # and we develop a 'jam' condition where this reaches a maximum
        # jams propagate upstream, and clear downstream - this ensures we never exceed flux capacity

        # if k doesn't exist already we initialise as zero
        if "flux_density" in self._edge_attributes:
            k = GetGraphAttributeToArray(Graph, 'flux_density')
        else:
            k = np.zeros_like(self._length)
        
        #draw a slice from random number array, if it exists, otherwise make an array
        rng = np.random.default_rng()
        if self._RNarray is not None:
            RNarray = ray.get(self._RNarray)
            # for the length of the input array we draw them in series
            if self._time_idx < self._RNarray_len:
                RNA = RNarray[self._time_idx] #num_edges by n
            else:
                #and for any subsequent we draw a slice randomly from the array
                idx = rng.integers(low = 0, high = len(RNarray))
                RNA = RNarray[idx] #num_edges by n
        else:
            RNA = rng.normal(size = (Graph.number_of_edges(),self._samp_n))
        
        #we define the distance possible to travel in time dt using the virtual velocity
        # for Engelund & Hansen we may find that QS > 0 but Uv is not > 0.
        # so as to avoid gridlock we use lmin to control this
        
        lmin = self._length * 0.01 # the minimum edge length to remove
        #lmin implies a Umin
        Umin = lmin/self._dt
        
        u = np.where(self._Uv > Umin,self._Uv,Umin)
        l = u*self._dt
        
        #xcrit is the point from which material downstream can exit the edge, but it cannot be longer than the edge length
        xcrit = np.where(l>self._length, self._length,l)
        
        #instead we might want to adaptively shorten the timestep to avoid this limitation
        #self._dt = np.where(l>self._length, self._length/u,self._dt)
        
        # maximum flux density is the flux capacity * dt over the length of link
        kmax = self._QSc*self._dt/self._length
        
        # if we exceed kmax, the link may be jammed - in this case we cannot add new sediment to the active transport
        # we allow k temporarily to exceed kmax though to conserve volume
        jammed = np.greater(k,kmax)
        # also we may exceed capacity with flux in
        constricted = np.greater(self._QSin,self._QSc)
        # thse define edges that do not hav free flow
        nonfree = np.logical_or(jammed==True,constricted==True)
        
        #report the jam conditions to the graph
        SetGraphAttributeFromArray(Graph, jammed, self._edge_keys, 'jammed')
       
        # for constricted edges we can only carry the max capacity in and the rest is excess flow, XQ
        XQcon = np.where(constricted, self._QSin-self._QSc,0.0)
        XQ = np.where(jammed, self._QSin,XQcon) # this came in, so needs later to be stored as till
        
        # in non free edges only negative till mobilisation (deposition) is allowed
        deposition_only = np.where(self._dQSdx < 0.0,self._dQSdx,0.0)
        self._dQSdx = np.where(nonfree, deposition_only, self._dQSdx)
        
        #now we deal with volumes. Volume components are:
        #VSinit - active sediment on edge at 'beginning' of timestep: detritus is 'as_edge'
        #VSin - volume entering edge nominally at 'beginning' of timestep: detritus is 'as_node'
            #XVSin - excess volume coming in - included in VSdep
        #VSnew - volume entering edge due to till mobilisation nominally at 'middle' of timestep
            #VSmob - subvolume of VSnew for mobilised from basal sediment: detritus is 'basal'
            #VSerod - subvolume of VSnew for mobilised from erosion: detritus is 'basement'
        #VS - active sediment on edge nominally at 'middle' of timestep: detritus is 'as_edge'
            #XVS - excess volume - included in VSdep
        #VSdep - volume leaving edge due to till demobilisation nominally at 'middle' of timestep: detritus is 'as_edge'
        #VSout - volume leaving edge, nominally at 'end' of timestep: detritus is 'as_edge'
        #VSback - 'backflow' from node at the end of the timestep, initially zero: detritus is 'as_edge'
        #VSfinal - volume that remains on edge at the end of the timestep: detritus is 'as_edge'
        
        #Get volume elemnts using function GetEdgeVolumes
        VolArrays =  GetEdgeVolumes(k, kmax, self._QSin, XQ,self._dQSdx, self._length, xcrit, self._mt, self._dt)
        #Output order is: VSinit,VSin,VSdep,VSerod,VSmob,VSout,VSback,VSfinal, XVS
        
        VSout = VolArrays[5]
        SetGraphAttributeFromArray(Graph, VSout, self._edge_keys, 'VSout')
        
        #Convert outgoing volume to flux and report back to graph
        QSout = VSout/self._dt
        SetGraphAttributeFromArray(Graph, QSout, self._edge_keys, 'QSout')
        
        #volumetric flux capacity
        VScap = np.where(np.isfinite(self._QSc),self._QSc*self._dt,0.0)
        SetGraphAttributeFromArray(Graph, VScap, self._edge_keys, 'VScap')
        
        #for each node we now assess the following:
        # total volume flux in from upstream edges vs total volume flux capacity of downstream edges
        # to conserve volume we must use volumes not Q as dt is not necessarily the same on all edges
        #get Vs in from all upstream edges, and max cap Vs out; also collate incoming grain size info 
        #arrays for volume to and from node 
        VStoNode = np.zeros_like(Graph.nodes(), dtype = float)
        VScfromNode = np.zeros_like(Graph.nodes(), dtype = float)
        # a list for the density distributions
        d_dists = [None]*len(Graph.nodes())
        #iterate through the nodes
        for t,u in enumerate(Graph.nodes()):
            vols = []
            dists = []
            for key in Graph.pred[u]:
                if Graph.nodes[key]['node_status'] > 0: #no outlet or floating nodes
                    v = Graph.pred[u][key]['VSout']
                    if np.isfinite(v):
                        vols.append(v) 
                    else: 
                        vols.append(0)
                    dist = Graph.pred[u][key]['d_distribution']
                    if np.isfinite(dist[0]) and np.isfinite(dist[1]):
                        dists.append(dist)
                    else: 
                        dists.append((self._meanD, self._stdD))
            VStoNode[t]=np.sum(vols)
            d_dists[t] = CombineDdistsArray(dists,vols/VStoNode[t],RArray = RNA[t],def_mean = self._meanD, def_std = self._stdD)[1]
            for key in Graph.succ[u]:
                if Graph.succ[u][key]['status'] != 0: #no constraint from outlet segments
                    qc = Graph.succ[u][key]['VScap']
                    if np.isfinite(qc):
                        qc = qc
                    else: 
                        qc = 0.0
                    jam = Graph.succ[u][key]['jammed']
                    if jam:
                        VScfromNode[t]+=0
                    else:
                        VScfromNode[t]+=qc

        # report nodal grain size to graph
        d_dist_node = {u:d_dists[i] for i,u in enumerate(self._node_keys)}
        nx.set_node_attributes(Graph,d_dist_node, 'd_dist_node')

        #if incoming volume exceeds capacity define a backflow
        Backflow = np.where(VStoNode > VScfromNode, VStoNode-VScfromNode, 0.0)
        #a = nx.get_node_attributes(Graph,'node_status')
        #node_status = np.array([a[key] for key in a])
        Out_nodes = [i for i,j in enumerate(Graph.succ) if len(Graph.succ[j])<1]
        Backflow[Out_nodes] = 0.0 #no backflow if the node has no succcessors
        Outflow = VStoNode-Backflow
        
        #Report Outflow to graph
        values = {u:Outflow[i] for i,u in enumerate(self._node_keys)}
        nx.set_node_attributes(Graph,values, 'VSo')
        
        #and now assign forward and backward volume flow to edges
        for t,u in enumerate(Graph.nodes()):
            if VStoNode[t]>0:
                BFR = Backflow[t]/VStoNode[t]
            else:
                BFR = 0.0 #no backflow if VSout is zero
            
            if VScfromNode[t]>0:
                VSR = Outflow[t]/VScfromNode[t]
            else:
                VSR = 0.0 #no outflow if capacity is zero
                
            #back flow to predecssor edges
            values = {(key,u): Graph.pred[u][key]['VSout']*BFR for key in Graph.pred[u]}
            nx.set_edge_attributes(Graph,values, 'VSback')
            # adjusted out flow from predecssor edges
            values = {(key,u): Graph.pred[u][key]['VSout']-Graph.pred[u][key]['VSout']*BFR for key in Graph.pred[u]}
            nx.set_edge_attributes(Graph,values, 'VSout')
            # inflow to succs for next iteration, but only to unjammed edges
            values = {(u,key): Graph.succ[u][key]['VScap']*VSR*(1-Graph.succ[u][key]['jammed']) for key in Graph.succ[u]}
            nx.set_edge_attributes(Graph,values,'inVS')
        
        # update arrays from graph
        VSout = GetGraphAttributeToArray(Graph, 'VSout')
        VolArrays[5] = VSout
        
        VSback = GetGraphAttributeToArray(Graph, 'VSback')
        VolArrays[6] = VSback
        
        a = nx.get_edge_attributes(Graph,'inVS')
        QSin = np.array([a[key] for key in a])/self._dt
         
        #set the QSin on the graph
        SetGraphAttributeFromArray(Graph, QSin, self._edge_keys, 'QSin')
        
        #Update VSfinal and flux density at end of timestep
        VSfinal = VolArrays[7]+VSback
        VolArrays[7] = VSfinal
        k = VSfinal/self._length
        
        #update flux density on the graph
        SetGraphAttributeFromArray(Graph, k, self._edge_keys, 'flux_density')
        
        #Now recalculate the grain size distribution for edges given the volumes in and out and residual sediment
        
        #new samples of global distribution with sample size n_samp
        newDs = [NewDdist(self._meanD, self._stdD, self._samp_n)[1] for i in self._edge_keys]
        #input distributions from nodes
        d_dist_in = [d_dist_node[u] for i,(u,v) in enumerate(self._edge_keys)]
        
        #volumetric proportions defind using function MixVols
        PVolArrays = MixVols(VolArrays)
        P_as_edge = PVolArrays[0]
        #SetGraphAttributeFromArray(Graph, P_as_edge, self._edge_keys, 'pV_edge')
        P_as_node = PVolArrays[1]
        #SetGraphAttributeFromArray(Graph, P_as_node, self._edge_keys, 'pV_node')
        P_basal = PVolArrays[2]
        #SetGraphAttributeFromArray(Graph, P_basal, self._edge_keys, 'pV_sed')
        P_basement = PVolArrays[3]
        #SetGraphAttributeFromArray(Graph, P_basement, self._edge_keys, 'pV_base')
        
        #volume proportions for each volume element
        volPs = [[P_as_edge[i],P_as_node[i],P_basal[i],P_basement[i]] for i,j in enumerate(self._edge_keys)]
        
        #dist for each volume element
        try:
            dists = [[self._d_dist[i],d_dist_in[i],self._sed_d_dist[i], newDs[i]] for i,j in enumerate(self._edge_keys)]
        except IndexError:
            raise("Index error accessing arrays")
        #get new edge distributions with function CombineDdists
        edge_dists = [CombineDdistsArray(dists[i],volPs[i],RArray = RNA[i],def_mean = self._meanD, def_std = self._stdD) for i,j in enumerate(self._edge_keys)]
        
        #get new grainsize distribution for the basal sediment layer
        #volume of till on edge:
        TTedge = GetGraphAttributeToArray(Graph, 'till_thickness')
        VT = TTedge*(1-self._bed_porosity)*self._edgewidth
        # volume deposited/mobilised - one should be zero
        VSdep = VolArrays[2]
        VSmob = VolArrays[4]

        # as a proportion
        Pdep = np.where(np.isfinite(VSdep/(VT+VSmob+VSdep)),VSdep/(VT+VSmob+VSdep),0.0)
        Pmob = np.where(np.isfinite(VSmob/(VT+VSmob+VSdep)),VSmob/(VT+VSmob+VSdep),0.0)
        Pbas = np.where(np.isfinite(VT/(VT+VSmob+VSdep)),VT/(VT+VSmob+VSdep),0.0)
        
        #first we need to add VSdep to VT
        volPs = [[Pdep[i],Pbas[i]] for i,j in enumerate(self._edge_keys)]
        dists = [[edge_dists[i][1], self._sed_d_dist[i]] for i,j in enumerate(self._edge_keys)]
        basal_dists = [CombineDdistsArray(dists[i],volPs[i],RArray = RNA[i],def_mean = self._meanD, def_std = self._stdD) for i,j in enumerate(self._edge_keys)]
        
        #then we need to remove VSmob from the combination
        volPs = [[Pbas[i]+Pdep[i],Pmob[i]] for i,j in enumerate(self._edge_keys)]
        dists = [[basal_dists[i][1],dists[i][0]] for i,j in enumerate(self._edge_keys)]
        basal_dists = [ExtractDdistsArray(dists[i],volPs[i],RArray = RNA[i],def_mean = self._meanD, def_std = self._stdD) for i,j in enumerate(self._edge_keys)]
        
        #report to graph
        #edge
        a = {(u,v): edge_dists[i][0] for i,(u,v) in enumerate(self._edge_keys)}
        nx.set_edge_attributes(Graph,a,'d_median')
        a = {(u,v): edge_dists[i][1] for i,(u,v) in enumerate(self._edge_keys)}
        nx.set_edge_attributes(Graph,a,'d_distribution')
        #basal
        a = {(u,v): basal_dists[i][0] for i,(u,v) in enumerate(self._edge_keys)}
        nx.set_edge_attributes(Graph,a,'sed_d_median')
        a = {(u,v): basal_dists[i][1] for i,(u,v) in enumerate(self._edge_keys)}
        nx.set_edge_attributes(Graph,a,'sed_d_distribution')
        return Graph, VolArrays, PVolArrays


    def _calc_till_transport_lite(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
                
        """method to calculate transport of till on the network using a kinematic wave approach"""
        # A kinematic-wave transport model conserving volume is used see Newell 1993 Transport Research B vol 27B part I-III
        # active sediment flux is stored as a transient flux density k, in m^3/m
        # and we develop a 'jam' condition where this reaches a maximum
        # jams propagate upstream, and clear downstream - this ensures we never exceed flux capacity

        # if k doesn't exist already we initialise as zero
        if "flux_density" in self._edge_attributes:
            k = GetGraphAttributeToArray(Graph, 'flux_density')
        else:
            k = np.zeros_like(self._length)
        
        #we define the distance possible to travel in time dt using the virtual velocity
        # for Engelund & Hansen we may find that QS > 0 but Uv is not > 0.
        # so as to avoid gridlock we use lmin to control this
        
        lmin = self._length * 0.01 # the minimum edge length to remove
        #lmin implies a Umin
        Umin = lmin/self._dt
        
        u = np.where(self._Uv > Umin,self._Uv,Umin)
        l = u*self._dt
        
        #xcrit is the point from which material downstream can exit the edge, but it cannot be longer than the edge length
        xcrit = np.where(l>self._length, self._length,l)
        
        #instead we might want to adaptively shorten the timestep to avoid this limitation
        #self._dt = np.where(l>self._length, self._length/u,self._dt)
        
        # maximum flux density is the flux capacity * dt over the length of link
        kmax = self._QSc*self._dt/self._length
        
        # if we exceed kmax, the link may be jammed - in this case we cannot add new sediment to the active transport
        # we allow k temporarily to exceed kmax though to conserve volume
        jammed = np.greater(k,kmax)
        # also we may exceed capacity with flux in
        constricted = np.greater(self._QSin,self._QSc)
        # thse define edges that do not hav free flow
        nonfree = np.logical_or(jammed==True,constricted==True)
        
        #report the jam conditions to the graph
        SetGraphAttributeFromArray(Graph, jammed, self._edge_keys, 'jammed')
       
        # for constricted edges we can only carry the max capacity in and the rest is excess flow, XQ
        XQcon = np.where(constricted, self._QSin-self._QSc,0.0)
        XQ = np.where(jammed, self._QSin,XQcon) # this came in, so needs later to be stored as till
        
        # in non free edges only negative till mobilisation (deposition) is allowed
        deposition_only = np.where(self._dQSdx < 0.0,self._dQSdx,0.0)
        self._dQSdx = np.where(nonfree, deposition_only, self._dQSdx)
        
        #now we deal with volumes. Volume components are:
        #VSinit - active sediment on edge at 'beginning' of timestep: detritus is 'as_edge'
        #VSin - volume entering edge nominally at 'beginning' of timestep: detritus is 'as_node'
            #XVSin - excess volume coming in - included in VSdep
        #VSnew - volume entering edge due to till mobilisation nominally at 'middle' of timestep
            #VSmob - subvolume of VSnew for mobilised from basal sediment: detritus is 'basal'
            #VSerod - subvolume of VSnew for mobilised from erosion: detritus is 'basement'
        #VS - active sediment on edge nominally at 'middle' of timestep: detritus is 'as_edge'
            #XVS - excess volume - included in VSdep
        #VSdep - volume leaving edge due to till demobilisation nominally at 'middle' of timestep: detritus is 'as_edge'
        #VSout - volume leaving edge, nominally at 'end' of timestep: detritus is 'as_edge'
        #VSback - 'backflow' from node at the end of the timestep, initially zero: detritus is 'as_edge'
        #VSfinal - volume that remains on edge at the end of the timestep: detritus is 'as_edge'
        
        #Get volume elemnts using function GetEdgeVolumes
        VolArrays =  GetEdgeVolumes(k, kmax, self._QSin, XQ,self._dQSdx, self._length, xcrit, self._mt, self._dt)
        #Output order is: VSinit,VSin,VSdep,VSerod,VSmob,VSout,VSback,VSfinal, XVS
        
        VSout = VolArrays[5]
        SetGraphAttributeFromArray(Graph, VSout, self._edge_keys, 'VSout')
        
        #Convert outgoing volume to flux and report back to graph
        #QSout = VSout/self._dt
        #SetGraphAttributeFromArray(Graph, QSout, self._edge_keys, 'QSout')
        
        #volumetric flux capacity
        VScap = np.where(np.isfinite(self._QSc),self._QSc*self._dt,0.0)
        SetGraphAttributeFromArray(Graph, VScap, self._edge_keys, 'VScap')
        
        #for each node we now assess the following:
        # total volume flux in from upstream edges vs total volume flux capacity of downstream edges
        # to conserve volume we must use volumes not Q as dt is not necessarily the same on all edges
        #get Vs in from all upstream edges, and max cap Vs out; also collate incoming grain size info 
        #arrays for volume to and from node 
        VStoNode = np.zeros_like(Graph.nodes(), dtype = float)
        VScfromNode = np.zeros_like(Graph.nodes(), dtype = float)
        #iterate through the nodes
        for t,u in enumerate(Graph.nodes()):
            vols = []
            for key in Graph.pred[u]:
                if Graph.nodes[key]['node_status'] > 0: #no outlet or floating nodes
                    v = Graph.pred[u][key]['VSout']
                    if np.isfinite(v):
                        vols.append(v) 
                    else: 
                        vols.append(0)
            VStoNode[t]=np.sum(vols)
            for key in Graph.succ[u]:
                if Graph.succ[u][key]['status'] != 0: #no constraint from outlet segments
                    qc = Graph.succ[u][key]['VScap']
                    if np.isfinite(qc):
                        qc = qc
                    else: 
                        qc = 0.0
                    jam = Graph.succ[u][key]['jammed']
                    if jam:
                        VScfromNode[t]+=0
                    else:
                        VScfromNode[t]+=qc

        #if incoming volume exceeds capacity define a backflow
        Backflow = np.where(VStoNode > VScfromNode, VStoNode-VScfromNode, 0.0)
        #a = nx.get_node_attributes(Graph,'node_status')
        #node_status = np.array([a[key] for key in a])
        #Out_nodes = np.where(node_status <= 0)[0]
        Out_nodes = [i for i,j in enumerate(Graph.succ) if len(Graph.succ[j])<1]
        Backflow[Out_nodes] = 0.0 #no backflow if the node is an outlet node or sink node
        Outflow = VStoNode-Backflow
        
        #Report Outflow to graph
        values = {u:Outflow[i] for i,u in enumerate(self._node_keys)}
        nx.set_node_attributes(Graph,values, 'VSo')
        
        #and now assign forward and backward volume flow to edges
        for t,u in enumerate(Graph.nodes()):
            if VStoNode[t]>0:
                BFR = Backflow[t]/VStoNode[t]
            else:
                BFR = 0.0 #no backflow if VSout is zero
            
            if VScfromNode[t]>0:
                VSR = Outflow[t]/VScfromNode[t]
            else:
                VSR = 0.0 #no outflow if capacity is zero
                
            #back flow to predecssor edges
            values = {(key,u): Graph.pred[u][key]['VSout']*BFR for key in Graph.pred[u]}
            nx.set_edge_attributes(Graph,values, 'VSback')
            # adjusted out flow from predecssor edges
            values = {(key,u): Graph.pred[u][key]['VSout']-Graph.pred[u][key]['VSout']*BFR for key in Graph.pred[u]}
            nx.set_edge_attributes(Graph,values, 'VSout')
            # inflow to succs for next iteration, but only to unjammed edges
            values = {(u,key): Graph.succ[u][key]['VScap']*VSR*(1-Graph.succ[u][key]['jammed']) for key in Graph.succ[u]}
            nx.set_edge_attributes(Graph,values,'inVS')
        
        # update arrays from graph
        VSout = GetGraphAttributeToArray(Graph, 'VSout')
        VolArrays[5] = VSout
        
        VSback = GetGraphAttributeToArray(Graph, 'VSback')
        VolArrays[6] = VSback
        
        a = nx.get_edge_attributes(Graph,'inVS')
        QSin = np.array([a[key] for key in a])/self._dt
         
        #set the QSin on the graph
        SetGraphAttributeFromArray(Graph, QSin, self._edge_keys, 'QSin')
        
        #Update VSfinal and flux density at end of timestep
        VSfinal = VolArrays[7]+VSback
        VolArrays[7] = VSfinal
        k = VSfinal/self._length
        
        #update flux density on the graph
        SetGraphAttributeFromArray(Graph, k, self._edge_keys, 'flux_density')
        return Graph,VolArrays

    def _calc_Exner_equation(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
           
        """method to calculate evolving till thickness according to Exner Equation"""
        #formulation equivalent to Delaney's Julia code - but here with porosity
        dHdt = (-self._dQSdx/(1-self._bed_porosity)+self._mt/(1-self._bed_porosity))/self._edgewidth
        
        #excess volume as height
        XVS = self._VolArrays[8]
        XVasH = XVS/(self._length*self._edgewidth)
        
        #bedrock lowering from erosion - no porosity
        dBRE = -self._mt*self._dt/self._edgewidth
        
        #report to graph
        SetGraphAttributeFromArray(Graph, dHdt, self._edge_keys, 'dHdt')
        SetGraphAttributeFromArray(Graph, dBRE, self._edge_keys, 'dBRE')
        
        NewTill = self._dt*dHdt+XVasH/(1-self._bed_porosity)
        
        #update till thickness on graph
        TT = GetGraphAttributeToArray(Graph, 'till_thickness')+NewTill
        SetGraphAttributeFromArray(Graph, TT, self._edge_keys, 'till_thickness')
        
        #update downwind node only with mean till thickness of upwind links
        #avoids excess till and upwinding of changes in till height
        NBE = np.zeros_like(Graph.nodes(), dtype = float)
        NTT = np.zeros_like(Graph.nodes(), dtype = float)
        for i,u in enumerate(Graph.nodes()):
            tt,db = 0.0,0.0
            n,m = 0,0
            for key in Graph.pred[u]:
                t = Graph.pred[u][key]['till_thickness']
                b = Graph.pred[u][key]['dBRE']
                if np.isnan(t):
                    tt+=0
                else:
                    tt+=t
                    n+=1
                if np.isnan(b):
                    db+=0
                else:
                    db+=b
                    m+=1
            if n> 0:
                NTT[i] = tt/n
                NBE[i] = db/m

        #set new higher bed elevation for positive NTT, lower for negative NTT
        if "bed_elevation" in self._node_attributes:
            a = nx.get_node_attributes(Graph,'bed_elevation')
            BE = np.array([a[key] for key in a])
        else:
            BE = np.zeros_like(Graph.nodes(), dtype= float)
        if "bedrock_elevation" in self._node_attributes:
            a = nx.get_node_attributes(Graph,'bedrock_elevation')
            BRE = np.array([a[key] for key in a])
        else: 
            BRE = BE
        
        BRE += NBE
        NBE = np.where(NTT>=0.0,BRE+NTT,BRE)
        NBRE = np.where(NTT<0.0,BRE+NTT,BRE)
 
        #update till thickness on graph - excluding values < 0
        newTT = np.where(TT < 0, 0.0, TT)
        SetGraphAttributeFromArray(Graph, newTT, self._edge_keys, 'till_thickness')
        
        #update bed elevation
        values = {u:NBE[i] for i,u in enumerate(Graph.nodes())}
        nx.set_node_attributes(Graph,values, 'bed_elevation')

        #update bedrock elevation
        values = {u:NBRE[i] for i,u in enumerate(Graph.nodes())}
        nx.set_node_attributes(Graph,values, 'bedrock_elevation')
        return Graph

    def _calc_channel_flux_to_node(self, Graph = None):
        
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
        
        """method to calculate flux into the nodes from all upstream edgs and return value to downstream edges"""
        #water flow rate and volume
        QW = np.zeros_like(Graph.nodes(), dtype = float)
        VW = np.zeros_like(Graph.nodes(), dtype = float)
        for i,u in enumerate(Graph.nodes()):
            for key in Graph.pred[u]:
                v = Graph.pred[u][key]['QWout']
                if np.isnan(v):
                    QW[i] += 0
                else:
                    QW[i] += v
                try:
                    dt = self._time - Graph.pred[u][key]['last_time']
                except KeyError:
                    dt = self._time
                if np.isnan(v):
                    VW[i] += 0
                else:
                    VW[i] += v*dt    
        #report these to graph
        values = {u:QW[i] for i,u in enumerate(Graph.nodes())}
        nx.set_node_attributes(Graph,values, 'QW')
        values = {u:VW[i] for i,u in enumerate(Graph.nodes())}
        nx.set_node_attributes(Graph,values, 'VW')
        #and from this nodal value we distribute as input to the downstream edges
        fluxnode = np.zeros_like(self._length)
        #flux at the node may go to several edges so we reweight between those
        for i,(u,v) in enumerate(self._edge_keys):
            weights = {}
            aw = 0
            for key in Graph.succ[u]:
                weights[key] = 1-Graph.succ[u][key]['weight']
                aw += weights[key]
            fluxnode[i] = Graph.nodes(data = "QW")[u] * weights[v]/aw
        # we remove the original channel flux value
        QW_influx = fluxnode-self._cflux
        #if direct methods we set this to zero as we do not wish it to change
        if self._cflux_method =="Flux" or self._cflux_method == "FluxArea":
            values = {(u,v): 0 for i,(u,v) in enumerate(self._edge_keys)}
            nx.set_edge_attributes(Graph,values, 'QWin')
        #else we pass the value to the graph for the next timestep
        else:
            SetGraphAttributeFromArray(Graph, QW_influx, self._edge_keys, 'QWin')
        return Graph

    def _DetritusTracking(self, Graph = None):
        if Graph == None:
            Graph = ray.get(self._graph)
        
        #check attributes exist
        self.CheckAttributes(Graph)
        
        """method to track detritus on the network as a passive tracer properties"""
        #"None" will avoid these calculations without any impact on model evolution
        if self._detritus_method != "None":
            # if there is not a detritus property (dict) on the edge initialise with class 'init' and value 1
            if "detritus" in self._edge_attributes:
                det = nx.get_edge_attributes(Graph,'detritus')
            else:
                det = {(u,v): {'init':1} for i,(u,v) in enumerate(self._edge_keys)}
            #we also need a property on the nodes
            if "detritus_node" in self._node_attributes:
                det_n = nx.get_node_attributes(Graph,'detritus_node')
            else:
                det_n = {u: {'init':1} for i,u in enumerate(self._node_keys)}
            #and a property for the basal sediment layer
            if "sed_detritus" in self._edge_attributes:
                sed_det = nx.get_edge_attributes(Graph,'sed_detritus')
            else:
                sed_det = {(u,v): {'init':1} for i,(u,v) in enumerate(self._edge_keys)}
            
            #proportions on edge
            P_as_edge = self._PVolArrays[0]
            P_as_node = self._PVolArrays[1]
            P_basal = self._PVolArrays[2]
            P_basement = self._PVolArrays[3]
            #total proportion should be 1
            TP = P_as_edge+P_as_node+P_basal+P_basement
            
            #valid methods are 'SedErod','NodeProp'
            #SedErod separates mobilisation of till (class 'basal') from 'fresh' erosion of bedrock (class 'basement') 
            det_ps = {}
            if self._detritus_method == "SedErod":
                try:
                    Dclasses = Graph.graph['DetClasses']
                except KeyError:
                    Dclasses = ['init','basal','basement']
                for i,key in enumerate(self._edge_keys):
                    if np.abs(1-TP[i]) > 0.01:
                        print('WARNING: edge {} has total probability {} not adding to 1, using init, but check for NANs or negative volumes'.format(key,TP[i]))
                        ps = np.array([1,0,0])
                    else:
                        Basal = np.array([0,1,0])* P_basal[i]  # mobilised from the bed
                        Basement = np.array([0,0,1])* P_basement[i]
                        try:
                            As_Edge = np.array([det[key][Dclass] for Dclass in Dclasses])*P_as_edge[i]
                            As_Node = np.array([det_n[key[0]][Dclass] for Dclass in Dclasses])*P_as_node[i]
                        except KeyError:
                            #here we consider missing keys as 'init' because we don't want to keep the details
                            As_Edge = np.array([1,0,0]) * P_as_edge[i]
                            As_Node =  np.array([1,0,0]) * P_as_node[i]
                        ps = Basal + Basement + As_Edge + As_Node
                    det_ps[key]= {Dclasses[0]:ps[0],Dclasses[1]:ps[1],Dclasses[2]:ps[2]}

            #NodeProp uses bedrock classes defined on nodes
            elif self._detritus_method == "NodeProp":
                sed_det_ps = {}
                if "detritus_prop" in self._node_attributes:
                    prop = nx.get_node_attributes(Graph,'detritus_prop')
                else: 
                    raise ValueError('property "detritus_prop" is not a node attribute')
                try:
                    Dclasses = Graph.graph['DetClasses']
                except KeyError:
                    # if not given we find those of the active graph...
                    Dclasses = ['init','basal','basement']+list(set([prop[n] for n in prop]))
                    #print('Det classes {}'.format(Dclasses))
                # work out detritus on the edge
                for i,key in enumerate(self._edge_keys):
                    #add any unrepresented classes to Dclasses
                    classlist = list(sed_det[key].keys())
                    newclasses = [item for item in classlist if item not in Dclasses]
                    if len(newclasses) > 0:
                        Dclasses += newclasses
                        #print('added Det classes from basal {}'.format(newclasses))
                    classlist = list(det[key].keys())
                    newclasses = [item for item in classlist if item not in Dclasses]
                    if len(newclasses) > 0:
                        Dclasses += newclasses
                        #print('added Det classes from edge {}'.format(Dclasses))
                    classlist = list(det_n[key[0]].keys())
                    newclasses = [item for item in classlist if item not in Dclasses]
                    if len(newclasses) > 0:
                        Dclasses += newclasses
                        #print('added Det classes from node {}'.format(newclasses))
                    #get detritus prop at upstream and downstream nodes
                    DPu, DPv = prop[key[0]],prop[key[1]] 
                    #are these the same?
                    if DPu == DPv:
                        l = np.array([1 if b == DPu else 0 for b in Dclasses])
                        Basement = l * P_basement[i]
                    else:
                        l = np.array([0.5 if b == DPu or b == DPv else 0 for b in Dclasses])
                        Basement = l * P_basement[i]
                    try: #sediment remobilisation uses the proportions already in the till layer
                        Basal = np.array([sed_det[key][Dclass] for Dclass in Dclasses])*P_basal[i]
                    except KeyError:
                        Basal = np.array([1 if b == 'basal' else 0 for b in Dclasses])*P_basal[i]
                    try:
                        As_Edge = np.array([det[key][Dclass] for Dclass in Dclasses])*P_as_edge[i]
                    except KeyError:
                        As_Edge = np.array([1 if b == 'init' else 0 for b in Dclasses])*P_as_edge[i]
                    try:
                        As_Node = np.array([det_n[key[0]][Dclass] for Dclass in Dclasses])*P_as_node[i]   
                    except KeyError:
                        As_Node =  np.array([1 if b == 'init' else 0 for b in Dclasses])*P_as_node[i]
                    ps = Basal + Basement + As_Edge + As_Node
                    #new local dict for data
                    det_ps[key] = {n: ps[m] for m,n in enumerate(Dclasses)}

                #detritus of basal sediment layer
                TTedge = GetGraphAttributeToArray(Graph, 'till_thickness')
                VT = TTedge*(1-self._bed_porosity)*self._edgewidth
                VSdep = self._VolArrays[2]
                Pdep = np.where(np.isfinite(VSdep/(VT+VSdep)),VSdep/(VT+VSdep),0.0)
            
                for i,key in enumerate(self._edge_keys):
                    sed_det_ps[key] = {}
                    ToSed = np.array([det_ps[key][Dclass] for Dclass in Dclasses])*Pdep[i]
                    try:
                        Sed = np.array([sed_det[key][Dclass] for Dclass in Dclasses])*(1-Pdep[i])
                    except KeyError:
                        Sed = np.array([1 if b == 'basal' else 0 for b in Dclasses])*(1-Pdep[i])
                    sedps = ToSed+Sed
                    for k,l in enumerate(Dclasses):
                        sed_det_ps[key][l] = sedps[k]
                #report this back to the graph
                nx.set_edge_attributes(Graph,sed_det_ps, 'sed_detritus')            
            
            #report this back to the graph
            nx.set_edge_attributes(Graph,det_ps, 'detritus')
            
            #now we need to assign values for outgoing sediment to the nodes        
            det_node = {}
            for t,u in enumerate(Graph.nodes()):
                det_node[u] = det_n[u]
                vols = []
                dtps = []
                for key in Graph.pred[u]:
                    if Graph.nodes[key]['node_status'] > 0: #no input from outlet or floating nodes
                        v = Graph.pred[u][key]['VSout']
                        if np.isfinite(v):
                            vols.append(v) 
                        else: 
                            vols.append(0.0)
                        dtp = Graph.pred[u][key]['detritus']
                        dtps.append(dtp)
                    V=np.nansum(vols)
                    if V > 0.0:
                        ps = np.zeros(len(Dclasses))
                        for i,j in enumerate(dtps):
                            vals = np.array([j[key] for key in j])*vols[i]/V
                            ps += vals
                        d = {}
                        for k,l in enumerate(Dclasses):
                            d[l] = ps[k]
                        det_node[u] = d
                        
            #report this back to the graph
            nx.set_node_attributes(Graph,det_node, 'detritus_node')
        return Graph

    def _update_time(self,time):
        self._time = time
        self._time_idx += 1
        
    def _reset_graph_time(self, Graph):
        #record last_time on subgraph edges
        SetGraphAttributeFromArray(Graph, self._time, self._edge_keys, 'last_time')
        return Graph

#on a 'steady' setup we have steady state inputs for hydraulic potential (or relevant inputs), hydrology input and erosion rate 
#thus we can calculate these once only

    def initialise_steady(self):
        #get graph attribute list
        G,self._dt = self._get_graph_atts()
        #recalculate updated Hydraulic Potential Gradient for network grid
        G = self._calc_HydraulicPotentialGradient(Graph = G)
        #calculate hydrology flow for network grid
        G = self._calc_channel_flux_on_edge(Graph = G)
        G = self._calc_channel_flux_to_node(Graph = G)
        #calculate erosion rate
        G = self._calc_erosion_rate(Graph = G)
        self._graph = ray.put(G)

    def initialise_steady_parallel(self, n_procs = os.cpu_count()):
        #this works itself but does not return 'self' parmaters from tasks
        #these are: self._cflux 
        #get SubGraphs
        Graph = ray.get(self._graph)
        E_Graphs = SplitGraphbyEdges(Graph, list(Graph.edges), processes=n_procs)
        def task(G):
            #get graph attribute list
            self._get_graph_atts(Graph = G)
            #recalculate updated Hydraulic Potential Gradient for network grid
            G = self._calc_HydraulicPotentialGradient(Graph = G)
            #calculate link flow for network grid
            G = self._calc_channel_flux_on_edge(Graph = G)
            #calculate erosion rate
            G = self._calc_erosion_rate(Graph = G)
            return G
        def update_main(Graph,G):
            edges = G.edges(data = True)
            Graph.add_edges_from(edges)
            return Graph
        @ray.remote
        def run_task(G):
            return task(G)
        results = [run_task.remote(E_Graphs[i]) for i in range(len(E_Graphs))]
        while len(results):
           done_id,results = ray.wait(results)
           Graph = update_main(Graph,ray.get(done_id[0]))
        self._graph = ray.put(Graph)
    
    # for timesteps we resolve the downstream flow of water and sediment
    def run_one_step_steady(self, time, lite = False):
        if type(time) is not int:
            time = int(time)
        #get new time
        self._update_time(time)
        #get graph attribute list for each iteration
        G,self._dt = self._get_graph_atts() #issue is here?
        #change density and grain size information
        G = self._calculate_edge_d_and_rhos(Graph = G)
        #calculate transport capacity
        G = self._update_transport(Graph = G)
        #calculate the till mobilisation
        G = self._calc_till_mobilisation(Graph = G)
        #calculate the till transport
        if lite:
            G, self._VolArrays = self._calc_till_transport_lite(Graph = G)
        else:
            G, self._VolArrays, self._PVolArrays = self._calc_till_transport(Graph = G)
        #calculate Exner equation
        G = self._calc_Exner_equation(Graph = G)
        #track detritus
        G = self._DetritusTracking(Graph = G)
        #calculate hydrology flow accummulation for next timestep
        self._calc_channel_flux_to_node(Graph = G)
        #reset graph time
        G = self._reset_graph_time(Graph = G)
        return G

    def get_graph(self):
        Graph = ray.get(self._graph)
        return(Graph)
    
    def set_graph(self, Graph):
            self._graph = ray.put(Graph)
    
    def CheckAttributes(self,Graph):
        # check for existence of necessary attributes not defined in __init__
        # if not found, will read from graph -- if not on graph an empty array is returned
         try:
             self._d_median = getattr(self,'_d_median')
         except AttributeError:
             self._d_median = GetGraphAttributeToArray(Graph, 'd_median')
         try:
             self._rhos_active = getattr(self,'_rhos_active')
         except AttributeError:
             self._rhos_active = self._rhos_grain
         try:
             self._DPhi = getattr(self,'_DPhi')
         except AttributeError:
             self._DPhi = GetGraphAttributeToArray(Graph, 'DPhi')
         try:
             self._cflux = getattr(self,'_cflux')
         except AttributeError:
             self._cflux = GetGraphAttributeToArray(Graph, 'channel_flux')
         try:
             self._mt = getattr(self,'_mt')
         except AttributeError:
            self._mt = GetGraphAttributeToArray(Graph, 'mt')
         try:
             self._QSc = getattr(self,'_QSc')
         except AttributeError:
             self._QSc = GetGraphAttributeToArray(Graph, 'QSc')
         try:
             self._QSin = getattr(self,'_QSin')
         except AttributeError:
             self._QSin = GetGraphAttributeToArray(Graph, 'QSin')
         try:
             self._Uv = getattr(self,'_Uv')
         except AttributeError:
             self._Uv = GetGraphAttributeToArray(Graph, 'Uv')    
         try:
             self._dQSdx = getattr(self,'_dQSdx')
         except AttributeError:
             self._dQSdx = GetGraphAttributeToArray(Graph, 'dQSdx')
         try:
             self._d_dist = getattr(self,'_d_dist')
         except AttributeError:
             self._d_dist = GetGraphAttributeToArray(Graph, 'd_distribution')
         try:
             self._sed_d_dist = getattr(self,'_sed_d_dist')
         except AttributeError:
             self.sed_d_dist = GetGraphAttributeToArray(Graph, 'sed_d_distribution')
         try:
             self._erod = getattr(self,'_erod')
         except AttributeError:
             self._erod = GetGraphAttributeToArray(Graph, 'erosion_potential')  

    # or for dynamic time-variable inputs we do everything every timestep
    def run_one_step_dynamic(self, time):
        #get new time
        self._update_time(time)
        #get graph attribute list
        self._get_graph_atts()
        #recalculate updated Hydraulic Potential Gradient for network grid
        self._calc_HydraulicPotentialGradient()
        #calculate link flow for network grid
        self._calc_channel_flux_on_edge()
        #calculate erosion rate
        self._calc_erosion_rate()
        #change density and grain size information
        self._calculate_edge_d_and_rhos()
        #calculate transport capacity
        self._update_transport()
        #calculate the till mobilisation
        self._calc_till_mobilisation()
        #calculate the till transport
        self._calc_till_transport()
        #calculate Exner equation
        self._calc_Exner_equation()
        #track detritus
        self._DetritusTracking()
        #calculate hydrology flow accummulation for next timestep
        self._calc_channel_flux_to_node()
        #reset graph time
        self._reset_graph_time()

@ray.remote(max_restarts = 0)
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
        nx.set_node_attributes(self._graph,-1, "partition_ID")
        nx.set_edge_attributes(self._graph,-1, "partition_ID")
        for key in Comms.keys():
            SG = self._graph.subgraph(Comms[key])    
            nx.set_edge_attributes(SG,key, "partition_ID")
            nx.set_node_attributes(SG,key, "partition_ID")

    def PartitionGraph(self, p_prop='partition_ID'):
        p_IDs = nx.get_node_attributes(self._graph,p_prop)
        ID_array = np.array([p_IDs[key] for key in p_IDs])
        self._IDs = np.unique(ID_array)
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
            rng = np.random.default_rng()
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

def SetGraphAttributeFromArray(Graph, array, edge_keys, attribute):
    try:
        values = {(u,v): array[i] for i,(u,v) in enumerate(edge_keys)}
    except (TypeError,IndexError):
        values = {(u,v): array for i,(u,v) in enumerate(edge_keys)}
    nx.set_edge_attributes(Graph,values,attribute)

def GetGraphAttributeToArray(Graph, attribute):
    a = nx.get_edge_attributes(Graph,attribute)
    arr = np.array([a[key] for key in a])
    return arr
    
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

def DarcyWeisbach(DPhi,Qw,beta=np.pi/6,fr=0.015,rw=1000.0,dhmin=0.0):
    """method to calculate channel geometries with Darcy-Weisbach equation"""
    #Darcy-Weisbach formula factor
    s = 2*(beta-np.sin(beta))/(beta/2+np.sin(beta/2))**2
    P = s*fr*rw
    #hydraulic diameter
    Dh = (P*Qw**2/np.abs(DPhi))**0.2
    Dh = np.where(Dh < dhmin,dhmin,Dh)
    #channel section area
    S = Dh**2/2*(beta/2+np.sin(beta/2))**2/(beta-np.sin(beta))
    #channel width
    wc = 2*np.sin(beta/2)*np.sqrt(2*S/(beta-np.sin(beta)))
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
    Vu = np.where(Tau_star >= Tau_c_star,a*((rs-rw)*g*d/rw)**0.5*(Tau_star-Tau_c_star)*((Tau_star)**0.5-(Tau_c_star)**0.5),Vmin)
    return Vu

def VirtualVelocity_2(Tau,d,rs,rw=1000.0, a = 0.96, b =1.5, D50 = 0.11, g = 9.81, Vmin = 0.0):
    """method to calculate virtual velocity using equation 17 from Kloesch and Habersack 2018"""
    Scale = (rs-rw)*g*d
    Tau_star = Tau/Scale
    Tau_c_star = 0.055*(d/D50)**-0.83
    Vu = np.where(Tau_star >= Tau_c_star,a*((rs-rw)*g*d/rw)**0.5*(Tau_star-Tau_c_star)**b,Vmin)
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
    VSdep = np.where(dQSdx<0.0, -dQSdx*length*dt,0.0) #deposition to basal sediment layer
    VSnew = dQSdx*length*dt+VSdep #mobilisation of sediment
    #now the balance
    VS = VSinit+VSin+VSnew-VSdep
    #active sediment must be greater than zero
    VS = np.where(VS>0,VS,0.0)
    VS = np.where(VS<kmax*length,VS,kmax*length) #limit by the maximum capacity of edge to have active sediment
    XVS = np.where(VS>kmax*length,VS-kmax*length,0.0) + XVSin
    VSdep += XVS  #excess volume deposited to basal sediment layer
    # for detritus and grain size we want to distinguish erosion and remobilisation
    conB = mt-dQSdx #con B
    VSerod = np.where(conB > 0, VSnew, mt*length*dt)
    # mobilised till from the basal sediment layer
    VSmob = VSnew-VSerod # will be zero except where con B is < 0 where it should be positive
    if np.nanmin(VSmob) < 0:
        print ('negative volume found {} in VSmob, assigning excess to VSdep'.format(np.nanmin(VSnew)))
        VSdep = np.where(VSmob < 0, -VSmob+VSdep,VSdep)
        VSmob = np.where(VSmob < 0, 0.0,VSmob)
    #At end of the timestep
    VSout = VS*xcrit/length #sediment leaving the edge
    VSback = np.zeros_like(VS)
    VSfinal = np.where(VS-VSout>0,VS-VSout,0.0) #sediment left on the edge
    return [VSinit,VSin,VSdep,VSerod,VSmob,VSout,VSback,VSfinal, XVS]

def MixVols(VolArrays):
    """method to calculate volumetric mixtures on edges"""
    VSinit = VolArrays[0] #as_edge
    VSin= VolArrays[1] #as_node
    VSerod= VolArrays[3] #basement
    VSmob= VolArrays[4] #basal
    #new arrays for volume elements
    V_as_edge = np.zeros_like(VSinit)
    V_as_node = np.zeros_like(VSinit)
    V_basal = np.zeros_like(VSinit)
    V_basement = np.zeros_like(VSinit)
    #begin timestep
    V_as_edge += VSinit
    V_as_node += VSin
    #middle timestep
    V_basal += VSmob
    V_basement += VSerod
    V_Total = V_as_edge + V_as_node +V_basal + V_basement
    #V proportions -- we assume if V is 0 at this stage that nothing is happening
    P_as_edge = np.where(V_Total > 0,V_as_edge/V_Total,1.0)
    P_as_node = np.where(V_Total > 0,V_as_node/V_Total,0)
    P_basal = np.where(V_Total > 0,V_basal/V_Total,0)
    P_basement = np.where(V_Total > 0,V_basement/V_Total,0)
    return [P_as_edge, P_as_node, P_basal, P_basement]

# grain-size distributions
def NewDdist(mean,sigma,n):
    """method to draw a sample from a distribution"""
    if sigma == 0:
        #convert from phi to mm
        median = (2**-mean)/1000.
        dist = (mean,0)
    else:
        rng = np.random.default_rng()
        D_arr = rng.normal(mean,sigma,n)
        medianPhi = np.nanmedian(D_arr)
        median = (2**-medianPhi)/1000
        dist = (np.nanmean(D_arr),np.nanstd(D_arr))
    return median,dist

def CombineDdists(dists,vPs,n, def_mean = 0, def_std = 1):
    """method to combine several distribution samples by volume"""
    if def_std == 0:
        median = (2**-def_mean)/1000
        dist = (def_mean,def_std)
    else:    
        rng = np.random.default_rng()
        n_els = [int(i*n) for i in vPs]
        if np.nansum(n_els) > n-len(vPs):
            for i,j in enumerate(dists):
                if i == 0:
                    try:
                        D_arr =  rng.normal(j[0],j[1],n_els[i])
                    except ValueError:
                        #print (vPs)
                        D_arr =  rng.normal(j[0],j[1],n_els[i])
                else:
                    arr =  rng.normal(j[0],j[1],n_els[i])
                    D_arr = np.concatenate((D_arr,arr))
        else:
            D_arr = rng.normal(def_mean,def_std,n)
        medianPhi = np.median(D_arr)
        median = (2**-medianPhi)/1000.
        dist = (np.nanmean(D_arr),np.nanstd(D_arr))
    return median,dist

def CombineDdistsArray(Ddists, DvPs, RArray = None, n = None, def_mean = 0, def_std = 1):
    """method to combine several distribution samples by volume"""
    if def_std == 0: # single value option
        median = (2**-def_mean)/1000.
        dist = (def_mean,def_std)
    else:    
        if RArray is not None:
            n = len(RArray)
        elif n is not None:
            #make a 1D array of random numbers
            rng = np.random.default_rng()
            RArray = rng.normal(size = n)
        else:
            raise ValueError('Both RArray and samp_n are None, you must provide one of these')
        #get the number of elements for each volume input
        n_els = np.array([np.rint(i*n) for i in DvPs])
        #if there are not NaN issues, we can continue
        if np.nansum(n_els) > n-len(DvPs):
            begins = [int(np.nansum(n_els[:i])) for i,j in enumerate(n_els)]
            ends = [int(np.nansum(n_els[:i+1])) for i,j in enumerate(n_els)]
            #initialise an array for mu
            MuArray = np.ones(n)*def_mean
            #and sigma
            SigArray = np.ones(n)*def_std
            #Get each shift and scale from Ddists if both are finite
            for i, j in enumerate(Ddists):
                if np.isfinite(j[0]) and np.isfinite(j[1]):
                    MuArray[begins[i]:ends[i]] = j[0] 
                    SigArray[begins[i]:ends[i]] = j[1]
            #shift and scale RandArray
            DArray = RArray*SigArray+MuArray
            medianPhi = np.nanmedian(DArray)
            dist = (np.nanmean(DArray),np.nanstd(DArray))
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
            rng = np.random.default_rng()
            RArray = rng.normal(size = n)
        else:
            raise ValueError('Both RArray and samp_n are None, you must provide one of these')
        #get the number of elements for each volume input (except base)
        n_els = np.array([np.rint(i*n) for i in DvPs])
        #if there are not NaN issues, we can continue
        if np.nansum(n_els) > n-len(DvPs):
            #initialise Mu array
            MuArray = np.ones(n)*BaseDist[0]
            #and sigma array
            SigArray = np.ones(n)*BaseDist[1]
            #Make the Base Array
            BaseArray = RArray*SigArray+MuArray
            n_remaining = BaseArray.shape[0]
            #For other dists make elements from a Boolean array False in line with their probability, or do nothing
            for i, j in enumerate(Ddists):
                if i != 0 and n_els[i] > 0 and n_remaining > 0:
                    if np.isfinite(j[0]) and np.isfinite(j[1]):
                        rng = np.random.default_rng()
                        #make the Boolean array
                        Bool = np.full(BaseArray.shape, True, dtype = bool)
                        #make probability array for BaseArray given the distribution
                        ProbArray = norm.pdf(BaseArray, loc=j[0], scale=j[1])
                        try:
                                scale = 1.0/np.nansum(ProbArray)
                                ProbArray = ProbArray*scale
                        except(ZeroDivisionError):
                                print('ProbArray summed to zero {}. Adding to all {}'.format(np.nansum(ProbArray),1/len(ProbArray)))
                                ProbArray = ProbArray+1/len(ProbArray) #the case if we have all zeros
                        try:
                            Bool[rng.choice(BaseArray.shape[0], size = int(n_els[i]), replace = False,p = ProbArray)] = False
                        except(ValueError):
                            Bool[rng.choice(BaseArray.shape[0], size = int(n_els[i]), replace = True)] = False
                        BaseArray = BaseArray[Bool]
                        n_remaining = BaseArray.shape[0]
            medianPhi = np.nanmedian(BaseArray)
            median = (2**-medianPhi)/1000.
            dist = (np.nanmean(BaseArray),np.nanstd(BaseArray))
        else:
            median = (2**-def_mean)/1000.
            dist = (def_mean,def_std)
    return median,dist