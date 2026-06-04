import ray
import numpy as np
import networkx as nx
from GraphSSeT_NetworkX_parallel import SGST_supervisor

@ray.remote
class Multi_SGST_supervisor:
    def __init__(self, G, SGs, n_actors = 1, t = 0, checkpoint_freq = 1,
                 max_cycle_its = 100, min_dt = 900, max_dt = 86400):

        self.graph = G # main graph
        self.SGs = SGs #subgraphs(catchments)
        self.n_actors = n_actors #n_actors per SG

        self.one_week = 3600.*24.*7.
        self.checkpoint_freq = checkpoint_freq * self.one_week #checkpoint frequency given in weeks
        self.max_cycle_its = max_cycle_its #maximum iterations in a cycle before we run
        self.min_dt = min_dt #min dt
        self.max_dt = max_dt #max dt

        self.supervisors = []#list of supervisors (one per SG)
        self.actor_sets = []#list of actor sets (n_actors for each SG)
        self.times = [] #active time on each supervisor
        self.t = t #global time

    #variable dHdt_function (simplified)
    def _dt_from_dHdt(self,G,min_dt,max_dt,target_maxdeltaH = 0.25):
        A = nx.get_edge_attributes(G,'dHdt')
        dHdt = np.abs([A[key] for key in A.keys()])
        A = nx.get_edge_attributes(G,'till_thickness')
        H = np.array([A[key] for key in A.keys()])
        H = np.where(H<0.05,0.05,H)
        inv_dt = np.sort(dHdt/H)
        next_dt = target_maxdeltaH/inv_dt[-1]
        if next_dt < min_dt:
            dt = min_dt
        elif next_dt > max_dt:
            dt = max_dt
        else:
            dt = next_dt
        return dt

    # Initialise Supervisors
    def _initialise_supers(self, sgst_kwargs):
        if isinstance(self.n_actors,int):
            self.n_actors = [self.n_actors] * len(self.SGs)
        for i, Gi in enumerate(self.SGs): # instead of for loop we may want to use ray to deserialise?
            sgst_super = SGST_supervisor.options(num_cpus=1).remote(Gi, n=self.n_actors[i])
            print(f'catchment:{i} n_actors:{self.n_actors[i]}',flush = True)
            sgst_super.MakePartitions.remote()
            sgst_super.PartitionGraph.remote()
            actor_set = ray.get(sgst_super.SpawnWorkerActors.remote(sgst_kwargs))
            self.supervisors.append(sgst_super)
            self.actor_sets.append(actor_set)
            self.times.append(self.t)

    # Run one step, either to t+dt or to checkpoint time
    def _one_step(self, i, t, t_checkpoint, lite = False):
        sgst_super = self.supervisors[i]
        G = self.SGs[i]
        dt = self._dt_from_dHdt(G, self.min_dt, self.max_dt)
        t += dt
        if t > t_checkpoint:
            t = t_checkpoint
        # worker execution
        rus_result = sgst_super.RunWorkerActorsSemiStrict.remote(
            self.actor_sets[i],
            t,
            lite=lite
            )
        # return future result + updated time 
        return t, rus_result

    #Main run from one checkpoint to the next - here we block ONLY on the checkpoints
    def _run_to_checkpoint(self, lite = False):
        t_checkpoint = (self.t // self.checkpoint_freq + 1) * self.checkpoint_freq #next t that is a multiple of checkpoint frequency 
        while self.t < t_checkpoint:
            active = set(range(len(self.supervisors)))
            futures = {}
            local_t = self.times.copy()
            while active:
                for i in list(active):
                    
                    # checkpoint condition
                    if local_t[i] >= t_checkpoint:
                        active.discard(i)
                        continue
                    
                    t_new, fut = self._one_step(i, local_t[i],t_checkpoint, lite = lite)
                    local_t[i] = t_new
                    
                    futures[fut] = i
                
                if not futures:
                    break
                
                done, _ = ray.wait(list(futures.keys()), num_returns=1)
                for d in done:
                    i = futures.pop(d)
                    
                    if i not in active:
                        continue
                    
                    if local_t[i] >= t_checkpoint:
                        active.discard(i)
                        
            # Checkpoint reached for all actors
            results = ray.get([
                sgst_super.get_graph.remote()
                for sgst_super in self.supervisors
            ])
            print(f"Checkpoint reached for all actors at t={t_checkpoint/self.one_week:.2f} weeks", flush = True)
            # feed now into the next checkpoint
            self.SGs = results
            self.times = local_t
            self.t = t_checkpoint
    
    def Harmonise_SGs(self):
        #get the main graph from data store
        MainGraph = self.graph.copy()
        for G in self.SGs:
                # and update the main graph
                MainGraph.update(G)
        self.graph = MainGraph
    
    def get_t(self):
        return self.t
    
    def get_graphs(self):
        return self.graph, self.SGs