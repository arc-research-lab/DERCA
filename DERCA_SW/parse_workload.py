"""
Input workload format: A list, each element is a list containing the shape(MNK) of an Martix Multiplication(MM)
- e.g. [[256,256,256],[256,512,256]] describe a 2-layer MLP of shape 256x256x256 and 256x512x256
This scripts will parse the inputed workload shape, then:
1. compute the tile iterations (#tiles in M,K,N dim)
2. tag the attribute of each tile
    - start & end of a piece of output
    - start & end of a layer
    - start & end of a model
3. compute the cycles used in each tile (load/comp/store/load/persist)
    - i.e. the time for every possible preemption points(PPs)
    - Which PP will be enabled will be decided later by using the PPP algorithm(ours) or heuristics(baseline)
"""
import json
import math
from typing import List
from utils import print_iters

class AccConfig:
    """
    contains the attribute of the accelerator, e.g., #cycle of load/comp/store
    """
    def __init__(self, size_x:int, size_y:int, size_z:int, load:int, comp:int, store:int,
                  clean_output:int, persist:int, resume:int, kernel_mgmt:int):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z # tile size of M(x), K(y), N(z) dimension
        self.load = load
        self.comp = comp
        self.store = store
        self.clean_output = clean_output
        self.persist = persist
        self.resume = resume
        self.kernel_mgmt = kernel_mgmt
    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)
    @classmethod
    def from_json(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

class AccIter:
    """
    Stores the info of each AccIteration when computing the DNN.
    The iterations are the smallest segements of execution in CLARE, preemption can happen only betwwen two iterations
    The load/comp/store pipeline are represented in the iteration, #Iter = #tile(x*y*z)+2, with different load/comp/store latency
    The tiling are represented in the iterations, different iterations has different preemption/resume latency
    """
    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)
    def __init__(self):
        #id
        self.layer = 0
        self.idx = 0
        #time spent in the load/comp/store pipeline
        self.load = 0
        self.comp = 0
        self.store = 0
        self.exec = 0
        #location of this iter
        self.o_start = False
        self.o_end = False #start/end of a piece of an output
        self.l_start = False
        self.l_end = False #start/end of a layer
        #cycle for preemption
        ##records the PP **after** this segment: after this segment, the task is preempted by others
        self.so_r = 0#Swap Out of Recomp: preemption op for recompute, is to clean the output buffer or nothing if a layer ends
        self.so_p = 0#Swap Out of Persist: is to save the output buffer to DDR
        #cycle for resume
        ##records the PP **before** this segment: the task is resumed and begin this sgement
        self.si_r = 0
        self.si_p = 0
        #preemption strategy, how to do the PP **before** this segment
        self.strategy = None
        #if the workload is preemptive **after** this segment
        self.is_preemptive = None

class Workload:
    """stores all the execution latency information of a DNN model
    """
    def __init__(self,NN:List[list]=None,config:AccConfig=None,ID=None):
        self.iters:List[AccIter] = [] #list to contain all iterations
        self.NN=None
        self.config=None
        self.ID=ID
        if NN is not None:
            assert config is not None, "NN and config must be given together"
            self.decompose_NN(NN,config,ID)
    
    def decompose_NN(self,NN:List[list],config:AccConfig,ID=None):
        self.NN=NN
        self.config=config
        self.ID = ID
        #process one layer
        for layer_idx,layer_shape in enumerate(NN):
            self.iters+=Workload._decompose_layer(layer_idx,layer_shape,config)
    def _decompose_layer(layer_idx,layer_shape:list,config:AccConfig)->List[AccIter]:
        """Decompose a layer into iterations. Input: idx and shape of this layer, Output: a list of iterations"""
        iters_in_layer:List[AccIter] = []
        # #tiles in this layer
        TM = math.ceil(layer_shape[0]/config.size_x)
        TK = math.ceil(layer_shape[1]/config.size_y)
        TN = math.ceil(layer_shape[2]/config.size_z)
        # #iter is TM*TK*TN+2 due to the load-comp-store pipeline
        for i in range(TM*TK*TN+2):
            iters_in_layer.append(AccIter())
        #add id
        for i,iter in enumerate(iters_in_layer):
            iter.layer=layer_idx
            iter.idx=i
        Workload._add_latency(iters_in_layer,TM,TK,TN,config)
        Workload._add_location(iters_in_layer,TM,TK,TN,config)
        Workload._add_swap_out_ovhd(iters_in_layer,config)
        Workload._add_swap_in_ovhd(iters_in_layer,config)
        return iters_in_layer
    @staticmethod
    def _add_latency(iters_in_layer:List[AccIter],TM,TK,TN,config:AccConfig)->List[AccIter]:
        #add the load, comp to the iters
        for m in range(TM):
            for n in range(TN):
                for k in range(TK):#output stationary dataflow
                    iters_in_layer[m*TN*TK+n*TK+k].load=config.load
                    iters_in_layer[m*TN*TK+n*TK+k + 1].comp=config.load
        #add store
        for m in range(TM):
            for n in range(TN):
                iters_in_layer[m*TN*TK+n*TK+TK-1 +2].store=config.store
        #add exec
        for iter in iters_in_layer:
            iter.exec = max(iter.load,iter.comp,iter.store)
    @staticmethod
    def _add_location(iters_in_layer:List[AccIter],TM,TK,TN,config:AccConfig)->List[AccIter]:
        #add output piece location info
        for m in range(TM):
            for n in range(TN):
                iters_in_layer[m*TN*TK+n*TK].o_start = True #in this iter the acc load the first input 
                iters_in_layer[m*TN*TK+n*TK+TK-1 +2].o_end=True #in this iter the acc stores the output         
        #add the layer location info
        iters_in_layer[0].l_start =True
        iters_in_layer[-1].l_end =True
    @staticmethod
    def _add_swap_out_ovhd(iters_in_layer:List[AccIter],config:AccConfig)->List[AccIter]:
        #add swap-out ovhd
        for iter in iters_in_layer:
            if not iter.l_end:
                iter.so_r = config.clean_output #recompute: only needs clean output
                iter.so_p = config.persist #persist: store the data to DDR
            else:#after a layer is finished, no on-chip data --> no ovhd
                iter.so_r = 0
                iter.so_p = 0
    @staticmethod
    def _add_swap_in_ovhd(iters_in_layer:List[AccIter],config:AccConfig)->List[AccIter]:
        #add swap-in ovhd
        #persist
        for iter in iters_in_layer:
            if iter.l_start:
                iter.si_p = 0 #no on-chip data --> no loadback & ovhd
            else:
                iter.si_p = config.resume
        #recomp
        for i,iter in enumerate(iters_in_layer):
            if i == 0:#no on-chip data 
                iter.si_r = 0
                iter.last_o_start = 0
            elif i==1:
                iter.si_r = iters_in_layer[0].load
                iter.last_o_start = 0
            else:
                #find the start of this output piece
                for j in reversed(range(0,i-2+1)): #consider the pipeline
                    if iters_in_layer[j].o_start:
                        last_o_start_idx = j
                        break
                iter.last_o_start = last_o_start_idx
                #get the recomp overhead: the exec time of recomputing from the output start
                recomp_ovhd=0
                for recomp_idx, recomp_iter in enumerate(iters_in_layer[last_o_start_idx:i],start=last_o_start_idx):#need recomp the start, no need to comp this iter
                    if recomp_idx-last_o_start_idx == 0:
                        recomp_ovhd += recomp_iter.load # restart the pipeline: load only
                    elif recomp_idx-last_o_start_idx == 1:
                        recomp_ovhd += max(recomp_iter.load,recomp_iter.comp)#load & comp
                    else:
                        recomp_ovhd += recomp_iter.exec #though 3 branches, there shouldn't be any store operation in between
                iter.si_r = recomp_ovhd

    def print_iters(self, fields: List[str]=['layer','load','comp','store','o_start','o_end','l_start','l_end']):
        # Print header
        # header = " | ".join(f"{f}".ljust(12) for f in fields)
        # print(header)
        # print("-" * len(header))
        # # Print each row
        # for iter in self.iters:
        #     row = " | ".join(str(getattr(iter, f, "")).ljust(12) for f in fields)
        #     print(row)
        print_iters(self,fields)
    
    

if __name__ == '__main__':
    config = AccConfig.from_json("./configs/acc_config.json")
    # print(config)

    iter = AccIter()
    # print(iter)

    w1=Workload()
    w1.decompose_NN([[256,1024,4096],[256,1024,4096]],config)
    # w1.comp_ovhd(config)
    w1.print_iters(['layer','idx','load','comp','store','o_start','last_o_start','so_r','so_p','si_r','si_p'])