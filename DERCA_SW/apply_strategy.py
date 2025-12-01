"""
This script add different preemption strategy evaluated in CLARE:
- Input: parsed Workload class
- Based on selected strategies, change the 'is_preemptive' and 'strategy' attribute of each iteration within one class
    - preemption strategy: how to do the PP **before** this segment (persist or recompute)
    - is_preemptive: if the workload is preemptive **after** this segment
- Merge the non-preemptive iterations into metadata to be executed
- Output: Task_metadata class

- implemented preemption strategy:
    - non-preemptive (np)
    - layerwise-preemptive (lw)
    - intra-layer preemptive, recompute only, w/o preemption point placement (ir, w/o PPP)
    - intra-layer preemptive, persist only, w/o preemption point placement (ip, w/o PPP)
    - intra-layer preemptive, flexible, w/o preemption point placement (if, w/o PPP)
    - intra-layer preemptive, recompute only, w/t preemption point placement (ir, w/t PPP)
    - intra-layer preemptive, persist only, w/t preemption point placement (ip, w/t PPP)
    - intra-layer preemptive, flexible, w/t preemption point placement (if, w/t PPP)

Using the proposed heuristic, the strategy of each PP can be defined before schedulability PP placement
"""
from typing import List
from copy import deepcopy
from parse_workload import AccConfig,AccIter,Workload
from enum import Enum
from utils import print_iters

class PPStrategy(Enum):
    """a enum class for different PP implementation"""
    recomp = 'recompute'
    persist = 'persist' 
    layer = 'layerwise' #gap between two layer or at the end of a model, no ovhd
    NA = 'not applicable'

class PreemptionStrategy:
    "base class for all preemption strategies"
    def __init__(self):
        self.iters:List[AccIter] = []
        self.ID=None
    
    def from_workload(self,workload:Workload):
        raise NotImplementedError('[Preemption Strategy]: this is a abstract class, use a the child classes instead')

    def print_iters(self, fields: List[str]=['layer','idx','is_preemptive','strategy']):
        print_iters(self,fields)
    
class StrategyNonPreemptive(PreemptionStrategy):
    "Non-preemptive execution: the whole model is a whole"
    def __init__(self):
        super().__init__()
    def from_workload(self,workload:Workload):
        self.iters = deepcopy(workload.iters)
        for iter in self.iters:
            iter.is_preemptive = False
            iter.strategy = PPStrategy.NA #not applicable
        self.iters[-1].is_preemptive = True
        self.ID = workload.ID
    
class StrategyLayerwise(PreemptionStrategy):
    "Layerwise preemptive execution: each layer in a model is a whole"
    def __init__(self):
        super().__init__()
    def from_workload(self,workload:Workload):
        self.iters = deepcopy(workload.iters)
        #add strategy
        for iter in self.iters:
            if iter.l_start:
                iter.strategy = PPStrategy.layer
            else:
                iter.strategy = PPStrategy.NA
        self.iters[0].strategy = PPStrategy.NA
        #add is_preemptive
        for iter in self.iters:
            if iter.l_end:
                iter.is_preemptive = True
            else:
                iter.is_preemptive = False
        self.ID = workload.ID

class StrategyRecompute(PreemptionStrategy):
    """Recompute only strategy"""
    def __init__(self):
        super().__init__()
    def from_workload(self,workload:Workload):
        self.iters = deepcopy(workload.iters)
        #add strategy:
        for iter in self.iters:
            iter.strategy = PPStrategy.recomp
        self.iters[0].strategy = PPStrategy.NA
        #add is_preemptive
        for iter in self.iters:
            iter.is_preemptive = True
        self.ID = workload.ID

class StrategyPersist(PreemptionStrategy):
    """Persist only strategy"""
    def __init__(self):
        super().__init__()
    def from_workload(self,workload:Workload):
        self.iters = deepcopy(workload.iters)
        #add strategy:
        for iter in self.iters:
            iter.strategy = PPStrategy.persist
        self.iters[0].strategy = PPStrategy.NA
        #add is_preemptive
        for iter in self.iters:
            iter.is_preemptive = True
        self.ID = workload.ID

class StrategyFlexible(PreemptionStrategy):
    """Flexible strategy: choose the strategy based on the resume/swap-in ovhd"""
    def __init__(self):
        super().__init__()
    def from_workload(self,workload:Workload):
        self.iters = deepcopy(workload.iters)
        #add strategy:
        for iter in self.iters:
            #use the heuristic: compare the resume overhead before this iteration
            recomp_ovhd = iter.si_r
            persist_ovhd = iter.si_p
            if iter.l_start:#the layerwise PPs are still availiable, and requires no strategy
                iter.strategy = PPStrategy.layer
            elif recomp_ovhd > persist_ovhd:
                iter.strategy = PPStrategy.persist
            else:
                iter.strategy = PPStrategy.recomp
        self.iters[0].strategy = PPStrategy.NA
        #add is_preemptive
        for iter in self.iters:
            iter.is_preemptive = True
        self.ID = workload.ID

#for strategy with or without PPP, their 
class StrategyRecomputePPP(StrategyRecompute):
    pass

class StrategyPersistPPP(StrategyPersist):
    pass

class StrategyFlexiblePPP(StrategyFlexible):
    pass


if __name__ == '__main__':
    config = AccConfig.from_json("./configs/acc_config.json")
    # print(config)
    iter = AccIter()
    # print(iter)
    w1=Workload()
    w1.decompose_NN([[256,4096,256],[256,256,256]],config)
    # w1.comp_ovhd(config)
    # w1.print_iters(['layer','idx','load','comp','store','o_start','last_o_start','so_r','so_p','si_r','si_p'])
    s1 = StrategyFlexiblePPP()
    s1.from_workload(w1)
    s1.print_iters(['layer','idx','is_preemptive','si_r','si_p','strategy'])