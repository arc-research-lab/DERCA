"""
This script intakes the workloads that is parsed and applied strategy, and
- combine strategy classes into a whole taskset
    - each strategy class reflects a task
    - period of each tasks are required to input
    - the ddl of each task d == period, i.e. a job(instance of task) is required to be finished before its successor releases
- form the swap-in and swap-out operation latency to be compatible with the real-time theory
- conduct schedulability analysis to the task set --> the task set can meet ddl or not
- for the intra-layer preemptive model, also conduct preemption point placement algorithm to remove the redundant PPs
- generate:
    - (1) the metadata for accelerator, used in simulation and accelerator execution
    - (2) the schedulability analysis results
"""
"""
brief description for CLARE task modeling:
- A **task set**: several tasks
- A **task**: releases infinite instances of this task periodically
    - attributes: period(p), worst-case execution time(WCET)(e), 
- A **job**: one instance of task, every job is required to be finished before its deadline
- **Non-preemptive Region**: each task/job consists several Non-preemptive region where preemption can only happen between regions
    - attributes: WCET(b), preemption overhead(xi), xi is the overhead **before** each region
- NPR, AccIter, and DNN layer:
    - one layer has >=3 iterations in the Acc(load/comp/store)
    - one NPR has >=1 iterations
    - based on the strategies and taskset, a NPR can have less/euqal/more than a layer
    - still, the basic unit of CLARE scheduling is an Acc iteration
"""
from utils import print_iters
from typing import List
from copy import deepcopy
from parse_workload import AccIter, Workload, AccConfig
from apply_strategy import *
from utils import print_iters, lcm, debug_print
import math
from functools import cached_property


class AccRegion:
    """
    One non-preemptive region in schedulability analysis
    """
    def __init__(self, exec_time:int=0, ovhd:int=0, iters:List[AccIter]=[]):
        self.ovhd = ovhd #the preemption ovhd before this region, used in schedulability analysis
        self.so = 0 #swap-out ovhd after this region, used in simulation
        self.si = 0 #swap-in ovhd before this region, used in simulation
        self.iters = iters if iters is not None else [] #the acc iteration within this region, used to generate execution schedule
    def print_iters(self):
        print_iters(self,['layer','idx','is_preemptive','strategy','ovhd'])

    #cache the properties for code perf
    @property
    def exec_time(self):
        return sum(iter.exec for iter in self.iters)
    @property
    def exec_time_cached(self):
        if not hasattr(self, '_exec_time_cache'):
            self.refresh_caches()
        return self._exec_time_cache

    @property
    def wcet(self):
        return self.exec_time + self.ovhd
    @property
    def wcet_cached(self):
        if not hasattr(self, '_wcet_cache'):
            self.refresh_caches()
        return self._wcet_cache
    
    def refresh_caches(self):
        exec_time_val = self.exec_time  # compute once
        self._exec_time_cache = exec_time_val
        self._wcet_cache = exec_time_val + self.ovhd

class AccTask:
    """
    One Task in the schedulability analysis
    """
    def __init__(self, strategy=None):
        self.regions = []
        self.ID = None # used for future updates for generating metadata
        # self.period = None
        if strategy is not None:
            self._from_strategy(strategy)
            self._comp_resume_ovhd()
            self._comp_swap_op_latency()

    def _from_strategy(self, strategy):
        """decouple the workload into NPRs"""
        # assert isinstance(strategy, PreemptionStrategy), f"[Acctask.from_strategy]:the input strategy must be subclass of PreemptionStrategy, get{type(strategy)}"
        s = deepcopy(strategy)
        iters_in_region = []
        self.ID=strategy.ID
        for idx,iter in enumerate(s.iters):
            iter:AccIter
            iters_in_region.append(iter)
            if iter.is_preemptive or idx==len(s.iters)-1: #if is preemptive **after this iter**
                #create a NPR based on iters_in_region
                self.regions.append(AccRegion(iters=iters_in_region))
                #clear the buffer for next NPR
                iters_in_region = []
    
    def _comp_resume_ovhd(self):
        """compute the resume ovhd, i.e., the ovhd of 2nd+ regions, 
        the ovhd of the first region is the preemption ovhd(when this task preempt others)
        and the preemption ovhd will be affected by other tasks"""
        for idx, region in enumerate(self.regions):
            #skip the first region
            region:AccRegion
            if idx==0:
                continue
            #use the first iter to determine the ovhd 
            first_iter = region.iters[0]
            first_iter:AccIter
            if first_iter.strategy == PPStrategy.NA:
                raise ValueError("[AccTask._comp_resume_ovhd]:the first iter uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
            elif first_iter.strategy == PPStrategy.layer:
                region.ovhd = 0
            elif first_iter.strategy == PPStrategy.recomp:
                region.ovhd = first_iter.si_r
            elif first_iter.strategy == PPStrategy.persist:
                region.ovhd = first_iter.si_p
            else:
                raise ValueError("[AccTask._comp_resume_ovhd]:Non-exist strategy, got:{}".format(first_iter.strategy))
    
    def _comp_swap_op_latency(self):
        #swap-in
        for idx, region in enumerate(self.regions):
            region:AccRegion
            #first:region: no swap-in
            if idx==0:
                region.si = None
            else:
                #The swap-in operation affects the first  
                first_iter = region.iters[0]
                first_iter:AccIter
                if first_iter.strategy == PPStrategy.NA:
                    raise ValueError("[AccTask._comp_resume_ovhd]:the first iter of an NPR uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
                elif first_iter.strategy == PPStrategy.layer:
                    region.si = 0
                elif first_iter.strategy == PPStrategy.recomp:
                    region.si = first_iter.si_r
                elif first_iter.strategy == PPStrategy.persist:
                    region.si = first_iter.si_p
        #swap-out:
        for idx,region in enumerate(self.regions):
            region:AccRegion
            #last region: no swap out
            if idx == len(self.regions)-1:
                region.so = None
            else:
                last_iter = region.iters[-1]#swap out affects the last op
                next_iter = self.regions[idx+1].iters[0]#the strategy is stored next NPR, first region
                if next_iter.strategy == PPStrategy.NA:
                    raise ValueError("[AccTask._comp_resume_ovhd]:the first iter of an NPR uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
                elif next_iter.strategy == PPStrategy.layer:
                    region.so = 0
                elif next_iter.strategy == PPStrategy.recomp:
                    region.so = last_iter.so_r#the overhead is kept in this NPR
                elif next_iter.strategy == PPStrategy.persist:
                    region.so = last_iter.so_p

    @property
    def exec_time(self):
        return sum(region.exec_time for region in self.regions)
    @property
    def exec_time_cached(self):
        if not hasattr(self, '_exec_time_cache'):
            self.refresh_caches()
        return self._exec_time_cache
    @property
    def wcet(self):
        return sum(region.wcet for region in self.regions)
    @property
    def wcet_cached(self):
        if not hasattr(self, '_wcet_cache'):
            self.refresh_caches()
        return self._wcet_cache
    def refresh_caches(self):
        self._exec_time_cache = self.exec_time
        self._wcet_cache = self.wcet

    def print_iters(self):
        for npr in self.regions:
            npr:AccRegion
            npr.print_iters()
            print('--------------------------')

    def __repr__(self):
        reprs = [f"({r.ovhd}|{r.exec_time})" for r in self.regions]
        return f"<AccTask ID={self.ID}, regions=[{' -> '.join(reprs)}]>"

class AccTaskset:
    """
    A taskset is composed by several tasks, additionaly, the period info of each task is required
    Input: (1) List of Strategies (2)List of coresponding utilization
    Processing pipeline:
    - convert strategy to Task
    - compute WCET of each region
    - compute resume ovhd (first 3 steps done by AccTask)
    - compute preemption ovhd
    """
    def __init__(self, strategies:list=[], utils:list=[]):
        assert len(strategies)==len(utils), "[AccTaskset.__init__]: #strategies must == #utils"
        self.tasks:List[AccTask] = []
        self.utils = []
        self.periods = []
        self.sche_test_success = None #schedulability test result
        self.PPP_success = None
        if len(strategies)!=0:
            self.utils = deepcopy(utils)
            self._from_strategies(strategies)
            self._comp_period_fast()
            self._comp_preemption_ovhd()
    def _from_strategies(self, strategies):
        assert len(strategies)!=0
        for strategy in strategies:
            self.tasks.append(AccTask(strategy))
    def _comp_period(self):
        self.periods = [0]*len(self.tasks)
        for idx,task in enumerate(self.tasks):
            period = math.ceil(task.exec_time_cached/self.utils[idx])
            self.periods[idx]=period
    def _comp_period_fast(self):
        '''randomly generated periods will easily have a lcm of periods up to 10^14 cycles,
            which is hard to simulate, adjust the periods for small lcm'''
        p0 = 230000 #all periods will be multiple of 220000 cycles, i.e. 1 ms
        p1 = 1000 #adjust the periods of the first task to keep the overall util the same
        total_util = sum(self.utils)
        periods = [0]*len(self.tasks)
        #comp original periods
        for idx,task in enumerate(self.tasks):
            period = math.ceil(task.exec_time_cached/self.utils[idx])
            periods[idx]=period
        #increase the periods to the multiple of p0
        for i in range(len(periods)):
            periods[i]=math.ceil(periods[i]/p0)*p0
        #reduce the p of the first task to maintain the util
        while True:
            util=0
            for idx,task in enumerate(self.tasks):
                util+= task.exec_time_cached/periods[idx]
            if util >= total_util:
                break
            else:
                periods[0]-=p1
        self.periods = periods
        

    @property
    def sorted_tasks(self):
        assert len(self.periods)==len(self.tasks),'[AccTaskset.sorted_tasks]:unmatching #periods and #tasks'
        
        return sorted(zip(self.tasks, self.periods), key=lambda x: x[1])
    def _comp_preemption_ovhd(self):
        for idx, (task,_) in enumerate(self.sorted_tasks):
            #after sorting, the tasks can only preempt other tasks with higher idx
            #the preemption ovhd of one task, is the largest so ovhd of tasks with higher operations
            task:AccTask
            max_so = 0
            for preempted_task, _ in self.sorted_tasks[idx+1:]:
                preempted_task:AccTask
                for preempted_region in preempted_task.regions:
                    preempted_region:AccRegion
                    if preempted_region.so is not None:
                        if preempted_region.so>max_so:
                            max_so = preempted_region.so
            task.regions[0].ovhd=max_so
    def replace_task(self, old_task:AccTask, new_task:AccTask):
        """replace a task using a new AccTask, used in PPP"""
        for idx, task in enumerate(self.tasks):
            if task is old_task:
                self.tasks[idx] = new_task
                return
        raise ValueError("Task to replace not found in Taskset.")

class schedulability_analyzer():
    """
    a class contains the functions used in schedulability analysis
    In CLARE, the task/regions are indexed beginning with 1, here they're convert to 0-indexed
    """
    def __init__(self,taskset:AccTaskset):
        self.TS = deepcopy(taskset)
        self.sche_test_success = None

    @cached_property
    def _n(self):
        """#tasks, task ranging from [0, n-1]"""
        return len(self.TS.sorted_tasks)
    @cached_property
    def _q_max(self):
        """q^max: max NPR WCET in a task, index ranging from (0, n]
        q^max[n] = 0"""
        q_max = [None]*(self._n+1) #index from [0,n], where q_max[0] never used
        for idx, (task, period) in enumerate(self.TS.sorted_tasks):
            task:AccTask
            q_max[idx] = max(region.wcet_cached for region in task.regions)
        q_max[self._n] = 0
        return q_max
    @cached_property 
    def _p(self):
        """periods of each task, ranging from [0,n-1]"""
        return [period for _,period in self.TS.sorted_tasks]
    @cached_property
    def _d(self):
        """deadline, ranges from [0,n]
        to assist the analysis, define that d[n]=lcm(p[0],p[1],...p[n-1])
        where lcm stands for least common multiple
        in CLARE we set d[i]=p[i], i.e. a job must be finished before its successor releases"""
        d = deepcopy(self._p)
        d.append(lcm(self._p))
        return d  
    def _t(self,k):
        """the possible t points when computing beta:
        for computing each beta[k](ranging from [0,n-1]), the possible t satisfies:
            (1) d[k]<=t<d[k+1]
            (2) t = Gamma(t) = a*p[x]+d[x], where x ranging from [0,n-1]
        physical meaning, from this task k's ddl to next task k+1's ddl, all time instances of job release/ddl"""
        assert k>=0 and k<=self._n-1, '[sche_analyzer._t]:invalid k value'
        LB = self._d[k]
        UB = self._d[k+1]
        t = []
        for x in range(0,self._n):
            #release/ddl time instance for one job 
            #for: from dx(included) to UB(excluded), with p as step
            #if: ti is larger than LB
            tx = [ti for ti in range(self._d[x],UB,self._p[x])
                  if ti>=LB] 
            t+=tx
        #TODO: check out the exact logic when there's a case that LB=UB
        #2 cases will result in LB==UB
        # (1) 2 tasks has the exact same period
        # (2) the longest period happen to be the lcm of the others
        # handling: if LB==UB, add the UB in consideration
        # this introduces additional possible t, thus potentially reduces beta_k
        # Still, this is safe since the scheduability analysis represents a sufficient condiction of meet deadline
        # The cases with False result can still meet ddl in simulation
        if LB==UB:
            t.append(LB)
        return t
    @property #when conducting PPP, e will change since ovhd is reduced
    def _e(self):
        """
        WCET for each task, ranging from [0,n-1]
        e_i = sum(j)(b_ij + xi_ij), 
        where b_ij is the execution lengeth, xi_ij is the preemption ovhd of each task
        """
        return [task.wcet_cached for (task,period) in self.TS.sorted_tasks]
    @property
    def _e_cached(self):
        # if not hasattr(self, '_e_cache'):
        #     self.refresh_caches()
        #manually refresh the value
        return self._e_cache 
    def _DBF(self,j,t):
        """Demand Budget Func:
        j ranges from [0,n-1],t is the t's defined by Gamma(t) (in _t())"""
        # assert 0<=j and j<=self._n-1, "[sche_analyzer._DBF]: j value out of range"
        # DBF = 1 + math.floor((t-self._d[j])/self._p[j])
        DBF = 1 + (t-self._d[j])//self._p[j]
        DBF *= self._e_cached[j]
        # debug_print("DBF[{}][{}]:{} = {} x {}".format(j,t,DBF,1 + math.floor((t-self._d[j])/self._p[j]),self._e[j]))
        return DBF
    def _sum_DBF(self,t):
        """the sche analysis alway sum up the DBF func for all tasks at one time instance"""
        sum_DBF = sum(self._DBF(j,t) for j in range(0,self._n))
        return sum_DBF
    def _beta_k(self,k):
        """compute one beta point, k ranging from [0,n-1]"""
        assert 0<=k and k<=self._n-1, "[sche_analyzer._beta_k]:k value out of range"
        # debug_print("compute beta {}".format(k))
        possible_t = self._t(k)

        possible_beta = [t-self._sum_DBF(t) for t in possible_t]
        beta_k = min(possible_beta)
        return beta_k
    @property
    def _beta(self):
        """beta: ???, index ranging from [0,n-1]"""
        beta = [None]*self._n
        for idx in range(0,self._n):
            beta[idx] = self._beta_k(idx)
        return beta
    def refresh_caches(self):
        self._e_cache = self._e
    def schedulability_test(self)->AccTaskset:
        """Note: the therom indexes begining with 1, thus all list indexing should -1
        for i ranges from [1,n], q^max[i] <= min(beta[k]) where k = [0,i-1]"""
        self.refresh_caches()
        q_max = self._q_max
        beta = self._beta
        ineq_result = [q_max[i]<=min(beta[0:i]) for i in range(1,self._n+1)]
        # debug_print('ineq result:')
        # for i in range(1,self._n+1):
        #     print("[{}|{}]".format(q_max[i],min(beta[0:i])))
        success = all(ineq_result)
        if success:
            self.sche_test_success = True
            self.TS.sche_test_success = True
            return self.TS
        else:
            self.sche_test_success = False
            self.TS.sche_test_success = False
            return self.TS
            
class PP_placer(schedulability_analyzer):
    """inherit from schedulability analyzer, group functions for PP placement """
    def __init__(self, taskset):
        """since the taskset is copied, directly change it for the placed taskset"""
        super().__init__(taskset)
        self.PPP_success=None
        self.PPP_err_msg=None
    @property
    def _U(self):
        """Upper bound of the q^max(the max NPR length)
        for i ranges from [1,n], q^max[i] <= min(beta[k]) where k = [0,i-1]
        Thus U ranges from [1,n-1]:
            - task 0(first task): won't be preempted, merged as a whole
            - task i in [1,n-1]: relies on beta[0]~beta[i-1] --> to place Ti, T0~Ti-1 must be placed"""
        U = [None]*self._n
        U[0] = None #Task 0: no upper bound, shouldn't be used
        beta = self._beta
        for i in range(1,self._n):
            U[i] = min(beta[0:i])#U_1 to U_n-1
        return U

    def _merge_region(self, r1:AccRegion,r2:AccRegion)->AccRegion:
        """merge two consequtive Acc regions into one new region, r1 is the earlier one
        """
        merged = AccRegion()
        # merged.iters = deepcopy(r1.iters)
        #change attributes:
        ##iterations
        # merged.iters = deepcopy(r1.iters) + deepcopy(r2.iters) #for perf consideration, do deepcopy in the list
        merged.iters = r1.iters + r2.iters
        ##ovhd **before** the region: the ovhd between r1 and r2 is removed
        merged.ovhd = r1.ovhd
        ##si: **before** the region
        merged.si = r1.si
        ##so: **after** the region
        ##it's true that PPP will change the so of one task, affecting the preemotion ovhd of others
        ##the problem is, for task i,j(i<j) ovhd of task i relies on PPP of task j
        ##but PPP of task j needs to know the ovhd of task i
        ##Since PPP won't increase but decrease the ovhd(since so are removed), it's safe to keep the preemption ovhd 
        merged.so = r1.so
        return merged

    def _merge_list(self,regions:List[AccRegion])->AccRegion:
        """merge a list of regions as if merge them sequentially from List[0]
            can 'merge' list of only one region: return the region itself"""
        
        if len(regions) == 1:#only one region
            return deepcopy(regions[0])
        else:
            regions_local = deepcopy(regions)
            # regions_local = regions
            merged_region = self._merge_region(regions_local[0],regions_local[1])
            if len(regions) > 2:
                for region in regions_local[2:]:
                    merged_region = self._merge_region(merged_region,region)
            return merged_region
    
    def _merge_task(self, task:AccTask, Ui)->AccTask:
        """give the upper bound of NPR length(Ui) computed, merge the regions to place the PPs
        A sliding window algorithm is applied"""
        new_task = AccTask()
        new_task.ID = task.ID

        start_idx = 0
        end_idx = 0#start and end index of the sliding window
        trial_idx = 0#
        while end_idx < len(task.regions):
            #try to merge a new region, sliding window:[start_idx, end_idx]
            merged = self._merge_list(task.regions[start_idx:end_idx+1])
            if merged.wcet < Ui:
                #success
                if end_idx <len(task.regions)-1:#not the last region, try to merge the next one
                    end_idx += 1
                    continue
                else:#the last region, add to task & break
                    new_task.regions.append(merged)
                    break
            else:
                #fail, have to add a PP within the regions[start_idx, end_idx] to split them to two parts, 
                #namely 'left' and 'right' region on the timeline
                #the object is (1) both left and right region, wcet < ui and (2) the right region has the smallest wcet
                split_candidate = []#possible split solution
                for trial_idx in range(start_idx, end_idx):#left:[start_idx,trial_idx], right:(trial_idx, end_idx]
                    left_list = task.regions[start_idx:trial_idx+1]
                    right_list = task.regions[trial_idx+1:end_idx+1]
                    if len(left_list)==0 or len(right_list)==0:
                        continue#both slices must not be empty since otherwise no PP is inserted
                    left_region = self._merge_list(left_list)
                    right_region = self._merge_list(right_list)
                    if left_region.wcet > Ui or right_region.wcet > Ui:
                        continue#both regions have to satisfy the Ui bound
                    #a valid solution
                    split_candidate.append((trial_idx,right_region.wcet))
                if len(split_candidate) == 0:
                    #no valid solution found, PPP fails
                    raise ValueError("PPP fails")
                #get the solution with smallest right.wcet
                split_idx = min(split_candidate, key=lambda x: x[1])[0]
                #the left is kicked out of the sliding window
                new_task.regions.append(self._merge_list(task.regions[start_idx:split_idx+1]))
                #the right is the new sliding window
                start_idx = split_idx+1
                continue
        return new_task

    def PP_placement(self)->AccTaskset:
        """
        Merge the regions to remove redudant PPs, reducing ovhd.
        The key idea is to use the min(beta) or U as the upper bound of merging two regions
        for task 0, no other tasks can preempt it, thus it's placed as a whole
        for task i(i>0), U[i] is related to the wcet of task [0~i-1], thus we need to place PP in sequence of tasks
        """
        #iterate through the tasks:
        for idx,(task,_) in enumerate(self.TS.sorted_tasks):
            #refresh caches
            self.refresh_caches()
            if idx == 0:#merge the first task as a whole
                merged_task = AccTask()
                merged_task.ID = task.ID
                merged_task.regions = [self._merge_list(task.regions)]
                self.TS.replace_task(task,merged_task)
            else:#compute upperbound, then merge task
                try:
                    Ui = self._U[idx]
                    merged_task = self._merge_task(task,Ui)
                    self.TS.replace_task(task,merged_task)
                except ValueError as e:
                    #when PP placement fails
                    self.PPP_success = False
                    self.PPP_err_msg = e
                    self.TS.PPP_success = False
                    self.TS.sche_test_success = False    
                    return self.TS
        self.PPP_success = True
        self.TS.PPP_success = True
        self.TS.sche_test_success = True 
        return self.TS


if __name__ == '__main__':
    config = AccConfig.from_json("/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config.json")
    # print(config)
    iter = AccIter()
    # print(iter)
    w1=Workload()
    print('decompose_NN')
    w1.decompose_NN([[256,4096,256],[256,256,256]],config)
    # w1.comp_ovhd(config)
    # w1.print_iters(['layer','idx','load','comp','store','o_start','last_o_start','so_r','so_p','si_r','si_p'])
    print('apply strategy')
    s1 = StrategyLayerwise()
    # s1 = StrategyFlexible()
    s1.from_workload(w1)
    # s1.print_iters(['layer','idx','is_preemptive','si_r','si_p','strategy'])
    t1 = AccTask(s1)
    # print(t1)

    print('form taskset')
    taskset = AccTaskset([s1,s1],[0.4,0.4])
    for task,_ in taskset.sorted_tasks:
        print(task)

    print('begin sche analysis')
    ana = schedulability_analyzer(taskset)
    ana.schedulability_test()
    print('sche analysis:',ana.sche_test_success)
    debug_print('beta_value',ana._beta)

    print('begin PPP')
    PPP = PP_placer(taskset)
    PPP.PP_placement()
    print("PPP success:",PPP.PPP_success)
    print(PPP.TS.sorted_tasks)
    

