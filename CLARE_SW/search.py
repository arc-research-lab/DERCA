"""
This script is used for conduct schedulability analysis, PPP, and simulation in larger scale
The input DNN shape is fixed, and the searcher sweeps through different total utils and strategies
"""

from typing import List, Type
import random #random seed are set in the beginning
import pandas as pd
import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import signal
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm


from utils import uunifast, lcm, debug_print
from parse_workload import AccConfig,Workload
from apply_strategy import *
from schedulability_analysis import AccTaskset, schedulability_analyzer, PP_placer
from sim_util import ScheConfig, AccTasksetSim, SimManager


#####################
###random seed#######
random.seed(42)
#####################

# class TestResult:
#     """data class for handling the result"""
#     def __init__(self,sche_success:bool, PPP_success:bool, sim_success:bool):
#         self.sche_success = sche_success #pass schedulability analysis?
#         self.PPP_success = PPP_success #pass PPP?
#         self.sim_success = sim_success #pass simulation?

class TestDesignPt:
    """static class grouping series of funcs"""
    def test(DNN_shapes:List[List[List[int]]],utils:list,
                        acc_config:AccConfig,sche_config:ScheConfig,
                        strategy:str):
        """test a design point, with fixed DNNshape, util for each task, and strategy
        input: 
            - DNN_shapes(list): 3-d list: task-layer-M,K,N shape of the layer
            - utilizations(list), 
            - acc_config(AccConfig), sche_config(ScheConfig)
            - strategies(str):
        Output:
            - pd.dataframe:
                - 3-rows for sche_analysis, PPP, and simulation success 
                - one col for the strategy
            - Note the data processing will be handled by the warpper of this func
        Allowed strategies
            - np: Non-Preemptive
            - lw: LayerWise-preemptive
            - ip: Intra-layer-preemptive Persist
            - ir: Intra-layer-preemptive Recompute
            - if: Intra-layer-preemptive Flexible
            - ip-ppp,ir-ppp,if-ppp: ip,ir,if w/t PP placement optimization
        parse the shape --> workload --> strategies --> conduct sche analysis/PPP for each strategy --> simulation"""
        assert strategy in ['np','lw','ip','ir','if','ip-ppp','ir-ppp','if-ppp'],\
            "[test_design_pt.test] invalid strategy"
        #dump shape into workload
        workloads:List[Workload] = []
        for idx, task_shape in enumerate(DNN_shapes):
            workloads.append(Workload(task_shape,acc_config,f"task{idx}"))
        #conduct test
        result_df = None
        s = strategy
        if s == 'np':
            result_df=(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyNonPreemptive))
        elif s== 'lw':
            result_df=(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyLayerwise))
        elif s== 'ip':
            result_df=(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyPersist))
        elif s== 'ir':
            result_df=(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyRecompute))
        elif s== 'if':
            result_df=(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyFlexible))
        elif s== 'ip-ppp':
            result_df=(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyPersistPPP))
        elif s== 'ir-ppp':
            result_df=(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyRecomputePPP))
        elif s== 'if-ppp':
            result_df=(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyFlexiblePPP))
        return result_df

    def _test_sche_analysis(workloads:List[Workload], utils, sche_config:ScheConfig, strategy:str,strategy_cls:Type[PreemptionStrategy]):
        """test one of the strategy, suit for strategies w/o PPP
        Input: workload, utils, sche_config,strategy
        The strategy should be a class inherited from PreemptionStrategy, as shown in apply_strategy.py
        Output: 3x1 pd.dataframe"""
        #apply strategies
        strategies = []
        for workload in workloads:
            s = strategy_cls()
            s.from_workload(workload)
            strategies.append(s)
        #form taskset
        ts = AccTaskset(strategies,utils)
        #schedulability analysis
        ana = schedulability_analyzer(ts)
        ana.schedulability_test()
        if ana.sche_test_success:
            return pd.DataFrame(
                                {strategy: [True, False, True]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            ) #guarantee to pass sim, skip simulation for perf
        else:
            #dump taskset for simulation
            ts_sim = AccTasksetSim(ts)
            # sim_time = min(lcm(ts_sim.periods),1150000000)
            #conservertive estimation:
            sim_time = lcm(ts_sim.periods)
            if sim_time > 1150000000*20:
                return pd.DataFrame(
                                {strategy: [False, False, False]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )
            sim_manager = SimManager(sche_config,ts_sim,sim_time=sim_time)
            sim_success = sim_manager.run()
            # debug_print(f"{sum(utils)}:{utils}:{strategy} begin sim,periods:{ts_sim.periods}, simtime:{sim_time}")
            return pd.DataFrame(
                                {strategy: [False, False, sim_success]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )
            
    def _test_PPP(workloads:List[Workload], utils, sche_config:ScheConfig,strategy:str,strategy_cls:Type[PreemptionStrategy]):
        """test one of the strategy, suit for strategies w/t PPP
        Input: workload, utils, sche_config,strategy
        The strategy should be a class inherited from PreemptionStrategy, as shown in apply_strategy.py
        Output: TestResult obj"""
        #apply strategies
        strategies = []
        for workload in workloads:
            s = strategy_cls()
            s.from_workload(workload)
            strategies.append(s)
        #form taskset
        ts = AccTaskset(strategies,utils)
        #PP placement
        ppp = PP_placer(ts)
        ppp.PP_placement()
        if ppp.PPP_success:
            return pd.DataFrame(
                                {strategy: [True, True, True]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            ) #guarantee to pass sim, skip simulation for perf
        else:
            #dump taskset for simulation
            ts_sim = AccTasksetSim(ts)
            # sim_time = lcm(ts_sim.periods)
            #conservertive estimation:
            sim_time = lcm(ts_sim.periods)
            if sim_time > 1150000000*20:
                return pd.DataFrame(
                                {strategy: [False, False, False]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )
            sim_manager = SimManager(sche_config,ts_sim,sim_time=sim_time)
            sim_success = sim_manager.run()
            # debug_print(f"{sum(utils)}:{utils}:{strategy} begin sim,periods:{ts_sim.periods}, simtime:{sim_time}")
            return pd.DataFrame(
                                {strategy: [False, False, sim_success]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )

class Searcher:
    """search (1)same DNN types, (2)different utils, (3)different strategies and static the success rate
    generate/load some util distributions us uunifast algorithm, then test the DNN shape on it
    Use a workspace folder to handle the input and output, 
        (1) the workspace is not a dir: create and save all configs 
        (2) the workspace is a valid dir: use existing configs"""
    def __init__(self,acc_config_path:str,sche_config_path:str,
                 DNN_shapes:List[List[List[int]]],
                 utils:list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],#total utils
                 num_util = 100,#points generate for each util
                 strategies = ['np','lw','ip','ir','if','ip-ppp','ir-ppp','if-ppp'],
                 workspace:str = 'search_results'):
        self.workspace = workspace
        if not os.path.exists(self.workspace): #begin from scarch
            os.makedirs(self.workspace)
            # Copy config files into workspace with renamed files
            self.acc_config = AccConfig.from_json(acc_config_path)
            self.sche_config = ScheConfig.from_json(sche_config_path)
            self.DNN_shapes = DNN_shapes
            self.utils = utils
            self.num_util = num_util
            self.strategies = strategies
            shutil.copy(acc_config_path, os.path.join(self.workspace, 'acc_config.json'))
            shutil.copy(sche_config_path, os.path.join(self.workspace, 'sche_config.json'))
            # dump configs to json files
            self._save_json(self.DNN_shapes,'DNN_shapes.json')
            self._save_json(self.utils,'utils.json')
            self._save_json(self.strategies,'strategies.json')
            # generate 
            self.util_dict = self._gen_utils()
            self._save_json(self.util_dict,'util_dict.json')
        else: #load from existing workspace
            self.acc_config = AccConfig.from_json(os.path.join(self.workspace, 'acc_config.json'))
            self.sche_config = ScheConfig.from_json(os.path.join(self.workspace, 'sche_config.json'))
            self.DNN_shapes = self._load_json('DNN_shapes.json')
            self.utils = self._load_json('utils.json')
            self.strategies = self._load_json('strategies.json')
            self.util_dict = self._load_json('util_dict.json')     
    
    def worker(self,task):
        total_util, u, strategy = task
        result = TestDesignPt.test(self.DNN_shapes,u,self.acc_config,self.sche_config,strategy)
        return (total_util, u, strategy, result)
    
    def run(self):
        #init run_work
        run_works = []
        for total_util, u_list in self.util_dict.items():
            for u in u_list:
                for s in self.strategies:
                    run_works.append((total_util,u,s))
   
        raw_results = []
        try:
            with ProcessPoolExecutor(max_workers=None) as executor:
                futures = [executor.submit(self.worker, run_work) for run_work in run_works]
                for future in tqdm(as_completed(futures), total=len(run_works), desc="Processing"):
                    raw_result = future.result()
                    raw_results.append(raw_result)

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            print("killing all ...")
            for p in executor._processes.values():
                os.kill(p.pid, signal.SIGKILL)
            raise
        debug_print('All design points finished!')
        self._save_pkl(raw_results,'raw_results.pkl')
       
        accum_results = {}
        grouped = defaultdict(list)
        for total_util, u, strategy, df in raw_results:
            grouped[total_util].append(df)

        for total_util, dfs in grouped.items():
            merged = pd.DataFrame()  # start empty
            for df in dfs:
                # If column exists in merged, accumulate
                for col in df.columns:
                    if col in merged.columns:
                        merged[col] += df[col].astype(int)
                    else:
                        merged[col] = df[col].astype(int)
            accum_results[total_util] = merged
        self._save_pkl(accum_results,'accum_results.pkl')
        return accum_results

    def _save_json(self, data, filename):
        with open(os.path.join(self.workspace, filename), 'w') as f:
            json.dump(data, f)  
    def _load_json(self, filename):
        with open(os.path.join(self.workspace, filename), 'r') as f:
            return json.load(f)
    def _save_pkl(self, data, filename):
        with open(os.path.join(self.workspace, filename), 'wb') as f:
            pickle.dump(data, f)  
    def _load_pkl(self, filename):
        with open(os.path.join(self.workspace, filename), 'rb') as f:
            return pickle.load(f)
    def _gen_utils(self):
        num_task = len(self.DNN_shapes)
        util_dict = {}

        # generate first util
        f_util = self.utils[0]
        util_dict[f_util] = []
        for i in range(self.num_util):
            u = uunifast(num_task, f_util)
            while min(u) <0.02:#avoid generating too small u, which leads to long sim time
                u = uunifast(num_task, f_util)
            util_dict[f_util].append(u)

        # generate other utils
        for util in self.utils[1:]:
            util_dict[util] = []
            for i in range(self.num_util):
                u = deepcopy(util_dict[f_util][i])
                u[0] += (util - f_util)   # adjust first element
                util_dict[util].append(u) # append to the list
        return util_dict
    def dump_sche_rate(self, metric):
        '''show the success rate results of several metrics
        recorded metric: ['sim_success','sche_success','ppp_success']'''
        assert metric in ['sim_success','sche_success','ppp_success'], 'invalid metric'
        result = self._load_pkl('accum_results.pkl')
        rows = []
        for util, df in result.items():
            if 'sim_success' in df.index:
                rows.append(pd.Series(df.loc[metric], name=util))
            else:
                # If sim_success row not present, you can skip or fill with NaN
                rows.append(pd.Series([pd.NA]*df.shape[1], index=df.columns, name=util))
        # Step 2: Concatenate into a DataFrame with utils as index
        result_df = pd.concat(rows, axis=1)
        result_df.to_excel(os.path.join(self.workspace,f'{metric}.xlsx'))
        return result_df

def comp_WCET(DNN_shapes:List[List[List[int]]],utils:list,
                        acc_config:AccConfig,strategy:str):
    assert strategy in ['np','lw','ir-ppp','ip-ppp','if-ppp']
    """return the wcet of on DNN. If using PPP, return the wcet after PPP"""
    #convert DNN shape to workloads
    workloads:List[Workload] = []
    for idx, task_shape in enumerate(DNN_shapes):
            workloads.append(Workload(task_shape,acc_config,f"task{idx}"))
    #apply strategies:
    if strategy == 'np':
        strategy_cls = StrategyNonPreemptive
        # print('np')
    elif strategy == 'lw':
        strategy_cls = StrategyLayerwise
        # print('lw')
    elif strategy == 'ir-ppp':
        strategy_cls = StrategyRecomputePPP
        # print('ir-ppp')
    elif strategy == 'ip-ppp':
        strategy_cls = StrategyPersistPPP
        # print('ip-ppp')
    elif strategy == 'if-ppp':
        strategy_cls = StrategyFlexiblePPP
        # print('if-ppp')
    strategies = []
    for workload in workloads:
            s = strategy_cls()
            s.from_workload(workload)
            strategies.append(s)
            # s.print_iters(['layer','idx','is_preemptive','strategy','si_r','so_r','si_p','so_p'])
    #form taskset
    ts = AccTaskset(strategies,utils)
    #no PPP strategies: return wcet directly
    if strategy in ['np','lw']:
        sum_wcet = 0
        last_task, period = ts.sorted_tasks[-1]
        sum_wcet = last_task.wcet
        return sum_wcet
    #PPP strategies: conduct PPP, then return wcet       
    else:
        PPP = PP_placer(ts)
        ts_PPP = PPP.PP_placement()
        last_task, period = ts_PPP.sorted_tasks[-1]
        if(PPP.PPP_success):
            sum_wcet = last_task.wcet
        else:
            sum_wcet = last_task.exec_time
        # for task in ts_PPP.tasks:
        #     sum_wcet +=task.wcet
        return sum_wcet
 
if __name__ == "__main__":
    workspace='/home/shixin/RTSS2025_AE/fig11av4'
    acc_config = AccConfig.from_json('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config.json')
    sche_config = ScheConfig.from_json('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/sche_config.json')
    DNN = [
        [[1024,8192,1024],[1024,8192,1024]],
        [[1024,8192,1024],[1024,8192,1024]]
    ]
    start = datetime.now()
    util_list = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975,1]
    searcher = Searcher('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config_lightweight.json',
                        '/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/sche_config.json',
                        DNN,
                        utils=util_list,
                        num_util=100,
                        workspace=workspace)
    # result = searcher.run()
    # with open(os.path.join(workspace, 'accum_results.pkl'), 'rb') as f:
    #         result = pickle.load(f)
    # end = datetime.now()
    # print('sche_rate:')
    # rows = []
    # for u, df in result.items():
    #     print(u)
    #     print(df)
    searcher.dump_sche_rate('sim_success')

    

    # print('start time:',start)
    # print('end time: end',end)
    # print('elapse:',end-start)