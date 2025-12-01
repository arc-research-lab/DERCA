"""
This script warps the whole DERCA_SW logic, and generate the figures DERCA manuscript RTSS submission
(1) performance concerns:
    - due to the large amount of design points, conduct full simulations will be very time-consuming
    - e.g. to reproduce fig 11.a(w/t 16800 points in total) will take XX hours on a 64-core server
    - To conduct a small-scale reproduce of certain points, please refer to XX

(2) Figures to reproduce
    (1) Fig. 11(a)
    (2) Fig. 11(b)
    (3) Fig. 11(c)
    (4) Fig. 13

(3) Additional experiments: We plan to add one more figure in the camera-ready version, 
    comparing the simulation and schedulability analysis results,
    the code to produce this exp is also included in this file
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'DERCA_SW'))
import argparse
from datetime import datetime
import math
import pandas as pd
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from DERCA_SW.parse_workload import AccConfig, Workload
from DERCA_SW.apply_strategy import *
from DERCA_SW.schedulability_analysis import AccTaskset, PP_placer, AccTask, AccRegion
from DERCA_SW.sim_util import ScheConfig
from DERCA_SW.search import Searcher, comp_WCET
from DERCA_SW.utils import gen_bert_mi,gen_bert_t,gen_deit_t,gen_mlp_mixer,gen_pointnet,uunifast



util_list_large = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975,1]
util_list_small = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9, 0.95,1]
num_util_large = 100
num_util_small = 40

def reproduce_fig11a(size, workspace='./temp/fig11a'):
    assert size in ['small','large']
    acc_config_path = './DERCA_SW/configs/acc_config.json'
    sche_config_path = './DERCA_SW/configs/sche_config.json'
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = num_util_small
    else:
        u_list = util_list_large
        num_util = num_util_large
    DNN = [
        [[1024,8192,1024],[1024,8192,1024]],
        [[1024,8192,1024],[1024,8192,1024]]
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = full_workspace,
                    )
    result = searcher.run()
    result_df = searcher.dump_sche_rate('sche_success')
    result_df = searcher.dump_sche_rate('ppp_success')
    result_df = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df)

def reproduce_fig11b(size, workspace='./temp/fig11b'):
    assert size in ['small','large']
    acc_config_path = './DERCA_SW/configs/acc_config.json'
    sche_config_path = './DERCA_SW/configs/sche_config.json'
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = num_util_small
    else:
        u_list = util_list_large
        num_util = num_util_large
    DNN = [
        [[2048,128,2048],[2048,128,2048]],
        [[2048,128,2048],[2048,128,2048]]
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = full_workspace,
                    )
    result = searcher.run()
    result_df = searcher.dump_sche_rate('sche_success')
    result_df = searcher.dump_sche_rate('ppp_success')
    result_df = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df)

def reproduce_fig11c(size, workspace='./temp/fig11c'):
    """add different apps sequentially"""
    assert size in ['small','large']
    acc_config_path = './DERCA_SW/configs/acc_config.json'
    sche_config_path = './DERCA_SW/configs/sche_config.json'
    if not os.path.exists(f"./temp/fig11c_{size}"):
        os.makedirs(f"./temp/fig11c_{size}")
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = math.ceil(num_util_small/5)
    else:
        u_list = util_list_large
        num_util = math.ceil(num_util_large/5)

    app_workspace = os.path.join(full_workspace,'deit-t')
    DNN = [
        gen_deit_t(),
        gen_deit_t(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df1a = searcher.dump_sche_rate('sche_success')
    result_df1b = searcher.dump_sche_rate('ppp_success')
    result_df1c = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df1c)

    app_workspace = os.path.join(full_workspace,'bert-t')
    DNN = [
        gen_bert_t(),
        gen_bert_t(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df2a = searcher.dump_sche_rate('sche_success')
    result_df2b = searcher.dump_sche_rate('ppp_success')
    result_df2c = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df2c)

    app_workspace = os.path.join(full_workspace,'bert-mi')
    DNN = [
        gen_mlp_mixer(),
        gen_mlp_mixer(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df3a = searcher.dump_sche_rate('sche_success')
    result_df3b = searcher.dump_sche_rate('ppp_success')
    result_df3c = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df3c)

    app_workspace = os.path.join(full_workspace,'mlp-mixer')
    DNN = [
        gen_bert_mi(),
        gen_bert_mi(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df4a = searcher.dump_sche_rate('sche_success')
    result_df4b = searcher.dump_sche_rate('ppp_success')
    result_df4c = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df4c)

    app_workspace = os.path.join(full_workspace,'pointNet')
    DNN = [
        gen_pointnet(),
        gen_pointnet(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df5a = searcher.dump_sche_rate('sche_success')
    result_df5b = searcher.dump_sche_rate('ppp_success')
    result_df5c = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df5c)

    #rearrange the row idx
    row_order = ['np', 'lw', 'ir', 'ip', 'if', 'ir-ppp', 'ip-ppp', 'if-ppp']
    result_df1c = result_df1c.reindex(row_order)
    result_df2c = result_df2c.reindex(row_order)
    result_df3c = result_df3c.reindex(row_order)
    result_df4c = result_df4c.reindex(row_order)
    result_df5c = result_df5c.reindex(row_order)
    final_dfc = pd.DataFrame(result_df1c.values + result_df2c.values + result_df3c.values + result_df4c.values + result_df5c.values
                            , index=row_order, columns=result_df1c.columns)
    print(f"All search finished, total design point number: {num_util*5},simulation success num:")
    print(final_dfc)
    final_dfc.to_excel(os.path.join(full_workspace,'total_sim_success_rate.xlsx'))

    #also save other dataframe
    result_df1a = result_df1a.reindex(row_order)
    result_df2a = result_df2a.reindex(row_order)
    result_df3a = result_df3a.reindex(row_order)
    result_df4a = result_df4a.reindex(row_order)
    result_df5a = result_df5a.reindex(row_order)
    final_dfa = pd.DataFrame(result_df1a.values + result_df2a.values + result_df3a.values + result_df4a.values + result_df5a.values
                            , index=row_order, columns=result_df1a.columns)
    # print('schedulability analysis/PPP success rate')
    # print(final_dfc)
    final_dfa.to_excel(os.path.join(full_workspace,'total_sche_success.xlsx'))

def reproduce_fig13(size, workspace='./temp/fig13'):
    """compare the WCET of different configurations"""
    if not os.path.exists(f"./temp/fig13_{size}"):
        os.makedirs(f"./temp/fig13_{size}")
    full_workspace = f"./temp/fig13_{size}"
    acc_config_path = './DERCA_SW/configs/acc_config.json'
    acc_config = AccConfig.from_json(acc_config_path)
    
    u_list = [0.7,0.75,0.8,0.85,0.9,0.95,1]
    s_list = ['np','lw','ir-ppp','ip-ppp','if-ppp']
    # u_list = [0.7]
    # s_list = ['if-ppp']
    if size == 'small':
        num_util = 20
        DNN = [
        [[6144,512,4096],[6144,512,4096]],
        [[6144,512,4096],[6144,512,4096]],
        ]
    else:
        num_util = 20
        DNN = [
        [[6144,512,4096],[6144,512,4096]],
        [[6144,512,4096],[6144,512,4096]],
        [[6144,512,4096],[6144,512,4096]]
        ]
    
    result_df = pd.DataFrame(0,index=u_list,columns=s_list)
    #compute 
    tasks = [(u, s) for u in u_list for s in s_list]
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(worker, u, s, num_util, DNN, acc_config): (u, s) for u, s in tasks}
        for future in tqdm(as_completed(futures), total=len(futures)):
            u, s, total = future.result()
            result_df.loc[u, s] += total
    #process result
    avg_wcet = result_df/num_util/3 #3 tasks in one taskset
    avg_wcet.to_excel(os.path.join(full_workspace,f'avg_wcet.xlsx'))
    norm_avg_wcet = result_df/result_df.min().min()
    norm_avg_wcet.to_excel(os.path.join(full_workspace,f'norm_avg_wcet.xlsx'))
    print('average wcet of the tasks')
    print(avg_wcet)
    print('normalized average wcet of the tasks')
    print(norm_avg_wcet)
    
def reproduce_sche_vs_sim(size, workspace='./temp/sche_vs_sim'):
    if os.path.exists(f"./temp/fig11c_{size}"):
        print('use existing simulation results')
    else:
        print('no existing results, conduct simulation')
        reproduce_fig11c(size)
    full_workspace = workspace+f"_{size}"
    sim_success = pd.read_excel(f"./temp/fig11c_{size}/total_sim_success_rate.xlsx")
    sche_success = pd.read_excel(f"./temp/fig11c_{size}/total_sche_success.xlsx")

    # Format sche_success
    sche_success = sche_success.set_index(sche_success.columns[0])
    # sche_success.columns = sche_success.iloc[0]   # first row as column names
    # sche_success = sche_success[1:]               # drop the first row
    sche_success = sche_success.apply(pd.to_numeric)

    # Format sim_success
    sim_success = sim_success.set_index(sim_success.columns[0])
    # sim_success.columns = sim_success.iloc[0]
    # sim_success = sim_success[1:]
    sim_success = sim_success.apply(pd.to_numeric)

    print('real workload simulation success rate')
    print(sim_success)
    print('real workload schedulability analysis/PP placement success rate')
    print(sche_success)

    print('difference')
    difference = sim_success - sche_success
    print(difference)

def worker(u, s, num_util, DNN, acc_config):
    total = 0
    for _ in range(num_util):
        utils = uunifast(len(DNN), u)
        if min(utils) < 0.02:
            utils = uunifast(len(DNN), u)
        total += comp_WCET(DNN, utils, acc_config, s)
    print(f"finished: u={u}({utils}),s={s}, wcet={total}")
    return (u, s, total)   
    
parser = argparse.ArgumentParser(description="cmd tool for reproduce DERCA RTSS2025 Submission figures")
parser.add_argument(
    "--target",
    type=str,
    choices=["fig11a", "fig11b", "fig11c", "fig13", "sche_vs_sim"],
    help="The figure data to reproduce",
    required=True
)
parser.add_argument(
    "--size",
    type=str,
    choices=["small", "large"],
    help="size of the experiment",
    default="small"
)


if __name__ == '__main__':
    start = datetime.now()
    if not os.path.exists("temp"):
        os.makedirs("temp")
    args = parser.parse_args()
    target = args.target
    size = args.size
    if target == 'fig11a':
        reproduce_fig11a(size)
    if target == 'fig11b':
        reproduce_fig11b(size)
    if target == 'fig11c':
        reproduce_fig11c(size)
    if target == 'fig13':
        reproduce_fig13(size)
    if target == 'sche_vs_sim':
        reproduce_sche_vs_sim(size)

    end = datetime.now()
    print('start time:',start)
    print('end time: end',end)
    print('elapse:',end-start)