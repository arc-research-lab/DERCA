"""
This script use simpy to run the simulation:
Key simulation structure:
|-------------------|
|   job_generator   |
|-------------------|
     |
     | job_release
     |
     V
|-------------------|
|   scheduler       |
|-------------------|
  |              ^
  |instr         |feedback
  |              |
  V              |
|-------------------|
|   accelerator     |
|-------------------|
Three modules run in parallel:
job_generator:
    - generate jobs for each task periodically(may add sporadic in the future)
scheduler:
    - handles 3 events in a fifo manner:
    - get feedback:
        - process the finished region:
            - change the status of the corresponding job
            - if the job is finished, (1) check deadline compliance (2)record
        - allows the scheduler to issue next region
        - pay latency
    - handle job release:
        - add the newly released jobs to the heap
        - pay latency of each added job
    - release task:
        - releases task for exec and pay latency
        - if the job is finished, (1) remove the job from the heap and (2) pay heap latency for selecting the new job
acc:
    - wait for the instr
    - once get instr:
        - when preemption happens: pay swap-out ovhd
        - when resume happens: pay swap-in ovhd
        - pay execution time
    - send feedback when finish
"""


from parse_workload import AccConfig, Workload
from apply_strategy import *
from schedulability_analysis import AccRegion, AccTask, AccTaskset, schedulability_analyzer, PP_placer
from utils import debug_print, init_logger, lcm
from copy import deepcopy
import json
import simpy
import logging
from typing import List, Optional
import sys
import os


class AccRegionSim:
    """static class, caches the info needed in a region to optimize simulation"""
    def __init__(self, region:AccRegion):
        self.exec_time = region.exec_time
        self.wcet = region.wcet
        self.ovhd = region.ovhd
        self.si = region.si
        self.so = region.so
    def __repr__(self):
        return (f"AccRegionSim(exec_time={self.exec_time}, "
                f"wcet={self.wcet}, ovhd={self.ovhd}, "
                f"si={self.si}, so={self.so})")

class AccTaskSim:
    """static class, caches the info needed in a region to optimize simulation"""
    def __init__(self, task:AccTask):
        self.ID = task.ID
        self.period = None
        self.regions = []
        self.exec_time = task.exec_time
        self.wcet = task.wcet
        for region in task.regions:
            self.regions.append(AccRegionSim(region))
        self.num_region=len(self.regions)
    def printNPR(self):
        header = f"AccTaskSim(ID={self.ID}, period={self.period}, exec_time={self.exec_time}, wcet={self.wcet})\n"
        region_header = f"{'Idx':<4} {'exec_time':<10} {'wcet':<10} {'ovhd':<10} {'si':<10} {'so':<10}\n"
        region_lines = []
        for i, region in enumerate(self.regions):
            line = f"{i:<4} {region.exec_time:<10} {region.wcet:<10} {region.ovhd:<10} {region.si:<10} {region.so:<10}"
            region_lines.append(line)
        return header + region_header + "\n".join(region_lines)
    def __repr__(self):
        header = f"AccTaskSim(ID={self.ID}, period={self.period}, exec_time={self.exec_time}, wcet={self.wcet})\n"
        region_chain = " -> ".join(
            f"({r.si}|{r.exec_time}|{r.so})" for r in self.regions
        )
        return header + region_chain

class AccTasksetSim:
    def __init__(self, taskset:AccTaskset):
        self.tasks = []
        for task in taskset.tasks:
            self.tasks.append(AccTaskSim(task))
        self.utils = taskset.utils
        self.periods = taskset.periods
        self.sche_test_success = taskset.sche_test_success
        self.PPP_success = taskset.PPP_success
        #add periods to each task
        for idx, task in enumerate(self.tasks):
            task:AccTaskSim
            task.period = self.periods[idx]
        #add task id dict for lookup
        self._task_dict = {t.ID: t for t in self.tasks}

    def __repr__(self):
        header = (
            f"AccTasksetSim(utils={self.utils}, periods={self.periods}, "
            f"sche_test_success={self.sche_test_success}, PPP_success={self.PPP_success})\n"
        )
        task_reprs = "\n".join(repr(task) for task in self.tasks)
        return header + task_reprs
    
    def get_task(self,ID)->AccTaskSim:
        return self._task_dict.get(ID)
        
class ScheConfig:
    """Scheduler configuration parameters."""
    def __init__(self, feed_back_latency:int, task_release_latency:int,
                 issue_instr_latency:int, task_finish_latency:int,
                 heap_top_down_depth:int, heap_top_down_II:int,
                 heap_bottom_up_depth:int, heap_bottom_up_II:int):
        self.feed_back_latency = feed_back_latency
        self.task_release_latency = task_release_latency
        self.issue_instr_latency = issue_instr_latency
        self.task_finish_latency = task_finish_latency
        self.heap_top_down_depth = heap_top_down_depth
        self.heap_top_down_II = heap_top_down_II
        self.heap_bottom_up_depth = heap_bottom_up_depth
        self.heap_bottom_up_II = heap_bottom_up_II

    @classmethod
    def from_json(cls, filepath:str):
        """Load scheduler configuration from a JSON file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        required_keys = [
            "feed_back_latency", "task_release_latency", "issue_instr_latency", 
            "task_finish_latency", "heap_top_down_depth", "heap_top_down_II",
            "heap_bottom_up_depth", "heap_bottom_up_II"
        ]

        # Check all required keys exist
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

        return cls(**{k: data[k] for k in required_keys})
    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

class JobSim:
    """data class representing a job in simulation"""
    def __init__(self,task,job_id:int,num_region:int,release:int,ddl:int):
        self.task = task #task
        self.job_id = job_id
        self.region = 0 #the **next** region of this job to issue, idx begin from 0
        self.num_region = num_region #the total region num
        self.release = release
        self.ddl = ddl #release and end cycle
        self.process = None #first time be processed
        self.issue = [] #cycles of issue
        self.feedback = [] #cycles of feedbacks
    def __repr__(self):
        return (f"JobSim(task={self.task}, id={self.job_id}, region={self.region}/"
            f"{self.num_region}, release={self.release}, ddl={self.ddl})")
    
class JobGenerator:
    def __init__(self,env:simpy.Environment,taskset:AccTasksetSim,
                 task_release_fifo:simpy.Store,
                 logger:logging.Logger):
        self.env:simpy.Environment = env
        self.taskset:AccTasksetSim = taskset
        self.logger:logging.Logger = logger
        self.task_release_fifo:simpy.Store = task_release_fifo
        self.job_id:int = 0
    def _generate_job(self,env:simpy.Environment,
                      task:AccTaskSim,fifo:simpy.Store,
                      logger:logging.Logger):
        task_id = task.ID
        period = task.period
        while True:
            job_id = self.job_id
            self.job_id +=1
            release = env.now
            ddl = release + period
            new_job = JobSim(
                task=task_id,
                job_id=job_id,
                num_region=len(task.regions),
                release = release,
                ddl= ddl 
            )
            logger.info("[{}][JobGen] task released: task={}, Job={}, ddl={}"
                         .format(release, task_id, job_id, ddl))
            yield fifo.put(new_job)
            yield self.env.timeout(period) #after a period, run again for periodically release
    def run(self):
        for task in self.taskset.tasks:
            self.env.process(self._generate_job(self.env,task,self.task_release_fifo,self.logger))

class Instr:
    """data class for the instruction channel"""
    def __init__(self,job:JobSim,region:int,preempt:bool,resume:bool,
                 last_task:int,last_region:int):
        self.job:JobSim = job #pass the job to track the whole lifecycle
        #the task,job,release,ddl info is in the job obj
        self.region:int = region #region to execute
        self.preempt:bool = preempt
        self.resume:bool = resume #instr to conduct preemption
        self.last_task:int = last_task
        self.last_region:int = last_region #used for conduct swap-in
    def __repr__(self):
        return (f"Instr(job={self.job}, region={self.region}, "
            f"preempt={self.preempt}, resume={self.resume}, "
            f"last=({self.last_task},{self.last_region}))")

class Feedback:
    """small item notifying the execution is finished """
    def __init__(self,job:JobSim,region:int):
        self.job:JobSim = job
        self.region:int = region

class Scheduler:
    """simulate the heap behavior:
    - for performance, use a list to represent the heap, and assume the heap operation always take the worst
    - In HW implementation, seperate FIFOs are used to handle the job release, here a unified fifo are used
    - still, the priority of event handling(feedback>task_release>issue instr) is the same, thus the latency is bounded
    """
    def __init__(self,sche_config:ScheConfig,
                 env:simpy.Environment,taskset:AccTasksetSim,
                 task_release_fifo:simpy.Store,
                 instr_fifo:simpy.Store,
                 feedback_fifo:simpy.Store,
                 logger:logging.Logger):
        self.sche_config:ScheConfig = sche_config
        self.env:simpy.Environment = env
        self.taskset:AccTasksetSim = taskset
        self.task_release_fifo:simpy.Store = task_release_fifo
        self.instr_fifo:simpy.Store = instr_fifo
        self.feedback_fifo:simpy.Store = feedback_fifo
        self.logger:logging.Logger = logger
        self.heap:List[JobSim] = []#use a list to represent the heap
        #registers
        self.cur_task = None
        self.cur_region = None
        self.issue_flag = None
    def _schedule(self):  
        #hyperparams
        num_task = len(self.taskset.tasks)
        max_heap_top_down_latency = self.sche_config.heap_top_down_depth \
            + num_task * self.sche_config.heap_top_down_II
        max_heap_bottom_up_latency = self.sche_config.heap_bottom_up_depth \
            + num_task * self.sche_config.heap_bottom_up_II
        #init regs
        #current task and region: when t=0, it's always the task with the smallest p firstly
        min_task:AccTaskSim = min(self.taskset.tasks, key=lambda task:task.period)
        self.cur_task = min_task.ID
        self.cur_region =0
        self.issue_flag = True#to issue the first instr with no feedback before it
        #register events for listening
        task_release_evt = self.task_release_fifo.get()
        feedback_evt = self.feedback_fifo.get()

        #main loop
        while True:
            result = yield simpy.events.AnyOf(self.env, [task_release_evt, feedback_evt])
            #process feedback
            if feedback_evt in result.events:
                yield from self.__process_fb(feedback_evt)#use yield from to pass the yield events from subfuncs
                feedback_evt = self.feedback_fifo.get() #register the event for listening next time
            #process job release
            if task_release_evt in result.events:
                yield from self.__process_job_gen(task_release_evt,max_heap_bottom_up_latency)
                task_release_evt = self.task_release_fifo.get() 
            #issue instruction: it's ok if the flag is true but heap is empty,
            #since to join something to the heap, a job must be released and triggers the main loop
            if self.issue_flag and self.heap:
                yield from self.__process_instr_issue()
                yield from self.__process_task_finish(max_heap_top_down_latency)
    def __process_fb(self,feedback_evt):
        #allow issuing next instr
        feedback:Feedback = feedback_evt.value
        self.issue_flag = True
        yield self.env.timeout(self.sche_config.feed_back_latency)
        self.logger.debug("[{}][Sche] region finish: task:{}, job:{}, region:{}"
                          .format(self.env.now, feedback.job.task, feedback.job.job_id, feedback.region))
        #check ddl miss when job finished
        if feedback.job.region == feedback.job.num_region:
            self.logger.info("[{}][Sche] task finish: task:{}, job:{}, region:{}"
                          .format(self.env.now, feedback.job.task, feedback.job.job_id, feedback.region))
            if feedback.job.ddl < self.env.now:#deadline not meet
                self.logger.warning("[{}][Sche] deadline miss!: task:{}, job:{}"
                                  .format(self.env.now,feedback.job.task,feedback.job.job_id))
                raise ValueError("deadline miss")
    def __process_job_gen(self,task_release_evt,max_heap_bottom_up_latency):
        """process job release
            all job within the task release fifo will be processed in one run
            after each process, a latency will be paid"""
        job:JobSim = task_release_evt.value #get the job
        self.heap.append(job)
        #do not conduct heap sort, but pay sort latency
        yield self.env.timeout(max_heap_bottom_up_latency)
        self.logger.debug("[{}][Sche] Job added to heap: Task={}, Job={}"
                          .format(self.env.now, job.task, job.job_id))
        #check and process all other events
        while self.task_release_fifo.items:
            job=self.task_release_fifo.items.pop(0)
            self.heap.append(job)
            yield self.env.timeout(max_heap_bottom_up_latency)
            self.logger.debug("[{}][Sche] Job added to heap: Task={}, Job={}"
                              .format(self.env.now, job.task, job.job_id))
    def __process_instr_issue(self):
        """issue instructions: conduct EDF: 
        (1)pick the job with the smallest ddl
        (2)pick the next region of this job
        (3)check if preemption or resume happens
        (4)gen instr and put to the instr fifo """
        self.heap.sort(key=lambda job:job.ddl)#EDF
        next_job=self.heap[0]
        #gen and issue instr
        instr = Instr(job=next_job,
                        region=next_job.region,#seperate this since obj will be updated
                        #preemption: launch a new task but the current task hasn't finish
                        preempt= next_job.task != self.cur_task
                            and self.cur_region + 1 != len(self.taskset.get_task(self.cur_task).regions),
                        #resume: launch a new task not starting from begining
                        resume= next_job.task != self.cur_task
                            and next_job.region !=0,
                        last_task=self.cur_task,
                        last_region=self.cur_region
                        )
        yield self.env.timeout(self.sche_config.task_release_latency)
        yield self.instr_fifo.put(instr)
        self.logger.debug("[{}][Sche] issue region: task={}, job={}, region={}"
                          .format(self.env.now,next_job.task,next_job.job_id,next_job.region))
        #update status
        next_job.issue.append(self.env.now)
        self.issue_flag=False
        self.cur_task = next_job.task
        self.cur_region = next_job.region
        next_job.region+=1
    def __process_task_finish(self,max_heap_top_down_latency):
        """task finish:
        after the last region is issued, the job.region == job.num_region
        remove this job and pay latency
        in sim, the job obj can still be accessed via instr and feedback"""
        #the job just issued must be the top of heap
        if self.heap[0].region == self.heap[0].num_region:
            end_job = self.heap[0]
            self.heap.pop(0)
            self.logger.debug("[{}][Sche] last region issued: task={}, region={}"
                             .format(self.env.now,end_job.task,end_job.region))
            yield self.env.timeout(max_heap_top_down_latency)
    def run(self):
        self.env.process(self._schedule())

class Accelerator:
    def __init__(self,
                 env:simpy.Environment,taskset:AccTasksetSim,
                 instr_fifo:simpy.Store,
                 feedback_fifo:simpy.Store,
                 logger:logging.Logger):
        self.env:simpy.Environment = env
        self.taskset:AccTasksetSim = taskset
        self.instr_fifo:simpy.Store = instr_fifo
        self.feedback_fifo:simpy.Store = feedback_fifo
        self.logger:logging.Logger = logger
    def _acc(self):
        """simulate acceleration: pay preemption/resume ovhd first, then execution time
        return the feedback
        """
        while True:
            instr: Instr = yield self.instr_fifo.get()
            ctask_obj:AccTaskSim = self.taskset.get_task(instr.job.task)
            cregion_obj:AccRegionSim = ctask_obj.regions[instr.region]# get current task & region
            if instr.preempt:#pay preemption ovhd
                #the swap-out ovhd is stored after the last region
                ltask_obj:AccTaskSim = self.taskset.get_task(instr.last_task)
                lregion_obj:AccRegionSim = ltask_obj.regions[instr.last_region]#last task & region
                preempt_ovhd = lregion_obj.so
                self.logger.info("[{}][Acc] preemption: old_task:{}, old_region:{}, new_task:{}"
                                  .format(self.env.now,instr.last_task,instr.last_region,instr.job.task))
                yield self.env.timeout(preempt_ovhd)
            elif instr.resume:#pay resume ovhd
                #the swap-in ovhd is stored before this regiom
                resume_ovhd = cregion_obj.si
                self.logger.info("[{}][Acc] resume: old_task:{}, new_task:{}, new_region:{}"
                                  .format(self.env.now,instr.last_task,instr.job.task,instr.region))
                yield self.env.timeout(resume_ovhd)
            #pay execution time
            exec_time = cregion_obj.exec_time
            self.logger.debug("[{}][Acc] exec: task:{}, job:{}, region:{}"
                                  .format(self.env.now,instr.job.task,instr.job.job_id,instr.region))
            yield self.env.timeout(exec_time)
            #send feedback
            fb = Feedback(instr.job,instr.region)
            yield self.feedback_fifo.put(fb)
    def run(self):
        self.env.process(self._acc())

class SimManager:
    '''init and handle the simpy env'''
    def __init__(self,sche_config:ScheConfig,taskset:AccTasksetSim,
                 sim_time=220000000,
                 logger_enable = False, logger_name="logger",log_path=None,log_level=logging.INFO):
        #input params
        self.sche_config:ScheConfig = sche_config
        self.taskset:AccTasksetSim = taskset
        self.sim_time = sim_time
        #simulation envs
        self.env = simpy.Environment()
        self.task_release_fifo = simpy.Store(self.env)#at most n tasks are released
        self.instr_fifo = simpy.Store(self.env,capacity=1)
        self.feedback_fifo = simpy.Store(self.env,capacity=1)
        #logging
        self.logger = init_logger(logger_name,log_path,log_level,enable=logger_enable)
        #components
        self.job_generator = JobGenerator(self.env,self.taskset,
                                          self.task_release_fifo,
                                          self.logger)
        self.scheduler = Scheduler(self.sche_config,self.env,self.taskset,
                                   self.task_release_fifo,self.instr_fifo,self.feedback_fifo,
                                   self.logger)
        self.accelerator = Accelerator(self.env,self.taskset,
                                       self.instr_fifo,self.feedback_fifo,
                                       self.logger)
        self.job_generator.run()
        self.scheduler.run()
        self.accelerator.run()
    def run(self):
        """
        run the simulation and try-catch the valueError of deadline miss
        return bool value of the simulation success
        """
        try:
            self.env.run(until=self.sim_time)
        except ValueError as e:
            if "deadline miss" in str(e):
                return False
            else:
                raise  # re-raise unexpected ValueErrors
        return True
        


if __name__ == '__main__':
    config = AccConfig.from_json("/home/shixin/RTSS2025_AE/DERCA/DERCA_SW/configs/acc_config.json")
    # w1=Workload()
    debug_print('decompose_NN')
    # w1.decompose_NN([[1024,8192,1024],[1024,8192,1024]],config)
    w1=Workload([[1024,8192,1024],[1024,8192,1024]],config,'Task1')
    debug_print('apply strategy')
    s1 = StrategyFlexible()
    # s1 = StrategyNonPreemptive()
    # s1 = StrategyLayerwise()
    s1.from_workload(w1)
    print(s1.ID)

    w2 = Workload([[1024,8192,1024],[1024,8192,1024]],config,'Task2')
    s2 = StrategyFlexible()
    s2.from_workload(w2)

    debug_print('form taskset')
    taskset = AccTaskset([s1,s2],[0.7,0.2])
    for task in taskset.tasks: print(task.ID)

    # print('begin sche analysis')
    # ana = schedulability_analyzer(taskset)
    # ana.schedulability_test()
    # print('sche analysis:',ana.sche_test_success)

    print('begin PPP')
    PPP = PP_placer(taskset)
    TS = PPP.PP_placement()
    print("PPP success:",PPP.PPP_success)
    print(TS.tasks)

    ts_sim = AccTasksetSim(taskset)
    print(ts_sim)


    sche_config = ScheConfig.from_json("/home/shixin/RTSS2025_AE/DERCA/DERCA_SW/configs/sche_config.json")

    sim_manager = SimManager(
        sche_config=sche_config,
        taskset=ts_sim,
        sim_time=lcm(ts_sim.periods),
        logger_name='test',log_level=logging.INFO,log_path='test.log'
    )
    sim_manager.run()