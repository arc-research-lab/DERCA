from typing import List, Optional
import math
import inspect
import logging
import sys
import random

def print_iters(workload, fields: List[str]=['layer','idx','is_preemptive','strategy']):
    """Print a list of AccIter object in pandas dataframe manner"""
    # Print header
    header = " | ".join(f"{f}".ljust(12) for f in fields)
    print(header)
    print("-" * len(header))
    # Print each row
    for iter in workload.iters:
        row = " | ".join(str(getattr(iter, f, "")).ljust(12) for f in fields)
        print(row)

def lcm_pair(a, b):
    return abs(a * b) // math.gcd(a, b)

def lcm(numbers):
    from functools import reduce
    return reduce(lcm_pair, numbers)

def debug_print(*args, sep=' ', end='\n'):
    # Get calling frame info
    frame = inspect.currentframe()
    outer = inspect.getouterframes(frame, 2)[1]
    caller = outer.function
    lineno = outer.lineno  # ← this gives the source line number
    # Combine arguments into text
    message = sep.join(str(arg) for arg in args)
    print(f"[line {lineno}, {caller}] {message}", end=end)

def init_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        enable: bool = True,
        fmt: str = "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S"
    ) -> logging.Logger:
        """set enable to False to disable the logger"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # avoid duplicated handlers if called multiple times
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        if not enable:
            # debug_print('disable logger')
            logger.handlers.clear()     # remove any handlers
            logger.disabled = True      # completely disable the logger
            logger.propagate = False
        if log_file is None:# Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        else:# File handler
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

def uunifast(n, U_total):
    """Return a List of utilizations, the sum of util is U_total, #elements is n"""
    utilizations = []
    sum_u = U_total
    for i in range(1, n):
        next_sum_u = sum_u * (random.random() ** (1 / (n - i)))
        utilizations.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    utilizations.append(sum_u)
    utilizations.sort(reverse=True)
    return utilizations

#generate DNN shapes for realistic workloads
def gen_transformer(batch:int,seq_length:int,embed_dim:int,
                    num_head:int,mlp_ratio:int,num_layer:int):
    """return (shape:list, #layer per transformer blk)"""
    shape = []
    assert embed_dim%num_head==0, 'Error: the embed dim is not int multiple of num_heads'
    head_dim = int(embed_dim/num_head)
    mlp_dim = embed_dim*mlp_ratio
    for _ in range(num_layer):
        # [seq * batch, embed_dim, embed_dim * 3, 1],
        # [seq * batch, head_dim, seq, heads],
        # [seq * batch, seq, head_dim, heads],
        # [seq * batch, embed_dim, embed_dim, 1],
        # [seq * batch, embed_dim, mlp_dim, 1],
        # [seq * batch, mlp_dim, embed_dim, 1],
        #process a layer
        shape.append([seq_length*batch,embed_dim,embed_dim*3])#QKV dim
        for _ in range(num_head):#QK
            shape.append([seq_length*batch,head_dim,seq_length])
        for _ in range(num_head):#(QK)V
            shape.append([seq_length*batch,seq_length,head_dim])
        shape.append([seq_length*batch,embed_dim,embed_dim])#proj
        shape.append([seq_length*batch,embed_dim,mlp_dim])#proj
        shape.append([seq_length*batch,mlp_dim,embed_dim])#proj
    return (shape,4+2*num_head)

def gen_deit_t():
    shape ,num_blk_per_layer = gen_transformer(6,196,192,3,4,12)
    return shape

def gen_bert_t():
    shape ,num_blk_per_layer = gen_transformer(6,512,128,2,4,2)
    return shape

def gen_bert_mi():
    shape ,num_blk_per_layer = gen_transformer(6,512,256,4,4,4)
    return shape

def gen_mlp_mixer():
    return [
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512],
            [512, 196, 256], [512, 256, 196], [196, 512, 2048], [196, 2048, 512]
        ]

def gen_pointnet():
    #assume batch = 1 #point=1024
    mm_shapes = [
    [64, 3, 1024],     # conv1: 3→64, kernel 1x3
    [64, 64, 1024],    # conv2: 64→64, 1x1
    [64, 64, 1024],    # conv3: 64→64, 1x1
    [128, 64, 1024],   # conv4: 64→128, 1x1
    [1024, 128, 1024], # conv5: 128→1024, 1x1
    [1024, 1024, 1],   # max-pool over N=1024 points
    [512, 1024, 1],    # fc1: 1024→512
    [256, 512, 1],     # fc2: 512→256
    [40, 256, 1]       # fc3: 256→40
    ]
    return mm_shapes