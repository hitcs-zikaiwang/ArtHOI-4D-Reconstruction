import random
import numpy as np
import torch
from loguru import logger as loguru

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loguru.info(f"Random seed set to {seed}")