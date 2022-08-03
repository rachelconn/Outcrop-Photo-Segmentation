import logging
import numpy as np
from torch import Tensor

# TODO:
# Make sure mean and stdev are correct
# Add rotation, refine scale factors?
# Larger crop size, batch size

# Data settings
DATA_ROOT = 'e:/outcrop'
MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALES = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5)
CROP_SIZE = 513
IGNORE_LABEL = 255

# Model definition
N_CLASSES = 6
N_LAYERS = 101
STRIDE = 8
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
CLASS_WEIGHTS = Tensor(np.array([1.36519203, 0.70625046, 1.19515633, 1.15467191, 1.93585436, 1.09205763]))
BATCH_SIZE = 1
ITER_MAX = 100000
ITER_SAVE = 2000

LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = [0]

LOG_DIR = './logdir'
MODEL_DIR = './models/6class'
# NOTE: NUM_WORKERS has a huge effect on CPU memory usage when input images are large.
# Try lowering this value if you're running out of CPU memory.
NUM_WORKERS = 1

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
