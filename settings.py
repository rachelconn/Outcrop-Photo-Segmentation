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
VALID_MODEL_TYPES = ['isgeological', 'structuretype']
MODEL_TYPE = 'structuretype'
MODEL_DIR = './models/structuretype'

N_LAYERS = 101
STRIDE = 8
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
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

assert MODEL_TYPE in VALID_MODEL_TYPES, f'Provided MODEL_TYPE {MODEL_TYPE} is not valid.\nValid model types: {VALID_MODEL_TYPES}.'
MODEL_TYPE_N_CLASSES = {
    'isgeological': 2,
    'structuretype': 5,
}
N_CLASSES = MODEL_TYPE_N_CLASSES[MODEL_TYPE]

if MODEL_TYPE == 'isgeological':
    # CLASS_WEIGHTS = Tensor(np.array([1.89149931, 0.83934196]))
    CLASS_WEIGHTS = Tensor(np.array([1.5149931, 1]))
else:
    # CLASS_WEIGHTS = Tensor(np.array([1.58730677, 0.42480622, 1.21652997, 1.13550899, 3.19167557]))
    CLASS_WEIGHTS = Tensor(np.array([1.5, 0.6, 1.21652997, 1.13550899, 2]))
    # CLASS_WEIGHTS = Tensor(np.ones(N_CLASSES))

LOG_DIR = './logdir'
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
