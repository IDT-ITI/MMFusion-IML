from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = 'logs'
_C.GPUS = (0,)
_C.WORKERS = 4
_C.BATCH_SIZE = 8
_C.LEARNING_RATE = 0.005
_C.SGD_MOMENTUM = 0.9
_C.WD = 0.0
_C.EPOCHS = 100
_C.WARMUP_EPOCHS = 2
_C.POLY_POWER = 0.9
_C.ACCUMULATE_ITERS = 1
_C.LOSS_SDC = False
_C.LOSS_MWS = False

# Cudnn parameters
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'cmnext'
_C.MODEL.BACKBONE = 'CMNeXtMHSA-B2'
_C.MODEL.PRETRAINED = ''
_C.MODEL.MODALS = ('img', 'noiseprint', 'bayar', 'srm')
_C.MODEL.DETECTION = None
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.TRAIN_PHASE = 'localization'
_C.MODEL.NP_WEIGHTS = ''

# Dataset parameters
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN = []
_C.DATASET.VAL = []
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.IMG_SIZE = None
_C.DATASET.CLASS_WEIGHTS = None


def update_config(cfg, file):
    cfg.defrost()

    cfg.merge_from_file(file)
    cfg.freeze()
    return cfg


