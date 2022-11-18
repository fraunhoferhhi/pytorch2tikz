from enum import Enum

class COLOR(Enum):
    EDGE = "\EdgeColor"
    CONV = "\ConvColor"
    ACTIVATION = "\ActivationColor"
    DROPOUT = "\DropoutColor"
    POOL = "\PoolColor"
    
    VEC_INPUT = "\VecInputColor"
    LINEAR = "\LinearColor"
    EMBEDDING = "\EmbeddingColor"
    
    NORM = "\\NormColor"
    LSTM = "\LstmColor"

COLOR_VALUES = {
    'CONV': '#ffd232',
    'ACTIVATION': "#ff6400",
    'DROPOUT': "#ff0000",
    'POOL': "#c80000",
    
    'VEC_INPUT': "#008000",
    'LINEAR': "#ff00ff",
    'EMBEDDING': "#000080",
    
    'NORM': "#c40000",
    'LSTM': "#000080",
    'EDGE': '#555555'
}

class PICTYPE(Enum):
    BOX = "Box"
    RIGHTBANDEDBOX = "RightBandedBox"
    BALL = "Ball"

CM_FACTOR = 5

DIM_FACTOR = 8
OFFSET = 5
DEFAULT_VALUE = 2