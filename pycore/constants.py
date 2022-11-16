from enum import Enum

class COLOR(Enum):
    CONV = "\ConvColor"
    ACTIVATION = "\ActivationColor"
    DROPOUT = "\DropoutColor"
    POOL = "\PoolColor"
    
    VEC_INPUT = "\VecInputColor"
    LINEAR = "\LinearColor"
    EMBEDDING = "\EmbeddingColor"
    
    NORM = "\\NormColor"
    LSTM = "\LstmColor"

class PICTYPE(Enum):
    BOX = "Box"
    RIGHTBANDEDBOX = "RightBandedBox"
    BALL = "Ball"

CM_FACTOR = 5