from .blocks_abcs import Block1D
from .constants import COLOR, PICTYPE

class LinearBlock(Block1D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Linear_{name}',
                         fill=COLOR.LINEAR,
                         **kwargs)

class LinearActivationBlock(Block1D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'LinearAct_{name}',
                         fill=COLOR.LINEAR,
                         bandfill=COLOR.ACTIVATION,
                         pictype=PICTYPE.RIGHTBANDEDBOX,
                         **kwargs)

class EmbeddingBlock(Block1D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Embedding_{name}',
                         fill=COLOR.EMBEDDING,
                         **kwargs)

class LSTMBlock(Block1D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Embedding_{name}',
                         fill=COLOR.LSTM,
                         **kwargs)