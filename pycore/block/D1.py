from .abcs import Block
from ..constants import COLOR, PICTYPE

class LinearBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Linear_{name}',
                         fill=COLOR.LINEAR,
                         dim=1,
                         **kwargs)

class LinearActivationBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'LinearAct_{name}',
                         fill=COLOR.LINEAR,
                         bandfill=COLOR.ACTIVATION,
                         pictype=PICTYPE.RIGHTBANDEDBOX,
                         dim=1,
                         **kwargs)

class EmbeddingBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Embedding_{name}',
                         fill=COLOR.EMBEDDING,
                         dim=1,
                         **kwargs)

class LSTMBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'LSTM_{name}',
                         fill=COLOR.LSTM,
                         dim=1,
                         **kwargs)