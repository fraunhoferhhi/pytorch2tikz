from .abcs import Block
from ..constants import COLOR, PICTYPE

class LinearBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        kwargs['dim'] = 1
        
        super().__init__(f'Linear_{name}',
                         fill=COLOR.LINEAR,
                         caption=f'Linear{name}',
                         **kwargs)

class LinearActivationBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        kwargs['dim'] = 1

        super().__init__(f'LinearAct_{name}',
                         fill=COLOR.LINEAR,
                         bandfill=COLOR.ACTIVATION,
                         pictype=PICTYPE.RIGHTBANDEDBOX,
                         caption=f'Linear{name}',
                         **kwargs)

class EmbeddingBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        kwargs['dim'] = 1

        super().__init__(f'Embedding_{name}',
                         fill=COLOR.EMBEDDING,
                         caption=f'Embedding{name}',
                         **kwargs)

class LSTMBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        kwargs['dim'] = 1

        super().__init__(f'LSTM_{name}',
                         fill=COLOR.LSTM,
                         caption=f'LSTM{name}',
                         **kwargs)