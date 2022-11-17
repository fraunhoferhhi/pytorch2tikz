from typing import Tuple, Iterable
from .blocks_abcs import Block2D
from .constants import COLOR

class ActivationBlock(Block2D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Act_{name}', fill=COLOR.ACTIVATION, **kwargs)



class PoolBlock(Block2D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Pool_{name}', scale_factor=0.8, fill=COLOR.POOL, **kwargs)



class NormBlock(Block2D):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Norm_{name}', fill=COLOR.NORM, **kwargs)


class DropoutBlock(Block2D):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Dropout_{name}', fill=COLOR.DROPOUT, **kwargs)