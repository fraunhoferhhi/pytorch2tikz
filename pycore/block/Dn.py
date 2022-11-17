from .abcs import Block, FlatBlock
from ..constants import COLOR, PICTYPE

class ConvBlock(Block):

    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(f'Conv_{name}', fill=COLOR.CONV, dim=dim, **kwargs)

class ConvActivationBlock(Block):

    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(f'ConvAct_{name}', fill=COLOR.CONV, bandfill=COLOR.ACTIVATION, pictype=PICTYPE.RIGHTBANDEDBOX, dim=dim, **kwargs)

class DropoutBlock(FlatBlock):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Dropout_{name}', fill=COLOR.DROPOUT, **kwargs)

class ActivationBlock(FlatBlock):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Act_{name}', fill=COLOR.ACTIVATION, **kwargs)

class NormBlock(FlatBlock):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Norm_{name}', fill=COLOR.NORM, **kwargs)

class PoolBlock(FlatBlock):

    def __init__(self, name, scale_factor=0.8, **kwargs) -> None:
        super().__init__(f'Pool_{name}', fill=COLOR.POOL, scale_factor=scale_factor, **kwargs)
