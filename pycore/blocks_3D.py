from .blocks_abcs import Block3D
from .constants import COLOR, PICTYPE

class ConvBlock(Block3D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Conv_{name}', fill=COLOR.CONV, **kwargs)

class ConvActivationBlock(Block3D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'ConvAct_{name}', fill=COLOR.CONV, bandfill=COLOR.ACTIVATION, pictype=PICTYPE.RIGHTBANDEDBOX, **kwargs)