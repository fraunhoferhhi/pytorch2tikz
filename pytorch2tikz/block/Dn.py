from .abcs import Block, FlatBlock
from ..constants import COLOR, PICTYPE

class ConvBlock(Block):

    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(f'Conv_{name}', fill=COLOR.CONV, dim=dim, caption=f'Conv{name}', **kwargs)

class ConvActivationBlock(Block):

    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(f'ConvAct_{name}', fill=COLOR.CONV, bandfill=COLOR.ACTIVATION, pictype=PICTYPE.RIGHTBANDEDBOX, dim=dim, caption=f'Conv{name}', **kwargs)

class DropoutBlock(FlatBlock):

    def __init__(self, name, xlabel = False, ylabel = False, zlabel = False, **kwargs) -> None:
        super().__init__(f'Dropout_{name}', fill=COLOR.DROPOUT, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

class ActivationBlock(FlatBlock):

    def __init__(self, name, xlabel = False, ylabel = False, zlabel = False, **kwargs) -> None:
        super().__init__(f'Act_{name}', fill=COLOR.ACTIVATION, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

class NormBlock(FlatBlock):

    def __init__(self, name, xlabel = False, ylabel = False, zlabel = False, **kwargs) -> None:
        super().__init__(f'Norm_{name}', fill=COLOR.NORM, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

class PoolBlock(FlatBlock):

    def __init__(self, name, xlabel = False, ylabel = False, zlabel = False, **kwargs) -> None:
        super().__init__(f'Pool_{name}', fill=COLOR.POOL, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)
