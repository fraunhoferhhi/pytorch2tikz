from typing import Tuple, Iterable
from .blocks_abcs import Block2D
from .constants import COLOR

class ActivationBlock(Block2D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Act_{name}', fill=COLOR.ACTIVATION, **kwargs)



class PoolBlock(Block2D):
    @property
    def size(self) -> Tuple[int]:
        return tuple((self.args['width'], self.args['height'], self.args['depth']))
    
    @size.setter
    def size(self, size: Iterable[int]):
        self.args["width"] = self.default_size[0]
        self.args["height"] = size[1] * self.scale_factor
        self.args["depth"] = size[2] * self.scale_factor

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Pool_{name}', fill=COLOR.POOL, **kwargs)



class NormBlock(Block2D):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Norm_{name}', fill=COLOR.NORM, **kwargs)


class DropoutBlock(Block2D):
    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Dropout_{name}', fill=COLOR.DROPOUT, **kwargs)