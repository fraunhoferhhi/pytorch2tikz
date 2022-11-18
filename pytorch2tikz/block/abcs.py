from enum import Enum
from typing import Iterable, Tuple
import numpy as np
from abc import abstractmethod

from ..constants import DEFAULT_VALUE, DIM_FACTOR, CM_FACTOR, OFFSET, COLOR, PICTYPE

class TexElement:
    @property
    @abstractmethod
    def tex(self):
        return ''
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    def __str__(self) -> str:
        return self.tex

class Block(TexElement):
    
    def __init__(self,
                 name,
                 fill: COLOR = COLOR.LINEAR,
                 bandfill: COLOR = None,
                 pictype = PICTYPE.BOX,
                 opacity = 0.7,
                 size = (10,40,40),
                 default_size = DEFAULT_VALUE * DIM_FACTOR,
                 dim = 3,
                 scale_factor = np.zeros(3),
                 offset = (0,0,0),
                 to = (0,0,0),
                 caption = " ",
                 xlabel = True,
                 ylabel = False,
                 zlabel = True,
                 is_input = False) -> None:
        super().__init__()
        self.name = name
        self.pictype = pictype
        self.offset = np.array(offset)
        self.to = to
        self.is_input = is_input
        self.looped = False
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

        self.args = {
            "fill": fill,
            "opacity": opacity,
            "caption": caption
        }

        self.scale_factor = scale_factor
        self._default_size = default_size
        self._dim = dim
        self.dim = dim

        self.size = size
        
        if bandfill is not None:
            self.args["bandfill"] = bandfill

    @property
    def dim(self) -> int:
        return self._dim
    
    @dim.setter
    def dim(self, dim: int):
        self.default_size = [self._default_size] * 3
        self.default_size[-dim:] = [None] * dim

    @property
    def size(self) -> Tuple[int]:
        return tuple((self.args['width'], self.args['height'], self.args['depth']))
    
    @size.setter
    def size(self, size: Iterable[int]):
        for i, dim in enumerate(['width', 'height', 'depth']):
            if self.default_size[i] is None:
                self.args[dim] = size[i]
            else:
                self.args[dim] = self.default_size[i]

    @property
    def tex(self) -> str:
        if self.dim == 3 and self.xlabel:
            self.args['xlabel'] = f'{{{self.size[0]},}}'
        if self.dim >= 2 and self.ylabel:
            self.args['ylabel'] = f'{self.size[1]}'
        if self.dim >= 1 and self.zlabel:
            self.args['zlabel'] = f'{self.size[2]}'

        args = ''
        for k, v in self.args.items():
            if isinstance(v, Enum):
                v = v.value
            elif type(v) in [tuple, list] and isinstance(v[0], Enum):
                v = [i.value for i in v]
            elif k in ['width', 'height', 'depth']:
                i = ['width', 'height', 'depth'].index(k)
                v = max(v / DIM_FACTOR + v / DIM_FACTOR * self.scale_factor[i], DEFAULT_VALUE)

            if type(v) in [tuple, list]:
                args += f'\n        {k}={{{",".join(v)}}},'
            else:
                args += f'\n        {k}={v},'
        
        args = args[:-1]

        return f"""
\pic[shift={{{tuple(self.offset / CM_FACTOR)}}}] at {self.to}
    {{{self.pictype.value}={{
        name={self.name},{args}
        }}
    }};
"""
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}< {self.name}, {self.size}, dim={self.dim}>'


class FlatBlock(Block):
    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(name, dim=max(1, dim-1), **kwargs)

class Connection(TexElement):
    
    def __init__(self, block1: Block, block2: Block, backwards = False) -> None:
        super().__init__()
        self.block1 = block1
        self.block2 = block2
        self.backwards = backwards
        self.max_block = 0 if block1.size[2] > block2.size[2] else 1
        self.offset = max(block1.size[2], block2.size[2]) / DIM_FACTOR / CM_FACTOR / 2. * -1 - OFFSET

    @property
    def tex(self) -> str:
        if self.backwards:
            return f"""
\coordinate ({self.block1.name}-{self.block2.name}-1) at ($ ({self.block1.name}-padded-east) - (0,0,{self.offset}) $);
\coordinate ({self.block1.name}-{self.block2.name}-2) at ($ ({self.block2.name}-padded-west) - (0,0,{self.offset}) $);
\draw [connection]  ({self.block1.name}-east) -- ({self.block1.name}-padded-east) -- node {{\midarrow}}({self.block1.name}-{self.block2.name}-1) -- node {{\midarrow}}({self.block1.name}-{self.block2.name}-2) -- node {{\midarrow}}({self.block2.name}-padded-west) -- ({self.block2.name}-west);
"""
        else:
            return f"""\draw [connection] ({self.block1.name}-east) -- node {{\midarrow}} ({self.block2.name}-west);"""
    
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self.block1.name}, {self.block2.name}>'