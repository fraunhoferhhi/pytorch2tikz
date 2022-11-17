from enum import Enum
from typing import Iterable, Tuple
import numpy as np
from abc import abstractmethod

from ..constants import CM_FACTOR, COLOR, PICTYPE

class TexElement:
    @property
    @abstractmethod
    def tex(self):
        return ''
    
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.tex

class Block(TexElement):
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
                self.args[dim] = size[i] * self.scale_factor
            else:
                self.args[dim] = self.default_size[i]

    @property
    def tex(self) -> str:
        args = ''
        for k, v in self.args.items():
            if isinstance(v, Enum):
                v = v.value
            elif type(v) in [tuple, list] and isinstance(v[0], Enum):
                v = [i.value for i in v]

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
    
    def __init__(self,
                 name,
                 fill: COLOR = COLOR.LINEAR,
                 bandfill: COLOR = None,
                 pictype = PICTYPE.BOX,
                 opacity = 0.7,
                 size = (10,40,40),
                 default_size = 2,
                 dim = 3,
                 scale_factor = 1,
                 offset = (0,0,0),
                 to = (0,0,0),
                 caption = " ",
                 xlabel: Iterable[int] = None,
                 zlabel: int = None,
                 is_input = False) -> None:
        super().__init__()
        self.name = name
        self.pictype = pictype
        self.offset = np.array(offset)
        self.to = to
        self.is_input = is_input


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
        
        if xlabel is not None:
            if type(xlabel) in [tuple, list]:
                self.size[0] = [self.size[0]] * len(xlabel)
            self.args['xlabel'] = xlabel
        if zlabel is not None:
            self.args['zlabel'] = zlabel

class FlatBlock(Block):
    def __init__(self, name, dim=3, **kwargs) -> None:
        super().__init__(name, dim=max(1, dim-1), **kwargs)

class Connection(TexElement):
    
    def __init__(self, block1: Block, block2: Block, direction='lr') -> None:
        super().__init__()
        self.name1 = f'{block1.name}-east'
        self.name2 = f'{block2.name}-west'
        
        if direction != 'lr':
            tmp = self.name1
            self.name1 = self.name2
            self.name2 = tmp

    @property
    def tex(self) -> str:
        return f"""\draw [connection] ({self.name1}) -- node {{\midarrow}} ({self.name2});"""