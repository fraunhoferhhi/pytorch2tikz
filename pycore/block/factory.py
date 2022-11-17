import numpy as np
from torch import Tensor, nn
from torchvision.utils import save_image
from typing import Tuple, Union

from .abcs import Block, FlatBlock
from .inputs import ImgInputBlock, VecInputBlock
from ..mapping import BLOCK_MAPPING

class BlockFactory:
    def __init__(self,
                 start_size=(8, 64, 64),
                 offset=8,
                 scale_factor=0.8,
                 image_path='input_{i}.png') -> None:
        self.size = np.array(start_size)
        self.offset = offset
        self.scale_factor = scale_factor
        self.image_path = image_path

        self.to = (0,0,0)
        self.current_offset = np.zeros(3)
        self.last_block_id = 0
        self.last_block_dim = None
    
    def _get_block_type(self, module: nn.Module, dim=None) -> Tuple[type, int]:
        if module.__class__.__name__.lower().endswith('d'):
            dim = int(module.__class__.__name__[-2]) + 1
        
        for key, value in BLOCK_MAPPING.items():
            if key in str(type(module)):
                return value, dim
        
        raise Exception(f'could not found Block for {module}')

    def create(self, block: Union[type, Block], i: int, dim=None) -> Block:
        kwargs = {}
        if not isinstance(block, type):
            block, dim = self._get_block_type(block, dim)

        if dim is not None:
            kwargs['dim'] = dim
            print('here', block, kwargs['dim'])
        
        new_block: Block = block(
                 i,
                 size = self.size,
                 offset = self.current_offset,
                 to = self.to, **kwargs)

        self.to = f'({new_block.name}-east)'
        self.last_block_id = i
        self.current_offset = np.zeros(3)
        return new_block
    
    def create_input(self, x: Tensor) -> Block:
        if isinstance(self.to, Block):
            to = f'({self.to.name}-east)'
        else:
            to = self.to
    
        self.current_offset += np.array((-self.current_offset[0], self.size[1] + self.offset, 0.0))

        offset = tuple(self.current_offset)
        self.current_offset = np.zeros(3)

        if x.ndim > 3:
            im_path = self.image_path.replace('{i}', str(self.last_block_id + 1))
            save_image(x[0], im_path)

            new_block = ImgInputBlock(self.last_block_id + 1,
                                      im_path,
                                      to=to,
                                      offset=offset,
                                      size=self.size)
        else:
            new_block = VecInputBlock(self.last_block_id + 1,
                                      to=to,
                                      offset=offset,
                                      size=self.size)
        return new_block
    
    def add_gap(self, axis=0):
        offset = np.zeros(3)
        offset[axis] = self.offset
        self.current_offset = self.current_offset + offset
    
    def scale(self, scale: np.ndarray):
        self.size = self.size * (scale / self.scale_factor)