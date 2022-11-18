import numpy as np
from torch import Tensor, nn
from torchvision.utils import save_image
from typing import Tuple, Union, Iterable
import os.path as osp

from .abcs import Block
from .inputs import ImgInputBlock, VecInputBlock
from ..mapping import BLOCK_MAPPING
from ..constants import DEFAULT_VALUE, DIM_FACTOR

class BlockFactory:
    def __init__(self,
                 offset=8,
                 height_depth_factor=0.8,
                 width_factor=0.8,
                 linear_factor=0.8,
                 image_path='input_{i}.png') -> None:
        self.offset = offset
        self.height_depth_factor = height_depth_factor
        self.width_factor = width_factor
        self.linear_factor = linear_factor
        self.image_path = image_path

        self.to = (0,0,0)
        self.current_offset = np.zeros(3)
        self.last_block_id = 0
        self.last_block_size = None
        self.scale_factor = None
    
    def _get_block_type(self, module: nn.Module, dim=None) -> Tuple[type, int]:
        if module.__class__.__name__.lower().endswith('d'):
            dim = int(module.__class__.__name__[-2]) + 1
        
        for key, value in BLOCK_MAPPING.items():
            if key in str(type(module)):
                return value, dim
        
        raise Exception(f'could not found Block for {module}')

    def create(self, block: Union[type, Block], i: int, output_shape: Iterable[int]) -> Block:
        kwargs = {}
        dim = None

        # set dim and if block is not a type get its type
        if not isinstance(block, type):
            block, dim = self._get_block_type(block)

        if dim is None:
            if len(output_shape) < 3:
                dim = 1
            else:
                dim = len(output_shape) - 1
        
        # set size
        size = np.array([DEFAULT_VALUE, DEFAULT_VALUE, DEFAULT_VALUE])
        size[-dim:] = np.array(output_shape)[-dim:]
        if self.last_block_size is not None:
            _size = size.copy()
            _size[:-dim] = 0
            scale_diff = _size / self.last_block_size
            scale_diff[scale_diff == 1] = 0

            width_factor = -self.width_factor / 2 / 2

            scale_factor = scale_diff * np.array([width_factor, self.height_depth_factor, self.height_depth_factor])

            if self.scale_factor is not None and scale_factor[0] != 0 and self.scale_factor[0] != 0:
                self.scale_factor[0] -= self.scale_factor[0] * scale_factor[0] * 2
            elif self.scale_factor is not None and scale_factor[0] != 0:
                self.scale_factor[0] = scale_factor[0]
    
            if self.scale_factor is not None and np.all(scale_factor < 4):
                self.scale_factor[1:] += scale_factor[1:]
            # if scaling is too large use linear_factor for reduction
            elif not np.all(scale_factor < 4):
                self.scale_factor[1:] = -self.linear_factor
            else:
                self.scale_factor = np.clip(scale_factor, -1, 1)
        
        if self.scale_factor is not None:
            kwargs['scale_factor'] = self.scale_factor.copy()
            
    
        kwargs['dim'] = dim
        
        new_block: Block = block(
                 i,
                 size = size,
                 offset = self.current_offset,
                 to = self.to, **kwargs)

        self.to = f'({new_block.name}-east)'
        self.last_block_id = i
        self.last_block_size = size
        self.current_offset = np.zeros(3)
        return new_block
    
    def create_input(self, x: Tensor) -> Block:
        if isinstance(self.to, Block):
            to = f'({self.to.name}-east)'
        else:
            to = self.to

        current_offset = np.array((-self.current_offset[0], self.offset, 0.0))

        offset = tuple(self.current_offset + current_offset)

        if x.ndim > 3:
            im_path = self.image_path.replace('{i}', str(self.last_block_id + 1))
            save_image(x[0], im_path)

            new_block = ImgInputBlock(self.last_block_id + 1,
                                      osp.split(im_path)[1],
                                      to=to,
                                      offset=offset,
                                      size=np.array(x.shape[-3:]))
        else:
            new_block = VecInputBlock(self.last_block_id + 1,
                                      to=to,
                                      offset=offset,
                                      size=np.array([DEFAULT_VALUE, DEFAULT_VALUE, x.shape[-1]]))

        return new_block
    
    def add_gap(self, axis=0):
        offset = np.zeros(3)
        offset[axis] = self.offset
        self.current_offset = self.current_offset + offset