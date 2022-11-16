import numpy as np
from torch import Tensor
from torchvision.utils import save_image

from .blocks_abcs import Block
from .blocks_input import ImgInputBlock, VecInputBlock


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
        self.current_offset = np.array((0,0,0))
        self.last_block_id = 0
    
    def create(self, block_class, i: int) -> Block:
        new_block = block_class(
                 i,
                 size = self.size,
                 scale_factor = self.scale_factor,
                 offset = self.current_offset,
                 to = self.to)
        self.to = new_block.name
        self.last_block_id = i
        return new_block
    
    def create_input(self, tensor: Tensor) -> Block:
        if isinstance(self.to, Block):
            to = f'({self.to.name}-east)'
        else:
            to = tuple(self.to)
        
        self.current_offset += np.array((0, self.size[1] + self.offset, 0))
        offset = tuple(self.current_offset)

        if tensor.ndim > 3:
            im_path = self.image_path.replace('{i}', str(self.last_block_id + 1))
            save_image(tensor[0], im_path)
            return ImgInputBlock(self.last_block_id + 1,
                         im_path,
                         to=to,
                         offset=offset,
                         size=self.size
                        )
        else:
            return VecInputBlock(self.last_block_id + 1,
                                 to=to,
                                 offset=offset,
                                 size=self.size)
    
    def add_gap(self, axis=0):
        offset = np.zeros(3)
        offset[axis] = self.offset
        self.current_offset += offset
    
    def scale(self, scale: np.ndarray):
        self.size *= scale