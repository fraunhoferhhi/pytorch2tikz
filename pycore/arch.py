from typing import List, Tuple
import os.path as osp
import numpy as np
from torch import Tensor, nn, Size

from blocks import Block, Begin, ConvBlock, PoolBlock, Connection, End

class Architecure:

    def __init__(self,
                 start_size=(2, 64, 64),
                 pool_factor=0.8,
                 conv_factor=1.2) -> None:
        self._blocks: List[Block] = [
            Begin(),
        ]
        self._blocks_buffer: List[Tuple[nn.Module, Size]] = []
        self._size = np.array(start_size)
        self._tensor_size = None

        self.pool_reduction = np.array([1, pool_factor, pool_factor])
        self.conv_expansion = np.array([conv_factor, 1, 1])
    
    def __call__(self, module, x: Tensor) -> None:
        if self._tensor_size is None and isinstance(x, Tensor):
            self._tensor_size = x.shape
        # if isinstance(block, PoolBlock):
        #     self._size = self._size * self.pool_reduction
        # elif isinstance(block, ConvBlock):
        #     self._size = self._size * self.conv_expansion
        print(type(module), x[0].shape)

    def connect(self, block1, block2) -> None:
        self._blocks.append(Connection(block1, block2))
    
    def finish(self) -> None:
        self._blocks.append(End())