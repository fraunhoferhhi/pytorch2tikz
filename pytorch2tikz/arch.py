from typing import Tuple, Union
import numpy as np
from torch import Tensor, nn
import torch

from .block.abcs import Block, Connection
from .block.factory import BlockFactory
from .block.sequence import BlockSequence
from .constants import COLOR_VALUES
from .module_graph import create_module_graph

class Architecture:

    def __init__(self,
                 module: nn.Module,
                 block_offset=8,
                 height_depth_factor=0.8,
                 width_factor=0.8,
                 linear_factor=0.8,
                 image_path='./input_{i}.png',
                 ignore_layers=['batchnorm', 'flatten'],
                 colors=COLOR_VALUES) -> None:
        self._handles = []
        self.module = module

        _block_factory = BlockFactory(block_offset, height_depth_factor, width_factor, linear_factor, image_path)
        self._block_sequence = BlockSequence(_block_factory, ignore_layers, colors)

        self.inputs = []

        self._last_data_ptr = None
        self._last_block = None

        self.register_handles()
    
    def register_handles(self):
        root = create_module_graph(self.module)
        self.remove_handles()

        modules = []
        for c in root.bfs():
            if 'torch.nn.modules' in str(type(c.module)):
                handle = c.module.register_forward_hook(self)
                self._handles.append(handle)
                modules.append(c.module)
    
    def remove_handles(self):
        if len(self._handles) > 0:
            for h in self._handles:
                h.remove()
            self._handles = []
    
    def __call__(self, module: nn.Module, input: Union[Tensor, Tuple[Tensor]], output: Union[Tensor, Tuple[Tensor]]) -> None:
        # unpack tuple if output is tuple (e.g. for LSTMs)
        if type(output) == tuple:
            output = output[0]
        input = input[0]
    
        # get current shapes
        in_shape = input.shape
        out_shape = output.shape
        in_ptr = input.data_ptr()
        out_ptr = output.data_ptr()

        last_in_equals_out = False

        if self._last_data_ptr is not None:
            last_in_equals_out = self._last_data_ptr == in_ptr or 'activation' in str(type(module))

        self._last_data_ptr = out_ptr

        # set inputs
        if not last_in_equals_out:
            self._block_sequence.append_input(input, module)

        # check if tensor shape is equal to previous tensor shape, if not start new grouped blocks
        try:
            same_depth = torch.allclose(torch.tensor(in_shape), torch.tensor(out_shape))
        except RuntimeError:
            same_depth = False

        # if not same_depth add gap
        if not same_depth and 'pooling' not in str(type(module)):
            self._block_sequence.add_gap()

        # add current module to blocks
        self._block_sequence.append(module, out_shape)

        if 'pooling' in str(type(module)):
            self._block_sequence.add_gap()
    
    def remove_block(self, name: str):
        self._block_sequence.remove_block(name)

    def get_block(self, name: str) -> Block:
        return self._block_sequence.get_block(name)
    
    def remove_connection(self, block1: Union[Block, str], block2: Union[Block, str]):
        self._block_sequence.disconnect(block1, block2)
    
    def connect(self, block1: Union[Block, str], block2: Union[Block, str], conn_type=Connection):
        self._block_sequence.connect(block1, block2, conn_type)
    
    def disconnect(self, block1: Union[Block, str], block2: Union[Block, str]):
        self._block_sequence.disconnect(block1, block2)

    def get_tex(self) -> str:
        out = ''
        for b in self._block_sequence:
            out += f'\n{str(b)}'
        return out
    
    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            f.write(self.get_tex())
    
    def __repr__(self) -> str:
        out = 'Architecture[\n'
        for b in self._block_sequence:
            out += f'             {repr(b)},\n'
        out += ']\n'
        return out