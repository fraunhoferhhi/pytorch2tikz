from typing import Tuple, Union
import numpy as np
from torch import Tensor, nn
import torch

from .block.abcs import Block
from .block.factory import BlockFactory
from .block.sequence import BlockSequence
from .constants import COLOR_VALUES
from .module_graph import create_module_graph

class Architecure:

    def __init__(self,
                 module: nn.Module,
                 start_size=(8, 64, 64),
                 block_offset=8,
                 scale_factor=0.8,
                 image_path='./input_{i}.png',
                 ignore_layers=['batchnorm', 'flatten'],
                 colors=COLOR_VALUES) -> None:
        self._handles = []
        self.module = module

        _block_factory = BlockFactory(start_size, block_offset, scale_factor)
        self._block_sequence = BlockSequence(_block_factory, ignore_layers, colors)

        self.image_path = image_path

        self.inputs = []

        self._last_data_ptr = None
        self._last_block = None

        self.register_handles()
    
    def register_handles(self):
        root = create_module_graph(self.module)
        self.remove_handles()

        modules = []
        for c in root.bfs():
            if 'torch' in str(type(c.module)):
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
        in_shape = input.squeeze().shape
        out_shape = output.squeeze().shape
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
    
    def get_block(self, name: str) -> Block:
        return self._block_sequence.get_block(name)

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