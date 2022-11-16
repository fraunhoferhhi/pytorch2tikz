from typing import Tuple, Union
import numpy as np
from torch import Tensor, nn
import torch

from .block_factory import BlockFactory
from .block_sequence import BlockSequence

class Architecure:
    def __init__(self,
                 start_size=(8, 64, 64),
                 block_offset=8,
                 scale_factor=0.8,
                 image_path='./input_{i}.png',
                 ignore_layers=['batchnorm']) -> None:
        self._block_factory = BlockFactory(start_size, block_offset, scale_factor)
        self._blocks = BlockSequence(self._block_factory, ignore_layers)

        self.image_path = image_path

        self.inputs = []

        self._last_data_ptr = None
        self._last_block = None
    
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
        print(type(module), self._last_data_ptr, in_ptr, self._last_data_ptr == in_ptr)
        if self._last_data_ptr is not None:
            last_in_equals_out = self._last_data_ptr == in_ptr or 'activation' in str(type(module))
            pass
        self._last_data_ptr = out_ptr

        # set inputs
        if not last_in_equals_out:
            self._blocks.append_input(input, module)

        # check if tensor shape is equal to previous tensor shape, if not start new grouped blocks
        try:
            same_depth = torch.allclose(torch.tensor(in_shape), torch.tensor(out_shape))
        except  RuntimeError:
            same_depth = False

        # if not same_depth add gap
        if not same_depth:
            self._block_factory.add_gap()
            self._block_factory.scale(np.array(()))

        # add current module to blocks
        self._blocks.append(module)

        self._tensor_size = None

    def finalize(self) -> str:
        out = ''
        for b in self._blocks:
            out += f'\n{b}'
        return out