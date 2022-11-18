from typing import Tuple, Union
import numpy as np
from torch import Tensor, nn
import torch

from .block.factory import BlockFactory
from .block.sequence import BlockSequence
from .constants import COLOR_VALUES

class Architecure:
    def __init__(self,
                 start_size=(8, 64, 64),
                 block_offset=8,
                 scale_factor=0.8,
                 image_path='./input_{i}.png',
                 ignore_layers=['batchnorm'],
                 colors=COLOR_VALUES) -> None:
        self._block_factory = BlockFactory(start_size, block_offset, scale_factor)
        self._block_sequence = BlockSequence(self._block_factory, ignore_layers)

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
        except  RuntimeError:
            same_depth = False

        # if not same_depth add gap
        if not same_depth and 'pooling' not in str(type(module)):
            self._block_sequence.add_gap()
            # if len(out_shape) == len(in_shape) and last_in_equals_out:
            #     scale = np.array(out_shape) / np.array(in_shape)

            #     if len(scale) > 3:
            #         scale = scale[1:]
            #         self._block_sequence.scale(scale)
            #     elif len(scale) < 3:
            #         print(scale, in_shape, out_shape)
            #         scale = np.array([1,1, 1 / scale[-1]])
            #         self._block_sequence.scale(scale)

        # add current module to blocks
        self._block_sequence.append(module, out_shape, len(in_shape) - 1)

        self._tensor_size = None

    def finalize(self) -> str:
        out = ''
        for b in self._block_sequence:
            out += f'\n{b}'
        return out