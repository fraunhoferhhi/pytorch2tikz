from typing import List, Generator
from torch import nn, Tensor
import numpy as np

from .block_factory import BlockFactory
from .blocks_abcs import Block, Connection
from .blocks_3D import ConvActivationBlock
from .blocks_1D import LinearActivationBlock
from .blocks_input import ImgInputBlock
from .blocks_tex import Begin, End
from .mapping import BLOCK_MAPPING

class BlockSequence:

    def __init__(self,
                 block_factory: BlockFactory,
                 ignore_layers: List[str]=['batchnorm']) -> None:

        self.buffer: List[nn.Module] = []
        self.blocks: List[Block] = []
        self.last_blocks: List[Block] = []
        self.connections: List[Connection] = []

        self.block_factory = block_factory
        self.ignore_layers = ignore_layers

        self._fuseable_layers = ['conv', 'linear']
        self._seen_modules = {}
        self._added_gap = False

    def append(self, module: nn.Module):
        """appends the modules buffer if module should not be ignored ans was not seen before. If module is not fuseable call self.flush()

        Args:
            module (nn.Module): Module to be appended to buffer
        """
        if module in self._seen_modules.keys():
            return

        for l in self.ignore_layers:
            if l in str(type(module)):
                return

        self.buffer.append(module)
        
        fuseable = False
        for l in self._fuseable_layers:
            if l in str(type(module)):
                fuseable = True
                break
        
        if not fuseable:
            self.flush()
    
    def append_input(self, x: Tensor, module: nn.Module):
        if module not in self._seen_modules.keys():
            input_block = self.block_factory.create_input(x)
            self.blocks.append(input_block)
            self.last_blocks.append(input_block)

            print('create', input_block.__class__.__name__)

    def append_block(self, block: Block):
        self.blocks.append(block)
        for b in self.last_blocks:
            if not isinstance(b, ImgInputBlock) and self._added_gap:
                self.connect(b, block)
        self.last_blocks = [block]
        self._added_gap = False

    def flush(self):
        """translate modules in self.buffer to blocks in self.blocks and connections in self.connections"""

        # if buffer has only one entry add block
        if len(self.buffer) == 1:
            new_block_type = self.get_block_type(self.buffer[0])

        elif len(self.buffer) == 2:
            # if buffer has fuseable entries
            fuse_conv = 'conv' in str(type(self.buffer[0]))
            fuse_linear = 'linear' in str(type(self.buffer[0]))
            fuseable = (fuse_conv or fuse_linear) and 'activation' in str(type(self.buffer[1]))

            if fuse_conv and fuseable:
                new_block_type = ConvActivationBlock
            elif fuse_linear and fuseable:
                new_block_type = LinearActivationBlock
            else:
                for m in self.buffer:
                    new_block = self.block_factory.create(self.get_block_type(m), len(self.blocks))
                    self._seen_modules[m] = new_block
                    self.append_block(new_block)
                self.buffer = []
                return
        else:
            raise Exception(f'buffer has untypical length of {len(self.buffer)}')

        print('create', new_block_type.__name__.ljust(20), '  for  ', self.buffer[0])
        if len(self.buffer) > 1:
            for b in self.buffer[1:]:
                print(' ' * 35, b)
        
        new_block = self.block_factory.create(new_block_type, len(self.blocks))
        self.append_block(new_block)
        self._seen_modules[self.buffer[0]] = new_block
        self.buffer = []
    
    def scale(self, scale: np.ndarray):
        if len(self.blocks) > 1:
            self.block_factory.scale(scale)
    
    def add_gap(self, axis=0):
        self._added_gap = True
        self.block_factory.add_gap(axis)

    def connect(self, block1, block2) -> None:
        if not isinstance(block1, ImgInputBlock) and not isinstance(block2, ImgInputBlock):
            self.connections.append(Connection(block1, block2))
    
    def get_block_type(self, module: nn.Module):
        for i, (key, value) in enumerate(BLOCK_MAPPING.items()):
            if key in str(type(module)):
                return value

    def __iter__(self) -> Generator[Block, None, None]:
        yield Begin()
        for b in self.blocks:
            yield b
        for c in self.connections:
            yield c
        yield End()