from typing import List, Generator
from torch import nn, Tensor

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

    def append(self, module: nn.Module):
        """appends the modules buffer if module should not be ignored ans was not seen before. If module is not fuseable call self.flush()

        Args:
            module (nn.Module): Module to be appended to buffer
        """
        if module in self._seen_modules.keys():
            return

        for l in self.ignore_layers:
            if l not in str(type(module)):
                self.buffer.append(module)
        
        fuseable = False
        for l in self._fuseable_layers:
            if l in str(type(module)):
                fuseable = True
                break
        
        if not fuseable:
            self.flush()
    
    def append_input(self, tensor: Tensor, module: nn.Module):
        if module not in self._seen_modules.keys():
            input_block = self.block_factory.create_input(input)
            self.blocks.append(input_block)
            self.last_blocks.append(input_block)

    def append_block(self, block: Block):
        self.blocks.append(block)
        for b in self.last_blocks:
            if not b.is_input:
                self.connect(b, block)
        self.last_blocks = [block]

    def flush(self):
        """translate modules in self.buffer to blocks in self.blocks and connections in self.connections"""
        # if buffer has only one entry add block
        if len(self.buffer) == 1:
            new_block_type = self.get_block_type(self.buffer[0])
            return
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
                raise Exception(f'unable to fuse {self.buffer[0].__class__.__name__} with {self.buffer[1].__class__.__name__}')
        else:
            raise Exception(f'buffer has untypical length of {len(self.buffer)}')

        new_block = self.block_factory.create(new_block_type, len(self.blocks))
        self.append_block(new_block)
        self.buffer = []
    
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