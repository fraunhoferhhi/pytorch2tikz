from typing import List, Generator, Dict
from torch import nn, Tensor
import numpy as np

from .factory import BlockFactory
from .abcs import Block, Connection
from .Dn import ConvActivationBlock
from .D1 import LinearActivationBlock
from .inputs import ImgInputBlock
from .tex import Begin, End

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
        self._seen_modules: Dict[nn.Module, Block] = {}
        self._added_gap = False

    def append(self, module: nn.Module, dim):
        """appends the modules buffer if module should not be ignored ans was not seen before. If module is not fuseable call self.flush()"""
        for l in self.ignore_layers:
            if l in str(type(module)):
                return

        if module in self._seen_modules.keys():
            block = self._seen_modules[module]
            for b in self.last_blocks:
                if not block.looped and not b.looped:
                    print('  ', b, block)
                    self.connect(b, block, backwards=True)

                block.looped = True
            self.last_blocks = [block]
            
            self.flush()
            return

        self.buffer.append((module, dim))
        
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

            print('created', input_block.__class__.__name__)

    def append_block(self, block: Block):
        self.blocks.append(block)
        for b in self.last_blocks:
            if not isinstance(b, ImgInputBlock) and self._added_gap:
                self.connect(b, block)
        self.last_blocks = [block]
        self._added_gap = False

    def flush(self):
        """translate modules in self.buffer to blocks in self.blocks and connections in self.connections"""
        if len(self.buffer) == 0:
            return

        first_entry, dim = self.buffer[0]

        # if buffer has only one entry add block
        if len(self.buffer) == 1:
            new_block = first_entry

        elif len(self.buffer) == 2:
            # if buffer has fuseable entries
            fuse_conv = 'conv' in str(type(first_entry))
            fuse_linear = 'linear' in str(type(first_entry))
            fuseable = (fuse_conv or fuse_linear) and 'activation' in str(type(self.buffer[1][0]))

            if fuse_conv and fuseable:
                _, dim = self.block_factory._get_block_type(first_entry)
                new_block = ConvActivationBlock
            elif fuse_linear and fuseable:
                new_block = LinearActivationBlock
            else:
                for m, dim in self.buffer:
                    new_block = self.block_factory.create(m, len(self.blocks), dim)
                    self._seen_modules[m] = new_block
                    self.append_block(new_block)
                self.buffer = []
                return
        else:
            raise Exception(f'buffer has untypical length of {len(self.buffer)}')

        new_block = self.block_factory.create(new_block, len(self.blocks), dim)
        self.append_block(new_block)
        self._seen_modules[first_entry] = new_block
    
        print('created', new_block.__class__.__name__.ljust(20), ' for ', first_entry)
        if len(self.buffer) > 1:
            for b, _ in self.buffer[1:]:
                print(' ' * 34, b)

        self.buffer = []


    def scale(self, scale: np.ndarray):
        if len(self.blocks) > 1:
            self.block_factory.scale(scale)
    
    def add_gap(self, axis=0):
        self._added_gap = True
        self.block_factory.add_gap(axis)

    def connect(self, block1, block2, backwards=False) -> None:
        if not isinstance(block1, ImgInputBlock) and\
           not isinstance(block2, ImgInputBlock) and\
           not self.has_connection(block1, block2):
            self.connections.append(Connection(block1, block2, backwards, self.block_factory.size[1]))

    def has_connection(self, block1, block2) -> bool:
        new_conn = Connection(block1, block2)
        for c in self.connections:
            if c.name1 == new_conn.name1 and c.name2 == new_conn.name2:
                return True
        return False

    def __iter__(self) -> Generator[Block, None, None]:
        yield Begin()
        for b in self.blocks:
            yield b
        for c in self.connections:
            yield c
        yield End()