from typing import List, Generator, Dict, Tuple, Union
from torch import nn, Tensor
import numpy as np
import re

from .factory import BlockFactory
from .abcs import Block, Connection
from .Dn import ConvActivationBlock
from .D1 import LinearActivationBlock
from .inputs import ImgInputBlock
from .connections import LoopConnection
from .tex import Begin, End
from ..constants import COLOR_VALUES

LOG_CREATED = True

class BlockSequence:

    def __init__(self,
                 block_factory: BlockFactory,
                 ignore_layers: List[str]=['batchnorm', 'flatten'],
                 colors = COLOR_VALUES) -> None:

        self.buffer: List[nn.Module] = []
        self.blocks: List[Block] = []
        self.last_block: Block = None
        self.connections: List[Tuple[Block, Union[Block, nn.Module]]] = []

        self.block_factory = block_factory
        self.ignore_layers = ignore_layers

        self._fuseable_layers = ['conv', 'linear']
        self._seen_modules: Dict[nn.Module, Block] = {}
        self._block_map: Dict[str, Block] = {}
        self._connection_map: Dict[str, Connection] = {}
        self._added_gap = False
        self._colors = colors

    def append(self, module: nn.Module, output_shape: Tuple[int]):
        """appends the modules buffer if module should not be ignored ans was not seen before. If module is not fuseable call self.flush()"""
        for l in self.ignore_layers:
            if l in str(type(module)):
                return

        if module in self._seen_modules.keys():
            mod_block = self._seen_modules[module]
            if self.last_block is not None:
                if not mod_block.looped and not self.last_block.looped:
                    self.connect(self.last_block, mod_block)

                mod_block.looped = True
            self.last_block = mod_block

            self.flush()
            return

        self.buffer.append((module, output_shape))
        
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
            self._block_map[input_block.name] = input_block
            self.connect(input_block, module)


    def append_block(self, block: Block):
        self.blocks.append(block)
        self._block_map[block.name] = block
        if self.last_block is not None and self._added_gap:
            self.connect(self.last_block, block)

        self.last_block = block
        self._added_gap = False

    def flush(self):
        """translate modules in self.buffer to blocks in self.blocks and connections in self.connections"""
        if len(self.buffer) == 0:
            return

        first_entry, out_shape = self.buffer[0]

        # if buffer has only one entry add block
        if len(self.buffer) == 1:
            new_block = first_entry

        elif len(self.buffer) == 2:
            # if buffer has fuseable entries
            fuse_conv = 'conv' in str(type(first_entry))
            fuse_linear = 'linear' in str(type(first_entry))
            fuseable = (fuse_conv or fuse_linear) and 'activation' in str(type(self.buffer[1][0]))

            if fuse_conv and fuseable:
                new_block = ConvActivationBlock
            elif fuse_linear and fuseable:
                new_block = LinearActivationBlock
            else:
                for m, out_shape in self.buffer:
                    if m not in self._seen_modules.keys():
                        new_block = self.block_factory.create(m, len(self.blocks), out_shape)
                        self._seen_modules[m] = new_block
                        self.append_block(new_block)
                self.buffer = []
                return
        else:
            raise Exception(f'buffer has untypical length of {len(self.buffer)}')

        new_block = self.block_factory.create(new_block, len(self.blocks), out_shape)
        self.append_block(new_block)
        for mod, _ in self.buffer:
            self._seen_modules[mod] = new_block

        self.buffer = []
    
    def add_gap(self, axis=0):
        if self._added_gap == False:
            self._added_gap = True
            self.block_factory.add_gap(axis)

    def connect(self, block1: Union[str, Block], block2: Union[str, Block, nn.Module]) -> None:
        if isinstance(block1, str):
            block1 = self._block_map[block1]
        if isinstance(block2, str):
            block2 = self._block_map[block2]

        if not isinstance(block1, ImgInputBlock):
            self.connections.append((block1, block2))
    
    def disconnect(self, block1: str, block2: str):
        for c in self.connections:
            if isinstance(c[0], nn.Module) or isinstance(c[1], nn.Module):
                continue
    
            if c[0].name == block1 and c[1].name == block2:
                self.connections.remove(c)
    
    def get_block(self, name: str) -> Union[Block, None]:
        if name in self._block_map.keys():
            return self._block_map[name]
        else:
            return None
    
    def _parse_connections(self) -> List[Connection]:
        connections: List[Connection] = []
        added_connections: List[str] = []
        for b1, b2 in self.connections:
            if isinstance(b1, nn.Module):
                b1 = self._seen_modules[b1]
            if isinstance(b2, nn.Module):
                b2 = self._seen_modules[b2]
            
            if f'{b1.name}-{b2.name}' not in added_connections:
                id1 = int(re.match('^\w+_(\d+)', b1.name).group(1))
                id2 = int(re.match('^\w+_(\d+)', b2.name).group(1))

                if id1 > id2:
                    conn = LoopConnection(b1, b2)
                else:
                    conn = Connection(b1, b2)

                connections.append(conn)
                added_connections.append(f'{b1.name}-{b2.name}')
        self.connections = [(c.block1, c.block2) for c in connections]
        return connections

    def __iter__(self) -> Generator[Block, None, None]:
        yield Begin(self._colors)
        for b in self.blocks:
            yield b
        for c in self._parse_connections():
            yield c
        yield End()