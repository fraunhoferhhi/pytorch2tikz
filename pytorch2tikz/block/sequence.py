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

        self.block_factory = block_factory
        self.ignore_layers = ignore_layers

        self._fuseable_layers = ['conv', 'linear']
        self._seen_modules: Dict[nn.Module, Block] = {}
        self._block_map: Dict[str, Block] = {}
        self._connection_map: Dict[str, Connection] = {}
        self._connection_buffer: List[Tuple[Block, Union[Block, nn.Module]]] = []
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
        """translate modules in self.buffer to blocks in self.blocks and connections in self._connection_buffer"""
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

    def connect(self, block1: Union[str, Block], block2: Union[str, Block, nn.Module], conn_type: Connection = None) -> None:
        if not isinstance(block1, ImgInputBlock):
            if isinstance(block1, str):
                block1 = self._resolve_block(block1)
            if isinstance(block2, str):
                block2 = self._resolve_block(block2)
    
            self._connection_buffer.append((block1, block2, conn_type))
    
    def disconnect(self, block1: Union[str, Block], block2: Union[str, Block]):
        del self._connection_map[f'{self._resolve_block(block1).name}-{self._resolve_block(block2).name}']
    
    def get_block(self, name: str) -> Union[Block, None]:
        if name in self._block_map.keys():
            return self._block_map[name]
        else:
            return None
    
    def remove_block(self, block: Union[str, Block]):
        block = self._resolve_block(block)
        self.blocks.remove(block)
        del self._block_map[block.name]
        self.flush_connections()
        
        for c in self._connection_map:
            if c.block1.name == block.name or c.block2.name == block.name:
                self.disconnect(c.block1, c.block2)

    
    def _resolve_block(self, block: Union[Block, str]) -> Block:
        if isinstance(block, str):
            return self.get_block(block)
        else:
            return block

    def flush_connections(self) -> List[Connection]:
        for b1, b2, conn_type in self._connection_buffer:
            if isinstance(b1, nn.Module):
                b1 = self._seen_modules[b1]
            if isinstance(b2, nn.Module):
                b2 = self._seen_modules[b2]
            
            if conn_type is None:
                id1 = int(re.match('^\w+_(\d+)', b1.name).group(1))
                id2 = int(re.match('^\w+_(\d+)', b2.name).group(1))

                if id1 > id2:
                    conn_type = LoopConnection
                else:
                    conn_type = Connection

            if f'{b1.name}-{b2.name}' not in self._connection_map.keys():
                self._connection_map[f'{b1.name}-{b2.name}'] = conn_type(b1, b2)

        self._connection_buffer = []
    
    def __getitem__(self, key) -> Union[Block, Tuple[Block, Block]]:
        print(key)

    def __iter__(self) -> Generator[Block, None, None]:
        self.flush_connections()
        
        yield Begin(self._colors)
        for b in self.blocks:
            yield b
        for c in self._connection_map.values():
            yield c
        yield End()