from .abcs import Connection, Block
from ..constants import DIM_FACTOR, CM_FACTOR, OFFSET

class LoopConnection(Connection):
    
    def __init__(self, block1: Block, block2: Block) -> None:
        super().__init__(block1, block2)
        self.max_block = 0 if block1.size[2] > block2.size[2] else 1
        self.offset = max(block1.size[2], block2.size[2]) / DIM_FACTOR / CM_FACTOR / 2. * -1 - OFFSET

    @property
    def tex(self) -> str:
        return f"""
\coordinate ({self.block1.name}-{self.block2.name}-1) at ($ ({self.block1.name}-padded-east) - (0,0,{self.offset}) $);
\coordinate ({self.block1.name}-{self.block2.name}-2) at ($ ({self.block2.name}-padded-west) - (0,0,{self.offset}) $);
\draw [connection]  ({self.block1.name}-east) -- ({self.block1.name}-padded-east) -- node {{\midarrow}}({self.block1.name}-{self.block2.name}-1) -- node {{\midarrow}}({self.block1.name}-{self.block2.name}-2) -- node {{\midarrow}}({self.block2.name}-padded-west) -- ({self.block2.name}-west);
"""