
from .abcs import Block
from ..constants import CM_FACTOR, COLOR, DIM_FACTOR

class ImgInputBlock(Block):

    def __init__(self, name, file_path, **kwargs) -> None:
        super().__init__(f'ImgInput_{name}', **kwargs)
        self.file_path = file_path
    
    @property
    def tex(self) -> str:
        return f"""
\\node[canvas is zy plane at x=0] ({self.name}) at {self.to} {{\includegraphics[width={self.args['depth'] / DIM_FACTOR / CM_FACTOR}cm, height={self.args['height'] / DIM_FACTOR / CM_FACTOR}cm]{{{self.file_path}}}}};
"""

class VecInputBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'VecInput_{name}', COLOR.VEC_INPUT, dim=1, **kwargs)

