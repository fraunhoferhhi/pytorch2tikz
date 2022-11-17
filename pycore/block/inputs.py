
from .abcs import Block
from ..constants import CM_FACTOR, COLOR

class ImgInputBlock(Block):

    def __init__(self, name, file_path, **kwargs) -> None:
        super().__init__(f'ImgInput_{name}', **kwargs, is_input=True)
        self.file_path = file_path
    
    @property
    def tex(self) -> str:
        return f"""
\\node[canvas is zy plane at x=0] ({self.name}) at {self.to} {{\includegraphics[width={self.args['depth'] / CM_FACTOR}cm, height={self.args['height'] / CM_FACTOR}cm]{{{self.file_path}}}}};
"""

class VecInputBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'VecInput_{name}', COLOR.VEC_INPUT, is_input=True, dim=1, **kwargs)
