
from .blocks_abcs import Block3D, Block1D
from .constants import CM_FACTOR, COLOR

class ImgInputBlock(Block3D):

    def __init__(self, name, file_path, **kwargs) -> None:
        super().__init__(f'Input_{name}', **kwargs, is_input=True)
        self.file_path = file_path
    
    @property
    def tex(self) -> str:
        return f"""
\\node[canvas is zy plane at x=0] ({self.name}) at {self.to} {{\includegraphics[width={self.args['depth'] / CM_FACTOR}cm, height={self.args['height'] / CM_FACTOR}cm]{{{self.file_path}}}}};
"""

class VecInputBlock(Block1D):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'VecInput_{name}', COLOR.VEC_INPUT, is_input=True, **kwargs)

