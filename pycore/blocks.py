
from .tikzeng import *
import os.path as osp
from enum import Enum
from typing import Iterable

class COLOR(Enum):
    CONV = "\ConvColor"
    ACTIVATION = "\ActivationColor"
    POOL = "\PoolColor"
    LINEAR = "\LinearColor"
    EMBEDDING = "\LinearColor"
    NORM = "\\NormColor"
    LSTM = "\LstmColor"

class PICTYPE(Enum):
    BOX = "Box"
    RIGHTBANDEDBOX = "RightBandedBox"
    BALL = "Ball"


class Block:
    @property
    def tex(self) -> str:
        args = ''
        for k, v in self.args:
            if type(v) in [tuple, list]:
                args += f'\n        {k}={{{",".join(v)}}},'
            else:
                args += f'\n        {k}={v},'
        
        args = args[:-1]

        return f"""
\pic[shift={{{self.offset}}}] at {self.to}
    {{{self.pictype}={{{args}
        }}
    }};
"""
    
    def __init__(self,
                 name,
                 fill: COLOR,
                 bandfill: COLOR = None,
                 pictype = PICTYPE.BOX,
                 opacity = 0.7,
                 size = (40,10,40),
                 offset = (0,0,0),
                 to = (0,0,0),
                 caption = " ",
                 xlabel: Iterable[int] = None,
                 zlabel: int = None) -> None:
        self.name = name
        self.pictype = pictype
        self.size = list(size)
        self.offset = offset
        self.to = to


        self.args = {
            "fill": fill,
            "opacity": opacity,

            "width": self.size[0],
            "height": self.size[1],
            "depth": self.size[2],
            
            "caption": caption
        }
        
        if bandfill is not None:
            self.args["bandfill"] = bandfill
        
        if xlabel is not None:
            if type(xlabel) in [tuple, list]:
                self.size[0] = [self.size[0]] * len(xlabel)
            self.args['xlabel'] = xlabel
        if zlabel is not None:
            self.args['zlabel'] = zlabel
    
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.tex()

class Begin(Block):
    def __init__(self) -> None:
        pass

    def tex(self) -> str:
        pathlayers = osp.join(osp.dirname(__file__), 'layers')
        
        return f"""
\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage{{import}}
\subimport{{{pathlayers}}}{{init}}
\\usetikzlibrary{{positioning}}
\\usetikzlibrary{{3d}} %for including external image

\def\ConvColor{{rgb:yellow,5;red,2.5;white,5}}
\def\LinearColor{{rgb:blue,5;red,2.5;white,5}}
\def\LstmColor{{rgb:yellow,5;red,2.5;white,5}}
\def\ActivationColor{{rgb:yellow,5;red,5;white,5}}
\def\PoolColor{{rgb:red,1;black,0.3}}
\def\\NormColor{{rgb:red,1;black,0.3}}
\def\\UnpoolColor{{rgb:blue,2;green,1;black,0.3}}
\def\FcReluColor{{rgb:blue,5;red,5;white,4}}
\def\SoftmaxColor{{rgb:magenta,5;black,7}}
\def\SumColor{{rgb:blue,5;green,15}}

\\newcommand{{\copymidarrow}}{{\\tikz \draw[-Stealth,line width=0.8mm,draw={{rgb:blue,4;red,1;green,1;black,3}}] (-0.3,0) -- ++(0.3,0);}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=\edgecolor,opacity=0.7]
\\tikzstyle{{copyconnection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw={{rgb:blue,4;red,1;green,1;black,3}},opacity=0.7]
"""

class End(Block):
    def __init__(self) -> None:
        pass

    def tex(self) -> str:
        return """
\end{tikzpicture}
\end{document}
"""

class Connection(Block):
    
    def __init__(self, name1, name2) -> None:
        self.name1 = name1
        self.name2 = name2

    def tex(self) -> str:
        return f"""\draw [connection] ({self.name1}-east) -- node {{\midarrow}} ({self.name2}-west);"""

class InputBlock(Block):

    def __init__(self, name, file_path, **kwargs) -> None:
        super().__init__(f'Input_{name}', None, **kwargs)

        self.file_path = file_path
    
    def tex(self) -> str:
        return f"""
\\node[canvas is zy plane at x=0] ({self.name}) at {self.to} {{\includegraphics[width={self.args['width']}cm, height={self.args['height']}cm]{{{self.file_path}}}}};
"""

class ConvBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Conv_{name}', COLOR.CONV, **kwargs)

class ConvActivation(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(name, COLOR.CONV, bandfill=COLOR.ACTIVATION, pictype=PICTYPE.RIGHTBANDEDBOX, **kwargs)

class PoolBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Pool_{name}', COLOR.POOL, **kwargs)

class LinearBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Linear_{name}', COLOR.LINEAR, **kwargs)

class EmbeddingBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Linear_{name}', COLOR.EMBEDDING, **kwargs)

class NormBlock(Block):

    def __init__(self, name, **kwargs) -> None:
        super().__init__(f'Norm_{name}', COLOR.NORM, **kwargs)

class LSTMBlock(Block):

    def __init__(self, name, fill: COLOR, **kwargs) -> None:
        super().__init__(f'LSTM_{name}', fill, **kwargs)

#define new block
def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]

def block_Res( num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5 ):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset, 
            to="({}-east)".format( botton ),   
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ) 
                )
            ]
        botton = name
        lys+=ly
    
    lys += [
        to_skip( of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys


