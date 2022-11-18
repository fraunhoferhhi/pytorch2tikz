from .abcs import TexElement
from typing import Dict
from ..utils import hex_to_tex_color
from ..constants import COLOR


class ColorBlock(TexElement):
    def __init__(self, colors: Dict[str, str]) -> None:
        super().__init__()
        self.colors = colors

    @property
    def tex(self) -> str:
        out = ''
        for k, v in self.colors.items():
            out += f'\def{COLOR[k].value}{{{hex_to_tex_color(v)}}}\n'
        return out

class PositionsBlock(TexElement):
    @property
    def tex(self) -> str:
        return f"""%Define nodes to be used outside on the pic object
        \coordinate (\\name-west)   at (0,            0,    0);
        \coordinate (\\name-east)   at (\LastEastx,   0,    0);
        \coordinate (\\name-north)  at (\LastEastx/2, \y/2, 0);
        \coordinate (\\name-south)  at (\LastEastx/2,-\y/2, 0);       
        \coordinate (\\name-anchor) at (\LastEastx/2, 0,    0);
        
        \coordinate (\\name-near) at   (\LastEastx/2, 0,    \z/2);
        \coordinate (\\name-far)  at   (\LastEastx/2, 0,   -\z/2);       
        
        \coordinate (\\name-nearwest) at (0,         0, \z/2);
        \coordinate (\\name-neareast) at (\LastEastx,0, \z/2);
        \coordinate (\\name-farwest)  at (0,         0,-\z/2);
        \coordinate (\\name-fareast)  at (\LastEastx,0,-\z/2);
        
        \coordinate (\\name-northeast) at (\\name-north-|\\name-east);
        \coordinate (\\name-northwest) at (\\name-north-|\\name-west);
        \coordinate (\\name-southeast) at (\\name-south-|\\name-east);
        \coordinate (\\name-southwest) at (\\name-south-|\\name-west);
        
        \coordinate (\\name-nearnortheast)  at (\LastEastx, \y/2, \z/2);
        \coordinate (\\name-farnortheast)   at (\LastEastx, \y/2,-\z/2);
        \coordinate (\\name-nearsoutheast)  at (\LastEastx,-\y/2, \z/2);
        \coordinate (\\name-farsoutheast)   at (\LastEastx,-\y/2,-\z/2);
        
        \coordinate (\\name-nearnorthwest)  at (0, \y/2, \z/2);
        \coordinate (\\name-farnorthwest)   at (0, \y/2,-\z/2);
        \coordinate (\\name-nearsouthwest)  at (0,-\y/2, \z/2);
        \coordinate (\\name-farsouthwest)   at (0,-\y/2,-\z/2);

        % padded
        \coordinate (\\name-padded-west)   at (-0.5,0,0);
        \coordinate (\\name-padded-east)   at (\LastEastx + 0.5, 0,0) ;
        \coordinate (\\name-padded-north)  at (\LastEastx/2, \y/2+0.5,0);
        \coordinate (\\name-padded-south)  at (\LastEastx/2,-\y/2-0.5,0);
        
        \coordinate (\\name-padded-near)   at (\LastEastx/2,0, \z/2+0.5);
        \coordinate (\\name-padded-far)    at (\LastEastx/2,0,-\z/2-0.5);       
        
        \coordinate (\\name-padded-nearwest) at (-0.5,          0,\z/2);
        \coordinate (\\name-padded-neareast) at (\LastEastx+0.5,0,\z/2);
        \coordinate (\\name-padded-farwest)  at (-0.5,          0,-\z/2);
        \coordinate (\\name-padded-fareast)  at (\LastEastx+0.5,0,-\z/2);
        
        \coordinate (\\name-padded-northeast) at (\\name-padded-north-|\\name-padded-east);
        \coordinate (\\name-padded-northwest) at (\\name-padded-north-|\\name-padded-west);
        \coordinate (\\name-padded-southeast) at (\\name-padded-south-|\\name-padded-east);
        \coordinate (\\name-padded-southwest) at (\\name-padded-south-|\\name-padded-west);
        
        \coordinate (\\name-padded-nearnortheast)  at (\LastEastx+0.5, \y/2+0.5, \z/2+0.5);
        \coordinate (\\name-padded-farnortheast)   at (\LastEastx+0.5, \y/2+0.5,-\z/2-0.5);
        \coordinate (\\name-padded-nearsoutheast)  at (\LastEastx+0.5,-\y/2-0.5, \z/2+0.5);
        \coordinate (\\name-padded-farsoutheast)   at (\LastEastx+0.5,-\y/2-0.5,-\z/2-0.5);
        
        \coordinate (\\name-padded-nearnorthwest)  at (-0.5, \y/2+0.5, \z/2+0.5);
        \coordinate (\\name-padded-farnorthwest)   at (-0.5, \y/2+0.5,-\z/2-0.5);
        \coordinate (\\name-padded-nearsouthwest)  at (-0.5,-\y/2-0.5, \z/2+0.5);
        \coordinate (\\name-padded-farsouthwest)   at (-0.5,-\y/2-0.5,-\z/2-0.5);
"""

class Begin(TexElement):
    def __init__(self, colors: Dict[str, str]) -> None:
        super().__init__()
        self.colors = ColorBlock(colors)
        self.positions = PositionsBlock()

    @property
    def tex(self) -> str:        
        return f"""
\documentclass[border=8pt, multi, tikz]{{standalone}}
\\usepackage{{import}}
\\usetikzlibrary{{quotes,arrows.meta}}
\\usetikzlibrary{{positioning, calc, 3d}}

{self.colors}

\\newcommand{{\midarrow}}{{\\tikz \draw[-Stealth,line width =0.8mm,draw=\EdgeColor] (-0.3,0) -- ++(0.3,0);}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This Block can draw small Ball
%Elementwise or reduction operations can be drawn with this
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\\tikzset{{Ball/.pic={{\\tikzset{{/sphere/.cd,#1}}

\pgfmathsetmacro{{\\r}}{{\\radius*\scale}}

\shade[ball color=\\fill,opacity=\opacity] (0,0,0) circle (\\r);
\draw (0,0,0) circle [radius=\\r] node[scale=4*\\r] {{\logo}};

\coordinate (\\name-anchor) at ( 0 , 0  , 0)  ;
\coordinate (\\name-east)   at ( \\r, 0  , 0) ;
\coordinate (\\name-west)   at (-\\r, 0  , 0) ;
\coordinate (\\name-north)  at ( 0 ,  \\r , 0);
\coordinate (\\name-south)  at ( 0 , -\\r, 0) ;

\path (\\name-south) + (0,-20pt) coordinate (caption-node) 
edge ["\\textcolor{{black}}{{\\bf \caption}}"'] (caption-node); %Ball caption

}},
/sphere/.search also={{/tikz}},
/sphere/.cd,
radius/.store       in=\\radius,
scale/.store        in=\scale,
caption/.store      in=\caption,
name/.store         in=\\name,
fill/.store         in=\\fill,
logo/.store         in=\logo,
opacity/.store      in=\opacity,
logo=$\Sigma$,
fill=green,
opacity=0.10,
scale=0.2,
radius=0.5,
caption=,
name=,
}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Block can draw simple block of boxes with custom colors. 
% Can be used for conv, deconv etc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\\tikzset{{Box/.pic={{\\tikzset{{/boxblock/.cd,#1}}
        \\tikzstyle{{box}}=[every edge/.append style={{pic actions, densely dashed, opacity=.7}},fill opacity=\opacity, pic actions,fill=\\fill]
        
        \pgfmathsetmacro{{\y}}{{\cubey*\scale}}
        \pgfmathsetmacro{{\z}}{{\cubez*\scale}}
   
        %Multiple concatenated boxes
        \\foreach[count=\i,%
                 evaluate=\i as \\xlabel using {{array({{\\boxlabels}},\i-1)}},% 
                 evaluate=\\unscaledx as \k using {{\\unscaledx*\scale+\prev}}, remember=\k as \prev (initially 0)] 
                 \\unscaledx in \cubex
        {{
            \pgfmathsetmacro{{\\x}}{{\\unscaledx*\scale}}
            \coordinate (a) at (\k-\\x , \y/2 , \z/2); 
            \coordinate (b) at (\k-\\x ,-\y/2 , \z/2); 
            \coordinate (c) at (\k    ,-\y/2 , \z/2); 
            \coordinate (d) at (\k    , \y/2 , \z/2); 
            \coordinate (e) at (\k    , \y/2 ,-\z/2); 
            \coordinate (f) at (\k    ,-\y/2 ,-\z/2); 
            \coordinate (g) at (\k-\\x ,-\y/2 ,-\z/2); 
            \coordinate (h) at (\k-\\x , \y/2 ,-\z/2); 
        
            \draw [box] 
                (d) -- (a) -- (b) -- (c) -- cycle     
                (d) -- (a) -- (h) -- (e) -- cycle
                %dotted edges
                (f) edge (g)
                (b) edge (g)
                (h) edge (g)    
            ;
            \path (b) edge ["\\xlabel"',midway] (c);
            
            \\xdef\LastEastx{{\k}} %\k persists as \LastEastx after loop 
        }}%Loop ends
        \draw [box] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box     
        
        \coordinate (a1) at (0 , \y/2 , \z/2);
        \coordinate (b1) at (0 ,-\y/2 , \z/2);
        \\tikzstyle{{depthlabel}}=[pos=0,text width=14*\z,text centered,sloped]       
        
        \path (c) edge ["\small\zlabel"',depthlabel](f); %depth label
        \path (b1) edge ["\ylabel",midway] (a1);  %height label
        
        
        \\tikzstyle{{captionlabel}}=[text width=15*\LastEastx/\scale,text centered]       
        \path (\LastEastx/2,-\y/2,+\z/2) + (0,-25pt) coordinate (cap) 
        edge ["\\textcolor{{black}}{{ \\bf \caption}}"',captionlabel](cap) ; %Block caption/pic object label
         
        {self.positions}
        
    }},
    /boxblock/.search also={{/tikz}},
    /boxblock/.cd,
    width/.store        in=\cubex,
    height/.store       in=\cubey,
    depth/.store        in=\cubez,
    scale/.store        in=\scale,
    xlabel/.store       in=\\boxlabels,
    ylabel/.store       in=\ylabel,
    zlabel/.store       in=\zlabel,
    caption/.store      in=\caption,
    name/.store         in=\\name,
    fill/.store         in=\\fill,
    opacity/.store      in=\opacity,
    fill={{rgb:red,5;green,5;blue,5;white,15}},
    opacity=0.4,
    width=2,
    height=13,
    depth=15,
    scale=.2,
    xlabel={{{{"","","","","","","","","",""}}}},
    ylabel=,
    zlabel=,
    caption=,
    name=,
}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Block can draw simple block of boxes with custom colors. 
% Can be used for conv, deconv etc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\\tikzset{{RightBandedBox/.pic={{\\tikzset{{/block/.cd,#1}}
                
        \\tikzstyle{{box}}=[every edge/.append style={{pic actions, densely dashed, opacity=.7}},fill opacity=\opacity, pic actions,fill=\\fill]
        
        \\tikzstyle{{band}}=[every edge/.append style={{pic actions, densely dashed, opacity=.7}},fill opacity=\\bandopacity, pic actions,fill=\\bandfill,draw=\\bandfill]
        
        \pgfmathsetmacro{{\y}}{{\cubey*\scale}}
        \pgfmathsetmacro{{\z}}{{\cubez*\scale}}

        %Multiple concatenated boxes	 	  	
        \\foreach[count=\i,%
                 evaluate=\i as \\xlabel using {{array({{\\boxlabels}},\i-1)}},% 
                 evaluate=\\unscaledx as \k using {{\\unscaledx*\scale+\prev}}, remember=\k as \prev (initially 0)] 
                 \\unscaledx in \cubex
        {{
            \pgfmathsetmacro{{\\x}}{{\\unscaledx*\scale}}
            \coordinate (a)     at (\k-\\x   , \y/2 , \z/2); 
            \coordinate (art)   at (\k-\\x/3 , \y/2 , \z/2); %a_right_third
            \coordinate (b)     at (\k-\\x   ,-\y/2 , \z/2); 
            \coordinate (brt)   at (\k-\\x/3 ,-\y/2 , \z/2); %b_right_third
            \coordinate (c)     at (\k      ,-\y/2 , \z/2); 
            \coordinate (d)     at (\k      , \y/2 , \z/2); 
            \coordinate (e)     at (\k      , \y/2 ,-\z/2); 
            \coordinate (f)     at (\k      ,-\y/2 ,-\z/2); 
            \coordinate (g)     at (\k-\\x   ,-\y/2 ,-\z/2); 
            \coordinate (h)     at (\k-\\x   , \y/2 ,-\z/2); 
            \coordinate (hrt)   at (\k-\\x/3 , \y/2 ,-\z/2); %h_right_third
            
            %fill box color 			
            \draw [box] 
                (d) -- (a) -- (b) -- (c) -- cycle     
                (d) -- (a) -- (h) -- (e) -- cycle;
            %dotted edges
            \draw [box]
                (f) edge (g)
                (b) edge (g)
                (h) edge (g);
            %fill band color    
            \draw [band] 
                (d) -- (art) -- (brt) -- (c) -- cycle     
                (d) -- (art) -- (hrt) -- (e) -- cycle;
            %draw edges again which were covered by band
            \draw [box,fill opacity=0] 
                (d) -- (a) -- (b) -- (c) -- cycle     
                (d) -- (a) -- (h) -- (e) -- cycle;            
            	
            \path (b) edge ["\\xlabel"',midway] (c);
            
            \\xdef\LastEastx{{\k}} %\k persists as \LastEastx after loop 
        }}%Loop ends
        \draw [box] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box
        \draw [band] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box 
        \draw [pic actions] (d) -- (e) -- (f) -- (c) -- cycle; %East face edges of last box     
        
        \coordinate (a1) at (0 , \y/2 , \z/2);
        \coordinate (b1) at (0 ,-\y/2 , \z/2);
        \\tikzstyle{{depthlabel}}=[pos=0,text width=14*\z,text centered,sloped]       
        
        \path (c) edge ["\small\zlabels"',depthlabel](f); %depth label
        \path (b1) edge ["\ylabel",midway] (a1);  %height label 	  
        
        \\tikzstyle{{captionlabel}}=[text width=15*\LastEastx/\scale,text centered] 
        \path (\LastEastx/2,-\y/2,+\z/2) + (0,-25pt) coordinate (cap) 
        edge ["\\textcolor{{black}}{{ \\bf \caption}}"',captionlabel] (cap); %Block caption/pic object label
         
        {self.positions}
    }},
    /block/.search also={{/tikz}},
    /block/.cd,
    width/.store        in=\cubex,
    height/.store       in=\cubey,
    depth/.store        in=\cubez,
    scale/.store        in=\scale,
    xlabel/.store       in=\\boxlabels,
    ylabel/.store       in=\ylabel,
    zlabel/.store       in=\zlabels,
    caption/.store      in=\caption,
    name/.store         in=\\name,
    fill/.store         in=\\fill,
    bandfill/.store     in=\\bandfill,
    opacity/.store      in=\opacity,
    bandopacity/.store  in=\\bandopacity,
    fill={{rgb:red,5;green,5;blue,5;white,15}},
    bandfill={{rgb:red,5;green,5;blue,5;white,5}},
    opacity=0.4,
    bandopacity=0.6,
    width=2,
    height=13,
    depth=15,
    scale=.2,
    xlabel={{{{"","","","","","","","","",""}}}},
    ylabel=,
    zlabel=,
    caption=,
    name=,
}}


% color notation: rgb,<divisor of each value>: <color>,<quantity>;...
\def\ConvColor{{rgb,255:red,255;green,210;blue,50}}
\def\ActivationColor{{rgb,255:red,255;green,100;blue,0}}
\def\PoolColor{{rgb,255:red,200}}

\def\VecInputColor{{rgb,255:red,0;green,128;blue,0}}
\def\LinearColor{{rgb,255:red,255;green,0;blue,255}}
\def\EmbeddingColor{{rgb,255:red,0;green,0;white,128}}

\def\DropoutColor{{rgb,255:red,255}}
\def\LstmColor{{rgb:yellow,5;red,5;white,5}}
\def\\NormColor{{rgb:red,1;black,0.3}}
\def\\UnpoolColor{{rgb:blue,2;green,1;black,0.3}}
\def\FcReluColor{{rgb:blue,5;red,5;white,4}}
\def\SoftmaxColor{{rgb:magenta,5;black,7}}
\def\SumColor{{rgb:blue,5;green,15}}

\\newcommand{{\copymidarrow}}{{\\tikz \draw[-Stealth,line width=0.8mm,draw={{grey}}] (-0.3,0) -- ++(0.3,0);}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\tikzstyle{{connection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw=\EdgeColor,opacity=0.7]
\\tikzstyle{{copyconnection}}=[ultra thick,every node/.style={{sloped,allow upside down}},draw={{rgb:blue,4;red,1;green,1;black,3}},opacity=0.7]
"""

class End(TexElement):
    def __init__(self) -> None:
        pass

    @property
    def tex(self) -> str:
        return """
\end{tikzpicture}
\end{document}
"""