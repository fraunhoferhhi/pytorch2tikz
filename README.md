# Pytorch2Tikz
[//]: # "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2526396.svg)](https://doi.org/10.5281/zenodo.2526396)"

Generate Tikz figures for neural networks implemented in pytorch. It uses LaTeX snippets from [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) but you can now just run your network to plot everything automatically. For examples see `./examples`.

## Example

```python
from pytorch2tikz import Architecure

print('Load model')
model = vgg16(True)

print('Load data')
...

print('Init architecture')
arch = Architecure(model)

print('Run model')
with torch.inference_mode():
    for image, _ in data_loader:
        image = image.to(device, non_blocking=True)
        output = model(image)

print('Write result to out.tex')
arch.save('out.tex')
```

## Getting Started
```
pip install pytorch2tikz
```

## Detailed overview of the class Architecture

```python
Architecure(module: nn.Module,
            block_offset=8,
            height_depth_factor=0.8,
            width_factor=0.8,
            linear_factor=0.8,
            image_path='./input_{i}.png',
            ignore_layers=['batchnorm', 'flatten'],
            colors=COLOR_VALUES)
```

### Arguments
- `module` is the Model to plot
- `block_offset` offset to the next block; A block is created when the input dimensions change
- `height_depth_factor` scale the change of the next layer (last 2 dimensions); typically used to make the network a bit more compact
- `width_factor` scale the change of the next layer (first dimension); typically used to make the network a bit more compact
- `linear_factor` used when there is a drastic change in the last dimension (e.g. moving from conv to linear layers)
- `image_path` output path for recognized input images. `{i}` gets replaced by the current layer index
- `ignore_layers` define layers that should not be plotted. This can be a list of any substring of the `type(class)` (e.g. torch.nn.modules.batchnorm.BatchNorm)
- `colors` enum of colors. For an example check out `./pytorch2tikz/constants`

### Important Methods
```python
def get_block(self, name: str) -> Block:
    ...
```

get a specific block to alter its properties

```python
def get_tex(self) -> str:
    ...
```

generate the tex code

```python
    
def save(self, file_path: str):
    ...
```

generate and save the tex code to the given path

## Detailed overview of the class Block

```python
Block(name,
    fill: COLOR = COLOR.LINEAR,
    bandfill: COLOR = None,
    pictype = PICTYPE.BOX,
    opacity = 0.7,
    size = (10,40,40),
    default_size = DEFAULT_VALUE * DIM_FACTOR,
    dim = 3,
    scale_factor = np.zeros(3),
    offset = (0,0,0),
    to = (0,0,0),
    caption = " ",
    xlabel = True,
    ylabel = False,
    zlabel = True,
    is_input = False)
```