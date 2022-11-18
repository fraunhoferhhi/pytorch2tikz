from typing import Tuple

def hex_to_rgb(value) -> Tuple[int, int, int]:
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def hex_to_tex_color(value: str) -> str:
    r,g,b = hex_to_rgb(value)
    return f'rgb,255:red,{r};green,{g};blue,{b}'