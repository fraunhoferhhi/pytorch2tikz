from typing import List, Union
from torch import nn

class ModuleNode:

    def __init__(self, module: nn.Module, parent=None) -> None:
        self.parent = parent
        self.children: List[ModuleNode] = []
        self.module = module
    
    def bfs(self):
        queue = [self]
        while len(queue) > 0:
            queue.extend(queue[0].children)
            yield queue.pop(0)
    
    def __repr__(self) -> str:
        return f'Module: {str(type(self.module))}'

def create_graph(model: nn.Module,
                 ignore: List[nn.Module]=[]
    ) -> Union[ModuleNode, None]:

    t = str(type(model))
    if ('loss' in t)\
        or ('vocab' in t)\
        or (type(model) in ignore):
        return None

    root = ModuleNode(model)
    for child in model.children():
        if 'torch.nn.modules.container.Sequential' in str(type(child)):
            for child_seq in child.children():
                child_mod = create_graph(child_seq)
                if child_mod is not None:
                    root.children.append(child_mod)
        else:
            child_mod = create_graph(child)
            if child_mod is not None:
                root.children.append(child_mod)

    return root