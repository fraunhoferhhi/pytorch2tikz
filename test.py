import sys
sys.path.append('../fairseq-l-system-captioning/')
from hhi_pl_utils import Experiment

from pycore.arch import Architecure
from pycore.module_graph import create_graph

if __name__ == '__main__':
    print('Load model')
    e = Experiment()
    e.setup('../fairseq-l-system-captioning/configs/simplest/slim_lstm_normal.yml')
    e.init_classes()
    e.data.setup()

    e.model.vocab = e.data.vocab

    model = e.model
    handles = []

    print('Load data')
    it = iter(e.data.train_dataloader())
    batch = next(it)


    graph = create_graph(model)
    arch = Architecure()

    if len(handles) > 0:
        for h in handles:
            h.remove()
        handles = []

    modules = []
    for c in graph.bfs():
        if 'torch' in str(type(c.module)):
            handles.append(c.module.register_forward_hook(arch))
            modules.append(c.module)

    print('build arch')
    model.eval()
    model.validation_step(batch, 0)

    tex = arch.finalize()

    print('write to test_out.tex')
    with open('test_out.tex', 'w') as f:
        f.write(tex)