import sys
sys.path.append('../fairseq-l-system-captioning/')
from hhi_pl_utils import Experiment

from pytorch2tikz.arch import Architecure

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


    print('build arch')
    arch = Architecure(model)

    print('run model')
    model.eval()
    model.validation_step(batch, 0)

    print(arch)

    tex = arch.get_tex()

    print('write to test_out.tex')
    with open('test_out.tex', 'w') as f:
        f.write(tex)