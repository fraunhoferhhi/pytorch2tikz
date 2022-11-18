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

    print('Build arch')
    arch = Architecure(model, linear_factor=0)

    print('Run model')
    model.eval()
    model.validation_step(batch, 0)

    print('Final architecture', arch)

    print('Write result to test_out.tex')
    arch.save('test_out.tex')