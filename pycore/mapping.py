from .blocks_1D import LinearBlock, LSTMBlock, EmbeddingBlock
from .blocks_2D import NormBlock, PoolBlock, ActivationBlock, DropoutBlock
from .blocks_3D import ConvBlock

BLOCK_MAPPING = {
    'torch.nn.modules.conv': ConvBlock,
    'torch.nn.modules.batchnorm': NormBlock,
    'torch.nn.modules.pooling': PoolBlock,
    'torch.nn.modules.linear': LinearBlock,
    'torch.nn.modules.rnn': LSTMBlock,
    'torch.nn.modules.sparse.Embedding': EmbeddingBlock,
    'torch.nn.modules.dropout': DropoutBlock,
    'torch.nn.modules.activation': ActivationBlock
}