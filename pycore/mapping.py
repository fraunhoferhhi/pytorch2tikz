from .block.D1 import LinearBlock, LSTMBlock, EmbeddingBlock
from .block.Dn import ConvBlock, NormBlock, PoolBlock, ActivationBlock, DropoutBlock

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