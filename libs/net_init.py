from torch.nn import init
from torch import nn
import math


def initNetParams(net, method='uniform'):
    '''Init net parameters.'''

    for m in net.modules():
        if isinstance(m, nn.Linear):
            if method == 'uniform':
                init.uniform_(m.weight)
            elif method == 'norm':
                init.normal_(m.weight, std=1e-3)
            elif method == 'xavier':
                init.xavier_uniform_(m.weight)
            elif method == 'he':
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))

            if m.bias is not None:
                init.constant_(m.bias, 0)

