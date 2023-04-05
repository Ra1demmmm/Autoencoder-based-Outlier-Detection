import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict

def get_model_args(dim):
    if dim < 20:
        enc_num_layers = 1
        enc_dim_list = [dim]
        enc_bias_list = [True]
        enc_act_list = ['relu']
        latent_dim = round(dim/2)
        dec_num_layers = 1
        dec_dim_list = [dim]
        dec_bias_list = [True]
        dec_act_list = ['none']

    elif dim >= 20 and dim < 100:
        enc_num_layers = 2
        enc_dim_list = [dim, round(dim/2)]
        enc_bias_list = [True, True]
        enc_act_list = ['relu', 'relu']
        latent_dim = round(dim/4)
        dec_num_layers = 2
        dec_dim_list = [round(dim/2), dim]
        dec_bias_list = [True, True]
        dec_act_list = ['relu', 'none']

    elif dim >= 100 and dim < 200:
        enc_num_layers = 3
        enc_dim_list = [dim, round(dim / 2), round(dim / 4)]
        enc_bias_list = [True, True, True]
        enc_act_list = ['relu', 'relu', 'relu']
        latent_dim = round(dim / 8)
        dec_num_layers = 3
        dec_dim_list = [round(dim / 4), round(dim / 2), dim]
        dec_bias_list = [True, True, True]
        dec_act_list = ['relu', 'relu', 'none']

    elif dim >= 200:
        enc_num_layers = 3
        enc_dim_list = [dim, round(dim / 2), round(dim / 4)]
        enc_bias_list = [True, True, True]
        enc_act_list = ['relu', 'relu', 'relu']
        latent_dim = round(dim / 16)
        dec_num_layers = 3
        dec_dim_list = [round(dim / 4), round(dim / 2), dim]
        dec_bias_list = [True, True, True]
        dec_act_list = ['relu', 'relu', 'none']

    return (enc_num_layers, enc_dim_list, enc_bias_list, enc_act_list, latent_dim, dec_num_layers, dec_dim_list, dec_bias_list, dec_act_list)


class AE(nn.Module):
    def __init__(self,
                 enc_num_layers = 2,
                 enc_dim_list = [32, 16],
                 enc_bias_list = [True, True],
                 enc_act_list = ['relu', 'relu'],
                 latent_dim = 8,
                 dec_num_layers=2,
                 dec_dim_list=[16, 32],
                 dec_bias_list=[True, True],
                 dec_act_list=['relu', 'none']):
        super(AE, self).__init__()

        self.enc_num_layers = enc_num_layers
        self.enc_dim_list = enc_dim_list.copy()
        self.enc_bias_list = enc_bias_list.copy()
        self.enc_act_list = enc_act_list.copy()
        self.latent_dim = latent_dim
        self.dec_num_layers = dec_num_layers
        self.dec_dim_list = dec_dim_list.copy()
        self.dec_bias_list = dec_bias_list.copy()
        self.dec_act_list = dec_act_list.copy()

        blocks = []
        self.enc_dim_list.append(latent_dim)
        for n in range(enc_num_layers):
            blocks.append(('enc_fc_{}'.format(n+1), nn.Linear(self.enc_dim_list[n], self.enc_dim_list[n+1], bias=self.enc_bias_list[n])))
            if self.enc_act_list[n] != 'none':
                blocks.append(('enc_act_{}'.format(n+1), self.activation(self.enc_act_list[n])))
        self.encoder = nn.Sequential(OrderedDict(blocks))

        blocks = []
        self.dec_dim_list.insert(0, latent_dim)
        for n in range(dec_num_layers):
            blocks.append(('dec_fc_{}'.format(n+1), nn.Linear(self.dec_dim_list[n], self.dec_dim_list[n + 1], bias=self.dec_bias_list[n])))
            if self.dec_act_list[n] != 'none':
                blocks.append(('dec_act_{}'.format(n+1), self.activation(self.dec_act_list[n])))
        self.decoder = nn.Sequential(OrderedDict(blocks))

    def activation(self, name='relu'):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


class PAE(nn.Module):
    def __init__(self,
                 enc_num_layers = 2,
                 enc_dim_list = [32, 16],
                 enc_bias_list = [True, True],
                 enc_act_list = ['relu', 'relu'],
                 latent_dim = 8,
                 dec_num_layers=2,
                 dec_dim_list=[16, 32],
                 dec_bias_list=[True, True],
                 dec_act_list=['relu', 'none']):
        super(PAE, self).__init__()

        self.enc_num_layers = enc_num_layers
        self.enc_dim_list = enc_dim_list.copy()
        self.enc_bias_list = enc_bias_list.copy()
        self.enc_act_list = enc_act_list.copy()
        self.latent_dim = latent_dim
        self.dec_num_layers = dec_num_layers
        self.dec_dim_list = dec_dim_list.copy()
        self.dec_bias_list = dec_bias_list.copy()
        self.dec_act_list = dec_act_list.copy()

        blocks = []
        self.enc_dim_list.append(latent_dim)
        for n in range(enc_num_layers):
            blocks.append(('enc_fc_{}'.format(n+1), nn.Linear(self.enc_dim_list[n], self.enc_dim_list[n+1], bias=self.enc_bias_list[n])))
            if self.enc_act_list[n] != 'none':
                blocks.append(('enc_act_{}'.format(n+1), self.activation(self.enc_act_list[n])))
        self.encoder = nn.Sequential(OrderedDict(blocks))

        blocks = []
        self.dec_dim_list.insert(0, latent_dim)
        self.dec_dim_list[-1] = self.dec_dim_list[-1] * 2
        for n in range(dec_num_layers):
            blocks.append(('dec_fc_{}'.format(n+1), nn.Linear(self.dec_dim_list[n], self.dec_dim_list[n + 1], bias=self.dec_bias_list[n])))
            if self.dec_act_list[n] != 'none':
                blocks.append(('dec_act_{}'.format(n+1), self.activation(self.dec_act_list[n])))
        self.decoder = nn.Sequential(OrderedDict(blocks))

    def activation(self, name='relu'):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


