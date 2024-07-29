import torch
from torch.nn.parameter import Parameter
from utils.funcs import get_act
class MLP(torch.nn.Module):
    def __init__(self, c_in, c_out, middle_channels, hidden_act='relu', out_act='identity', dropout=0.):
        super(MLP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.middle_channels = middle_channels

        self.hidden_act = get_act(hidden_act)
        self.out_act = get_act(out_act)

        c_ins = [c_in] + middle_channels
        c_outs = middle_channels + [c_out]

        self.lins = torch.nn.ModuleList()
        for _, (in_dim, out_dim) in enumerate(zip(c_ins, c_outs)):
            self.lins.append(torch.nn.Linear(int(in_dim), int(out_dim)))

        self.drop = torch.nn.Dropout(dropout) if dropout > 0. else torch.nn.Identity()

    def forward(self, xs):
        if len(self.lins) > 1:
            for _, lin in enumerate(self.lins[:-1]):
                xs = lin(xs)
                xs = self.hidden_act(xs)
                xs = self.drop(xs)
            xs = self.lins[-1](xs)
            xs = self.out_act(xs)

        else:
            xs = self.drop(xs)
            xs = self.lins[-1](xs)

        return xs
    
    def get_parameters(self):
        # return set of parameters which can be optimized
        return list(self.parameters())

class MLPPredictor(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = torch.nn.Linear(in_features * 2, out_classes)

    def forward(self, z_src, z_dst):
        score = self.W(torch.cat([z_src, z_dst], 1))
        return score
    
