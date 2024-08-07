from typing import Tuple

from torch_geometric.nn import global_add_pool

from utils.Implicit_Module import Implicit_Module
from utils.Append_func import Append_func
from utils.funcs import get_act
from utils.MLP import MLP
import torch
import torch.nn.functional as F


class MGD(torch.nn.Module):

    def __init__(self,
                 in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, alpha: float, iter_nums: Tuple[int, int],
                 dropout_imp: float = 0., dropout_exp: float = 0.,
                 drop_input: bool = False, norm: str = 'LayerNorm',
                 residual: bool = True, rescale: bool = True,
                 linear: bool = True, double_linear: bool = True,
                 act_imp: str = 'tanh', act_exp: str = 'elu',
                 reg_type: str = '', reg_coeff: float = 0.,
                 final_reduce: str = ''):
        super().__init__()

        self.total_num, self.grad_num = iter_nums
        self.no_grad_num = self.total_num - self.grad_num

        self.reg_type = reg_type
        self.dropout_exp = dropout_exp
        self.act = get_act(act_exp)
        self.residual = residual
        self.rescale = rescale

        self.drop_input = drop_input

        self.extractor = torch.nn.Linear(in_channels, hidden_channels)

        middle_channels = [hidden_channels] * num_layers
        self.implicit_module = Implicit_Module(hidden_channels, middle_channels,
                                               alpha, norm, dropout_imp, act_imp,
                                               double_linear, rescale)
        self.Append = Append_func(coeff=reg_coeff, reg_type=reg_type)
        if linear:
            mlp_params = {'c_in': hidden_channels,
                          'c_out': out_channels,
                          'middle_channels': [],
                          'hidden_act': act_exp,
                          'dropout': dropout_exp}
        else:
            mlp_params = {'c_in': hidden_channels,
                          'c_out': out_channels,
                          'middle_channels': [hidden_channels],
                          'hidden_act': act_exp,
                          'dropout': dropout_exp}
        self.last_layer = MLP(**mlp_params)
        self.reduce = final_reduce

        self.init_weights()

        self.params_imp = list(self.implicit_module.parameters())
        # self.params_exp = list(self.extractor.parameters()) + list(self.last_layer.parameters()) + list(
        #     self.Append.parameters())
        self.params_exp = list(self.extractor.parameters()) + list(self.last_layer.parameters()) + list(
            self.Append.parameters())

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        torch.nn.init.xavier_normal_(self.last_layer.lins[-1].weight, gain=1.)

    def multiple_steps(self, iter_start, iter_num, z, x, edge_index, norm_factor, batch):
        for _ in range(iter_start, iter_start + iter_num):
            z = self.Append(z, x, edge_index=edge_index, norm_factor=norm_factor)
            z = self.implicit_module(z, x, edge_index, norm_factor, batch)
        return z

    def forward(self, x, edge_index, norm_factor, batch=None):
        if self.drop_input:
            x = F.dropout(x, self.dropout_exp, training=self.training)

        x = self.extractor(x)

        self.implicit_module._reset(x)

        z = torch.zeros_like(x)
        with torch.no_grad():
            z = self.multiple_steps(0, self.no_grad_num, z, x, edge_index, norm_factor, batch)
        new_z = self.multiple_steps(self.no_grad_num - 1, self.grad_num, z, x, edge_index, norm_factor, batch)

        if self.rescale:
            z = norm_factor * new_z + x if self.residual else new_z
        else:
            z = new_z + x if self.residual else new_z

        if self.reduce == 'add':
            z = global_add_pool(z, batch)
        return z
