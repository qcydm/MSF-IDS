from torch import sigmoid
import torch

def decode(src,dst,z):
    return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )
def bce(t_scores, f_scores):
        '''
        Computes binary cross entropy loss

        t_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that exist
        f_scores : torch.Tensor
            a 1-dimensional tensor of likelihood scores given to edges that do not exist
        '''
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5