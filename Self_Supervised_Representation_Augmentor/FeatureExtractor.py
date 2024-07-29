import torch.nn as nn
class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(FeatureExtractor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)