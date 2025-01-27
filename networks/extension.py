import torch
import torch.nn as nn

class DinaMixerOp(nn.Module):

    def __init__(self, dim, seq_len, num_head, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x