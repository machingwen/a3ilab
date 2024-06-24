import torch
import torch.nn as nn
from .embedding import TimeEmbedding, ConditionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, size),
            nn.GELU()
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, hidden_layers: int, input_size: int, emb_size: int, hidden_size: int) -> None:
        super().__init__()
#         self.time_embedding = TimeEmbedding()
#         self.obj_embedding = ConditionalEmbedding()
#         selg.atr_embedding = ConditionalEmbedding()
        concat_size = input_size + 3 * emb_size
        
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU(),]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, input_size))
        
        self.joint_mlp = nn.Sequential(*layers)
        
    def forward(self, x, c1, c2, t) -> torch.Tensor:
        x = torch.cat((x, c1, c2, t), dim=-1)
        x = self.joint_mlp(x)
        return x
        