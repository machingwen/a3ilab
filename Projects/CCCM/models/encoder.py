import torch
import torch.nn as nn
from .resnet import ResNet50
from torchvision.models import resnet18, resnet34, resnet50


class ClassEncoder(nn.Module):
    def __init__(self, num_atr, num_obj, d_model=128, emb_dim=512):
        super().__init__()
        
        self.atr_emb = nn.Sequential(
            nn.Embedding(num_embeddings=num_atr, embedding_dim=d_model),
            nn.Linear(d_model, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.obj_emb = nn.Sequential(
            nn.Embedding(num_embeddings=num_obj, embedding_dim=d_model),
            nn.Linear(d_model, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, atr, obj):
        atr_embedding = self.atr_emb(atr)
        obj_embedding = self.obj_emb(obj)
        
        return torch.cat([atr_embedding, obj_embedding], dim=-1)
    

def ImageEncoder(pretrained=False):
    encoder = resnet50()
    encoder.fc = nn.Identity()
    return encoder
    

    
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x