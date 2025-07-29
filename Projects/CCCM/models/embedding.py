import math
import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding_old(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0, device: torch.device = None):
        super().__init__()
        self.size = size
        self.scale = scale
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = emb.to(self.device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    

class ConditionalEmbedding(nn.Module):
    
    def __init__(self, num_label: int, d_model: int, emb_dim: int, dropout_prob: float = 0.1):
        assert d_model % 2 == 0
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding_table = nn.Embedding(num_embeddings=num_label + 1, embedding_dim=d_model)
        self.emb = nn.Sequential(
            nn.Linear(d_model, emb_dim),
            Swish(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.dropout_prob = dropout_prob
        self.num_classes = num_label
        
    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.tensor(force_drop_ids == 1, device=labels.device)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels
        
    def forward(self, labels: torch.LongTensor, force_drop_ids=None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        embeddings = self.emb(embeddings)
        return embeddings
    
    def __len__(self):
        return self.emb_dim
    
    
class LabelEmbedding(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.LongTensor, force_drop_ids=None, evaluation=False):
        # Disable label drop when sampling
        use_dropout = (self.dropout_prob > 0) and not evaluation
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def __len__(self):
        return self.hidden_size
    
class TextImageProjection(nn.Module):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: torch.FloatTensor, image_embeds: torch.FloatTensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # text
        text_embeds = self.text_proj(text_embeds)

        return torch.cat([image_text_embeds, text_embeds], dim=1)
    
    
class TimeEmbedding(nn.Module):
    def __init__(self, 
                 T : int = 1000,
                 in_channels : int = 128,
                 time_embed_dim : int = 512, 
                 cond_proj_dim : int = None
                ):
        
        super().__init__()   
        
        assert in_channels % 2 == 0
        
        self.cond_proj = None
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        
        _emb = torch.arange(0, in_channels, step=2) / in_channels * math.log(10000)
        _emb = torch.exp(-_emb)
        pos = torch.arange(T).float()
        _emb = pos[:, None] * _emb[None, :]
        #assert list(_emb.shape) == [T, d_model // 2]
        _emb = torch.stack([torch.sin(_emb), torch.cos(_emb)], dim=-1)
        #assert list(_emb.shape) == [T, d_model // 2, 2]
        _emb = _emb.view(T, in_channels) 
        
        self.emb_layer = nn.Embedding.from_pretrained(_emb, freeze=False)
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = Swish()
#         if cond is not None:
#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb, freeze=False),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )

    def forward(self, timesteps:torch.Tensor, cond:torch.Tensor=None):
        emb = self.emb_layer(timesteps)
        if cond is not None:
            emb = emb + self.cond_proj(cond)
        emb = self.linear_1(emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        
        return emb

#     def forward(self, timesteps:torch.Tensor):
#         emb = self.emb_layer(timesteps)
#         if cond is not None:
#             emb = emb + self.cond_proj(cond)
#         emb = self.linear_1(emb)
#         emb = self.act(emb)
#         emb = self.linear_2(emb)
        
        return emb
 