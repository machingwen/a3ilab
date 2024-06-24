import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from .encoder import ClassEncoder, ImageEncoder, ProjectionHead


class CCIPModel(nn.Module):
    def __init__(
        self,
        num_atr,
        num_obj,
        init_logit_scale=np.log(1 / 0.07),
        image_embedding=2048,
        class_embedding=512,
        projection_dim=256,
        origin=False
    ):
        super().__init__()
        
        self.num_atr = num_atr
        self.num_obj = num_obj
        self.class_emb_dim = class_embedding
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.origin = origin # paper implementation of loss function
        self.image_encoder = ImageEncoder()
        self.class_encoder = ClassEncoder(num_atr + 1, num_obj + 1, emb_dim=class_embedding)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.class_projection = ProjectionHead(embedding_dim=class_embedding * 2, projection_dim=projection_dim)
    
    def encode_image(self, image, normalized=False):
        image_features = self.image_encoder(image)
        image_embeddings = self.image_projection(image_features)
        return F.normalize(image_embeddings, dim=-1) if normalized else image_embeddings
    
    def encode_class(self, atr, obj, normalized=False):
        class_features = self.class_encoder(atr=atr, obj=obj)
        class_embeddings = self.class_projection(class_features)
        return F.normalize(class_embeddings, dim=-1) if normalized else class_embeddings

    def forward(self, image, atr, obj):
        
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.encode_image(image)
        class_embeddings = self.encode_class(atr=atr, obj=obj)
        
        # Calculating the Loss
        logits_per_class = self.logit_scale.exp() * class_embeddings @ image_embeddings.T
        logits_per_image = self.logit_scale.exp() * image_embeddings @ class_embeddings.T

        if self.origin:
            targets = torch.arange(image.shape[0], device=image.device)
            texts_loss = F.cross_entropy(logits_per_class, targets, reduction='none')
            images_loss = F.cross_entropy(logits_per_image, targets, reduction='none')

        else:
            images_similarity = self.logit_scale.exp() * image_embeddings @ image_embeddings.T
            texts_similarity = self.logit_scale.exp() * class_embeddings @ class_embeddings.T
            targets = F.softmax(
                (images_similarity + texts_similarity) / 2, dim=-1
            )
            texts_loss = cross_entropy(logits_per_class, targets, reduction='none')
            images_loss = cross_entropy(logits_per_image, targets.T, reduction='none')

        
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
