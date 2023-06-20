import torch
import torch.nn as nn
import copy
from models.utils import linear_position_embedding_sine, PositionEmbeddingSine


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([input_dim])))

        self.key_layer = nn.Linear(input_dim, input_dim)
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        # Softmax function for attention weights
        self.softmax = nn.Softmax(dim=-1)

        # Initialize: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L138
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # original Transformer initialization


class SelfAttention(Attention):
    def __init__(self, input_dim, in_channel=4096):
        super(SelfAttention, self).__init__(input_dim)

        self.self_linear_embedding = nn.Sequential(
            nn.Linear(in_channel, input_dim),
            nn.LayerNorm(input_dim)
        )
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=input_dim//2)

    def forward(self, corr):
        b, h, w = corr.size()
        corr_embedded = self.self_linear_embedding(corr)  # source correlation embedding
        corr_embedded_pos = self.position_embedding(corr_embedded.permute(0, 2, 1).view(b, 32, 32, 128))
        corr_embedded = corr_embedded + corr_embedded_pos.view(b, 32, 4096).permute(0, 2, 1)
        # corr_embedded = linear_position_embedding_sine(corr_embedded)  # source embedding with positional encoding

        query = self.query_layer(corr_embedded)
        key = self.key_layer(corr_embedded)
        value = self.value_layer(corr_embedded)

        # calculate similarity between query and key using scaled dot product
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # softmax normalization
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to value vectors (convex combination)
        attention_values = torch.matmul(attention_weights, value)

        # Return the attention values
        return attention_values


class CrossAttention(Attention):
    def __init__(self, input_dim, in_channel=32):
        super(CrossAttention, self).__init__(input_dim)
        self.cross_linear_embedding = nn.Sequential(
            nn.Linear(in_channel, input_dim),
            nn.LayerNorm(input_dim)
        )
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=input_dim//2)

    def forward(self, feature1, feature2):
        source = self.cross_linear_embedding(feature1.flatten(2).permute(0, 2, 1))
        target = self.cross_linear_embedding(feature2.flatten(2).permute(0, 2, 1))
        source_pos = self.position_embedding(feature1)
        target_pos = self.position_embedding(feature2)

        source = source + source_pos.flatten(2).permute(0, 2, 1)
        target = target + target_pos.flatten(2).permute(0, 2, 1)

        query = self.query_layer(source)
        key = self.key_layer(target)
        value = self.value_layer(target)

        # calculate similarity between query and key using scaled dot product
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # softmax normalization
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to value vectors (convex combination)
        attention_values = torch.matmul(attention_weights, value)

        # Return the attention values
        return attention_values


class Attention1D(nn.Module):
    """Cross-Attention on x or y direction,
    without multi-head and dropout support for faster speed
    """

    def __init__(self, in_channels,
                 y_attention=False,
                 double_cross_attn=False,  # cross attn feature1 before computing cross attn feature2
                 **kwargs,
                 ):
        super(Attention1D, self).__init__()

        self.y_attention = y_attention
        self.double_cross_attn = double_cross_attn

        # self attn feature1 before cross attn
        if double_cross_attn:
            self.self_attn = copy.deepcopy(Attention1D(in_channels=in_channels,
                                                       y_attention=not y_attention,
                                                       )
                                           )

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Initialize: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py#L138
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # original Transformer initialization

    def forward(self, feature1, feature2, position=None, value=None):
        b, c, h, w = feature1.size()

        # self attn before cross attn
        if self.double_cross_attn:
            feature1 = self.self_attn(feature1, feature1, position)[0]  # self attn feature1

        query = feature1 + position if position is not None else feature1
        query = self.query_conv(query)  # [B, C, H, W]

        key = feature2 + position if position is not None else feature2

        key = self.key_conv(key)  # [B, C, H, W]
        value = feature2 if value is None else value  # [B, C, H, W]
        scale_factor = c ** 0.5

        if self.y_attention:
            query = query.permute(0, 3, 2, 1)  # [B, W, H, C]
            key = key.permute(0, 3, 1, 2)  # [B, W, C, H]
            value = value.permute(0, 3, 2, 1)  # [B, W, H, C]
        else:  # x attention
            query = query.permute(0, 2, 3, 1)  # [B, H, W, C]
            key = key.permute(0, 2, 1, 3)  # [B, H, C, W]
            value = value.permute(0, 2, 3, 1)  # [B, H, W, C]

        scores = torch.matmul(query, key) / scale_factor  # [B, W, H, H] or [B, H, W, W]

        attention = torch.softmax(scores, dim=-1)  # [B, W, H, H] or [B, H, W, W]

        out = torch.matmul(attention, value)  # [B, W, H, C] or [B, H, W, C]

        if self.y_attention:
            out = out.permute(0, 3, 2, 1).contiguous()  # [B, C, H, W]
        else:
            out = out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return out, attention