import torch.nn as nn
from models.utils import predict_flow_layer, de_conv_layer
from models.attention import SelfAttention, CrossAttention
from models.update import BasicUpdateBlock


class CorrelationNet(nn.Module):
    def __init__(self, in_channel=256):
        super(CorrelationNet, self).__init__()
        self.correlation = nn.ModuleList()
        self._make_flow_prediction_layer(272, name="5_4")
        self._make_flow_prediction_layer(192, name="4_3", max_pooled=False)
        self._make_flow_prediction_layer(128, name="3_2")
        self._make_flow_prediction_layer(288, name="2_1", max_pooled=False)
        self._make_flow_prediction_layer(272, name="1_0")
        # self.correlation.append(predict_flow_layer(1025))
        # self.correlation.append(predict_flow_layer(2))
        self.softmax = nn.Softmax(dim=1)

    def _make_flow_prediction_layer(self, in_channel, name, max_pooled=True):
        pf_layer = predict_flow_layer(in_channel)
        upsample_pf_layer = de_conv_layer(f"flow_up_{name}", 2, 2, max_pooled=max_pooled)
        pf_seq = nn.Sequential(pf_layer, *upsample_pf_layer)
        self.correlation.append(pf_seq)


class SleceNet(nn.Module):
    def __init__(self, embedded_dim, downsample_factor):
        super(SleceNet, self).__init__()
        self.embedded_dim = embedded_dim
        self.downsample_factor = downsample_factor
        self.layers = nn.ModuleList()

        self.self_attention_layer = SelfAttention(input_dim=self.embedded_dim)
        self.cross_attention_layer = CrossAttention(input_dim=self.embedded_dim)

        # Update block
        self.update_block = BasicUpdateBlock(corr_channels=self.embedded_dim,
                                             hidden_dim=self.embedded_dim,
                                             context_dim=self.embedded_dim,
                                             downsample_factor=self.downsample_factor)

