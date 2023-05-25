import torch.nn as nn
from models.local_net_ae import FeatureExtractor
from models.correlation_net import CorrelationNetwork


class FlowModel(nn.Module):
    def __init__(self, device):
        super(FlowModel, self).__init__()
        self.device = device
        self.to(self.device)
        self.feature_net = FeatureExtractor().to(self.device)
        self.correlation_net = CorrelationNetwork().to(self.device)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x1, x2):
        feature1 = self.feature_net(x1, self.device)
        feature2 = self.feature_net(x2, self.device)
        pred_flow = self.correlation_net(feature1, feature2)
        return pred_flow
