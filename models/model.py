import os.path
import torch
import torch.nn as nn
from models.featurenet import FeatureExtractorNet
from models.correlationnet import CorrelationNet
from models.utils import correlate, warp, linear_position_embedding_sine, de_conv_layer


class FlowModel(nn.Module):
    def __init__(self, config, device, train=True):
        super(FlowModel, self).__init__()
        self.config = config
        self.device = device
        self.to(self.device)
        self.feature_net = FeatureExtractorNet().to(self.device)
        if train:
            self.load_encoder()
        self.patch_size = [4, 8, 8, 16, 16, 32]
        self.dilation_patch = [2, 4, 4, 8, 8, 16]

    def load_encoder(self):
        if os.path.exists(self.config['fe_save_path']):
            fe_net_weights = torch.load(self.config['fe_save_path'])
            self.feature_net.load_state_dict(fe_net_weights["state_dict"])
            for params in self.feature_net.parameters():
                params.requires_grad = False
            print(f"AE Model loaded from {self.config['fe_save_path']}")
        else:
            print(f"AE Model is not in the path {self.config['fe_save_path']}")

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr_volume = torch.matmul(fmap1.transpose(1, 2), fmap2)
        # corr = corr.view(batch, ht, wd, 1, ht, wd)
        # corr_volume = corr_volume.view(batch, ht, wd, ht, wd)
        # return corr_volume / torch.sqrt(torch.tensor(dim).float())
        min_val = torch.min(corr_volume)
        max_val = torch.max(corr_volume)
        corr_volume = (corr_volume - min_val) / (max_val - min_val)  # normalize the correlation
        corr_volume = 2 * corr_volume - 1  # set in the range of -1 and 1

        return corr_volume

    def extract_feature(self, x1, x2):
        x1_fe_out = [x1]
        x2_fe_out = [x2]

        for layers in self.feature_net.encoder:
            x1 = layers(x1)
            x2 = layers(x2)
            x1_fe_out.append(x1)
            x2_fe_out.append(x2)

        return x1_fe_out, x2_fe_out


class FlowCorrelationCNN(FlowModel):
    def __init__(self, config, device, train=True):
        super(FlowCorrelationCNN, self).__init__(config, device, train)
        self.correlation_net = CorrelationNet().to(self.device)

    def forward(self, x1, x2):
        predict_flow = None
        x1_fe_out, x2_fe_out = self.extract_feature(x1, x2)

        i = len(x1_fe_out) - 1
        j = 0
        wrapped_x2_out = x2_fe_out[i]

        for layers in self.correlation_net.correlation:
            b, c, h, w = wrapped_x2_out.size()
            flow = torch.zeros((b, 2, h, w))
            if wrapped_x2_out.is_cuda:
                flow.contiguous().cuda(non_blocking=True)
            corr_val = self.corr(x1_fe_out[i], wrapped_x2_out)
            x1_x2_correlate = correlate(x1_fe_out[i], wrapped_x2_out,
                                        patch_size=self.patch_size[j])  # [1, 16, 4, 128]

            corr_h_mean = torch.mean(x1_x2_correlate, dim=1)
            corr_w_mean = torch.mean(x1_x2_correlate, dim=2)

            u_disp = torch.argmax(corr_h_mean, dim=1)
            v_disp = torch.argmax(corr_w_mean, dim=1)
            flow[:, 0, :, :] = u_disp
            flow[:, 1, :, :] = v_disp

            # flow_img = flow_to_color(flow.detach().squeeze().numpy().transpose(1, 2, 0))

            # x1_corr_concat = torch.cat([x1_fe_out[i], x1_x2_correlate], dim=1)  # [1, 272, 4, 128]

            predict_flow = layers(flow)
            # flow_img_4 = flow_to_color(self.predict_flow.detach().squeeze().numpy().transpose(1, 2, 0))
            if i != 0:
                wrapped_x2_out = warp(x2_fe_out[i - 1], predict_flow)
                i = i - 1
            j = j + 1
        return predict_flow


class FlowTransformer(FlowModel):
    def __init__(self, config, device, train=True):
        super().__init__(config, device, train)
        self.embedded_dim = 512
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=1024,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=self.embedded_dim,
                      kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
        )
        self.linear_map = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64)
        )

        self.transformer = nn.Transformer(
            d_model=64,
            nhead=8,
            num_encoder_layers=5,
            num_decoder_layers=5,
            dropout=0.1,
        )
        self.out = nn.Linear(64, 2)

        self.upsample_flow = de_conv_layer(f"flow_up_1_2", 2, 2, max_pooled=True)

    def forward(self, x1, x2):
        predict_flow = None
        x1_fe_out, x2_fe_out = self.extract_feature(x1, x2)  # extract features from x1 and x2

        index = [5, 3, 1, 0]

        wrapped_x2_out = x2_fe_out[5]

        for idx, i in enumerate(index):
            b, dim, h, w = x1_fe_out[i].size()
            corr_val = self.corr(x1_fe_out[i], wrapped_x2_out)  # correlation operation between the lowest dimensions
            b, h1, w1, = corr_val.size()
            corr_val = corr_val.view(b, h, w, w1).permute(0, 3, 1, 2)

            corr_embedded = self.embedding_layer(corr_val).flatten(2)  # source correlation embedding

            corr_embedded = linear_position_embedding_sine(corr_embedded)  # source embedding with positional encoding

            corr_source = self.linear_map(corr_embedded.permute(0, 2, 1))  # correlation
            corr_target = torch.roll(corr_source, shifts=1, dims=-1)

            output = self.transformer(corr_source, corr_target)
            predict_flow = self.out(output)
            predict_flow = predict_flow.permute(0, 2, 1).view(b, 2, h1, w1)

            if i != 0:
                up_flow = self.upsample_flow(predict_flow)
                wrapped_x2_out = warp(x2_fe_out[index[idx+1]], up_flow)

        return predict_flow
