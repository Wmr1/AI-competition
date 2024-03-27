import torch
import torch.nn as nn
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomResNet50, self).__init__()
        original_resnet50 = models.resnet50(pretrained=pretrained)
        
        self.features = nn.Sequential(
            *list(original_resnet50.children())[:-2]  # 移除原始模型的最后一个全连接层和平均池化层
        )
        
    def forward(self, x):
        x = self.features(x)
        return x

class Neural_receiver(nn.Module):
    def __init__(self, subcarriers, timesymbols, streams, num_bits_per_symbol, d_model=512, nhead=8, num_layers=2):
        super(Neural_receiver, self).__init__()
        self.subcarriers = subcarriers
        self.timesymbols = timesymbols
        self.streams = streams
        self.num_bits_per_symbol = num_bits_per_symbol

        # 数据正则化
        self.normalization = nn.LayerNorm([self.timesymbols, self.subcarriers, 4 * self.streams])

        # 定义5层卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4 * self.streams, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 添加一个用于减少通道数的卷积层
        self.channel_reduction = nn.Conv2d(1024, 3, kernel_size=1)

        self.resnet50 = CustomResNet50()

        # Transformer配置
        transformer_layer = TransformerEncoderLayer(d_model=1024, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(40960, 1024),  # 修改这里
            nn.ReLU(),
            nn.Linear(1024, self.streams * self.num_bits_per_symbol * self.subcarriers * self.timesymbols)
        )

    def forward(self, y, template_pilot):
        batch_size = y.shape[0]
        
        z = torch.cat([y, template_pilot], dim=-1)  # 合并y和template_pilot
        
        z = self.normalization(z.permute(0, 2, 3, 1, 4).reshape(batch_size, self.timesymbols, self.subcarriers, -1))  # 数据正则化
        z = z.reshape(batch_size, self.timesymbols, self.subcarriers, -1).permute(0, 3, 1, 2)  # 调整维度
        
        z = self.conv_layers(z)  # 通过5层卷积

        z = self.channel_reduction(z)  # 通过新增加的卷积层减少通道数

        z = self.resnet50(z)  # 现在可以正确地通过CustomResNet50了
        
        # 调整z的形状以匹配Transformer的期望输入形状
        z_flat = z.reshape(batch_size, -1, 1024)  # 假设ResNet50的输出特征维度为1024
        z_trans = self.transformer_encoder(z_flat)
        
        z = z_trans.view(batch_size, -1)  # 将Transformer输出展平
        
        z = self.mlp(z)  # 通过MLP进行最终预测
        
        z = z.view(batch_size, self.streams, self.timesymbols, self.subcarriers, self.num_bits_per_symbol)  #
        return z