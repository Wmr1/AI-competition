# =======================================================================================================================
# =======================================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channel_list, H, W, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.channel_list = channel_list
        self._conv_1 = nn.Conv2d(self.channel_list[2], self.channel_list[0], kernel_size=3, padding='same')
        self._layer_norm_1 = nn.LayerNorm([self.channel_list[0], H, W])
        self._conv_2 = nn.Conv2d(self.channel_list[0], self.channel_list[1], kernel_size=3, padding='same')
        self._layer_norm_2 = nn.LayerNorm([self.channel_list[1], H, W])
        self._relu = nn.ReLU()

    def forward(self, inputs):
        x_ini = inputs
        x = self._layer_norm_1(x_ini)
        x = self._relu(x)
        x = self._conv_1(x)
        x = self._layer_norm_2(x)
        x = self._relu(x)
        x = self._conv_2(x)
        x_ini = x_ini + x
        return x_ini

NUM_SUBCARRIERS = 624
NUM_OFDM_SYMBOLS = 12
NUM_LAYERS = 2
NUM_BITS_PER_SYMBOL = 4
class Neural_receiver(nn.Module):
    def __init__(self, subcarriers, timesymbols, streams, num_bits_per_symbol, num_blocks=6, channel_list=[24, 24, 24],
                 **kwargs):
        super(Neural_receiver, self).__init__(**kwargs)
        self.subcarriers = subcarriers
        self.timesymbols = timesymbols
        self.streams = streams
        self.num_blocks = num_blocks
        self.channel_list = channel_list
        self.num_bits_per_symbol = num_bits_per_symbol
        self.NUM_SUBCARRIERS = 624
        self.NUM_OFDM_SYMBOLS = 12
        self.NUM_LAYERS = 2
        self.NUM_BITS_PER_SYMBOL = 4

        self.blocks = nn.Sequential()
        for block_id in range(self.num_blocks):
            block = ResBlock(channel_list=self.channel_list, H=self.timesymbols, W=self.subcarriers)
            self.blocks.add_module(name='block_{}'.format(block_id), module=block)
        self._conv_1 = nn.Conv2d(4 * self.streams, self.channel_list[2], kernel_size=3, padding='same')
        self._conv_2 = nn.Conv2d(self.channel_list[1], self.streams * self.num_bits_per_symbol, kernel_size=3,
                                 padding='same')

    def forward(self, y, template_pilot):
        # y : [batch size,NUM_LAYERS,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,2]
        # template_pilot : [batch size,NUM_LAYERS,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,2]
        batch_size = y.shape[0]
        y = y.permute(0, 2, 3, 1, 4)  # y :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS,2]
        y = torch.reshape(y, (batch_size, self.timesymbols, self.subcarriers, self.streams * 2))
        # y :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS*2]
        template_pilot = template_pilot.permute(0, 2, 3, 1, 4)
        # p :  [batch size,NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_LAYERS,2]
        template_pilot = torch.reshape(template_pilot,
                                       (batch_size, self.timesymbols, self.subcarriers, self.streams * 2))

        z = torch.cat([y, template_pilot], dim=-1)
        # 按最后一个维度切成8份，并确保每个切片形状正确
        z_splits = [z_split.contiguous().view(-1, 1) for z_split in z.split(1, dim=-1)]  # 列表，包含8个形状为 [batch_size * NUM_OFDM_SYMBOLS * NUM_SUBCARRIERS, 1] 的张量
        
        # 定义 MLP，假设隐藏层大小为256，输出层为NUM_BITS_PER_SYMBOL，输入大小现在是1
        mlps = [MLP(1, 256, self.num_bits_per_symbol).to(z.device) for _ in range(8)]
        
        # 对每份应用 MLP
        z_processed = [mlp(z_split) for z_split, mlp in zip(z_splits, mlps)]
        
        # 在最后一个维度上拼接处理过的张量，并重塑形状
        z_concat = torch.cat(z_processed, dim=1).view(batch_size, self.NUM_OFDM_SYMBOLS, self.NUM_SUBCARRIERS, -1)
        # import pdb;
        # pdb.set_trace()
        mlp = MLP(input_size=32, hidden_size=64,output_size=8).to(z_concat.device)
        # Reshape z_concat for MLP processing
        # Flatten the dimensions except for the last one
        z_flattened = z_concat.view(-1, 32)

        # Process through the MLP
        z_mlp_processed = mlp(z_flattened)
        
        # 重塑张量以匹配期望的输出形状
        z_reshaped = z_mlp_processed.view(batch_size, self.NUM_LAYERS, self.NUM_OFDM_SYMBOLS, self.NUM_SUBCARRIERS, self.NUM_BITS_PER_SYMBOL)
        # NUM_LAYERS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS,NUM_BITS_PER_SYMBOL
        # Assuming z_reshaped has the shape [batch_size, NUM_LAYERS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, NUM_BITS_PER_SYMBOL]

        z_softmax = F.softmax(z_reshaped, dim=-1)
        return z_softmax

