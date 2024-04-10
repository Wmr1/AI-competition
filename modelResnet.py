import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channel_in, H, W):
        super(ResBlock, self).__init__()
        self._conv_1 = nn.Conv2d(channel_in, channel_in, kernel_size=4, padding='same')
        self._layer_norm_1 = nn.LayerNorm([channel_in, H, W])
        self._conv_2 = nn.Conv2d(channel_in, channel_in, kernel_size=4, padding='same')
        self._layer_norm_2 = nn.LayerNorm([channel_in, H, W])
        self._relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self._layer_norm_1(x)
        out = self._relu(out)
        out = self._conv_1(out)
        out = self._layer_norm_2(out)
        out = self._relu(out)
        out = self._conv_2(out)
        out += residual
        return out

class Neural_receiver(nn.Module):
    def __init__(self, subcarriers, timesymbols, streams, num_bits_per_symbol, num_blocks=10, channel_list=[24, 24, 24]):
        super(Neural_receiver, self).__init__()
        self.subcarriers = subcarriers
        self.timesymbols = timesymbols
        self.streams = streams
        self.num_blocks = num_blocks
        self.channel_list = channel_list
        self.num_bits_per_symbol = num_bits_per_symbol

        self.template_pilot_conv = nn.Conv2d(4, 24, kernel_size=4, padding='same')

        self.blocks = nn.Sequential()
        for block_id in range(self.num_blocks):
            block = ResBlock(28, timesymbols, subcarriers)  # Adjusted for actual channel count
            self.blocks.add_module(f'block_{block_id}', block)

        self.final_conv = nn.Conv2d(28, streams * num_bits_per_symbol, kernel_size=4, padding='same')

    def forward(self, y, template_pilot):
        batch_size = y.shape[0]
        y = y.permute(0, 2, 3, 1, 4).reshape(batch_size, self.timesymbols, self.subcarriers, -1)
        template_pilot = template_pilot.permute(0, 2, 3, 1, 4).reshape(batch_size, self.timesymbols, self.subcarriers, -1)

        # Process template_pilot with convolution
        template_pilot = self.template_pilot_conv(template_pilot.permute(0, 3, 1, 2))
        template_pilot = template_pilot.permute(0, 2, 3, 1).reshape(batch_size, self.timesymbols, self.subcarriers, -1)

        # Stack y and processed template_pilot
        combined_input = torch.cat((y, template_pilot), dim=3).permute(0, 3, 1, 2)
        
        # Pass through ResBlock
        z = self.blocks(combined_input)
        
        # Apply final convolution
        z = self.final_conv(z)
        



        # Adjust output dimensions to match the target format
        z = z.permute(0, 2, 3, 1).reshape(batch_size, self.timesymbols, self.subcarriers, self.streams, self.num_bits_per_symbol)
        return z.permute(0, 3, 1, 2, 4)
