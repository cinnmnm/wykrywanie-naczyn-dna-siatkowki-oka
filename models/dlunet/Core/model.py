import torch

import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
	def __init__(self, in_ch, out_ch, dropout_rate=0.0):
		super().__init__()
		layers = [
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		]

		if dropout_rate > 0:
			layers.append(nn.Dropout2d(dropout_rate))
            
		layers.extend([
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		])
        
		if dropout_rate > 0:
			layers.append(nn.Dropout2d(dropout_rate))
            
		self.double_conv = nn.Sequential(*layers)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	def __init__(self, in_ch, out_ch, dropout_rate=0.0):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_ch, out_ch, dropout_rate)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Up(nn.Module):
	def __init__(self, in_ch, out_ch, dropout_rate=0.0):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_ch, out_ch, dropout_rate)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, base_c=64, dropout_rate=0.0):
		super().__init__()
        
		ch = [base_c, base_c*2, base_c*4, base_c*8, base_c*16]

		encoder_dropout = dropout_rate
		bottleneck_dropout = dropout_rate * 1.5 if dropout_rate > 0 else 0.0
		decoder_dropout = dropout_rate * 0.5 if dropout_rate > 0 else 0.0

		self.inc   = DoubleConv(n_channels, ch[0], 0.0)
		self.down1 = Down(ch[0], ch[1], encoder_dropout)
		self.down2 = Down(ch[1], ch[2], encoder_dropout)
		self.down3 = Down(ch[2], ch[3], encoder_dropout)
		self.down4 = Down(ch[3], ch[4], bottleneck_dropout)

		self.up1 = Up(ch[4], ch[3], decoder_dropout)
		self.up2 = Up(ch[3], ch[2], decoder_dropout)
		self.up3 = Up(ch[2], ch[1], decoder_dropout)
		self.up4 = Up(ch[1], ch[0], 0.0) 
		self.outc = OutConv(ch[0], n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits