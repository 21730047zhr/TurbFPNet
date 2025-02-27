import torch
import torch.nn as nn
import torch.nn.functional
from functools import partial
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math


class DWConv(nn.Module):
	def __init__(self, dim=512):
		super(DWConv, self).__init__()

		self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

	def forward(self, x, H, W):

		B, N, C = x.shape

		x = x.transpose(1, 2).view(B, C, H, W)

		x = self.dwconv(x)

		x = x.flatten(2).transpose(1, 2)

		return x


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(ConvLayer, self).__init__()

		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

	def forward(self, x):

		out = self.conv2d(x)

		return out


class UpsampleConvLayer(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(UpsampleConvLayer, self).__init__()

		self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

	def forward(self, x):

		out = self.conv2d(x)

		return out


class ResidualBlock(torch.nn.Module):
	def __init__(self, channels):
		super(ResidualBlock, self).__init__()

		self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
		self.relu = nn.ReLU()

	def forward(self, x):

		residual = x

		out = self.relu(self.conv1(x))

		out = self.conv2(out) * 0.1

		out = torch.add(out, residual)

		return out


class TransformerGroup(nn.Module):
	def __init__(self, img_size=160, patch_size=3, in_chans=1, num_classes=1, embed_dims=[32, 64, 128, 320, 512],
				num_heads=[1, 1, 2, 4, 8], convFFN_ratios=[2, 2, 2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
				attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 3, 4, 6, 3],
				sr_ratios=[8, 8, 4, 2, 1]):
		super().__init__()

		self.num_classes = num_classes
		self.depths = depths

		self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=2, in_chans=in_chans,
												embed_dim=embed_dims[0])
		self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[0],
												embed_dim=embed_dims[1])
		self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[1],
												embed_dim=embed_dims[2])
		self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[2],
												embed_dim=embed_dims[3])
		self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[3],
												embed_dim=embed_dims[4])

		# Stage 1 (h/2 x w/2)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
		cur = 0
		self.block1 = nn.ModuleList([TransformerBlock(dim=embed_dims[0], num_heads=num_heads[0], convFFN_ratio=convFFN_ratios[0],
														qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
														attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
														sr_ratio=sr_ratios[0])
									for i in range(depths[0])])
		self.norm1 = norm_layer(embed_dims[0])

		# Stage 2 (h/4 x w/4)
		cur += depths[0]
		self.block2 = nn.ModuleList([TransformerBlock(dim=embed_dims[1], num_heads=num_heads[1], convFFN_ratio=convFFN_ratios[1],
														qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
														attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
														sr_ratio=sr_ratios[1])
									for i in range(depths[1])])
		self.norm2 = norm_layer(embed_dims[1])

		# Stage 3 (h/8 x w/8)
		cur += depths[1]
		self.block3 = nn.ModuleList([TransformerBlock(dim=embed_dims[2], num_heads=num_heads[2], convFFN_ratio=convFFN_ratios[2],
														qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
														attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
														sr_ratio=sr_ratios[2])
									for i in range(depths[2])])
		self.norm3 = norm_layer(embed_dims[2])

		# Stage 4 (h/16 x w/16)
		cur += depths[2]
		self.block4 = nn.ModuleList([TransformerBlock(dim=embed_dims[3], num_heads=num_heads[3], convFFN_ratio=convFFN_ratios[3],
														qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
														attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
														sr_ratio=sr_ratios[3])
									for i in range(depths[3])])
		self.norm4 = norm_layer(embed_dims[3])

		# Stage 5 (h/32 x w/32)
		cur += depths[3]
		self.block5 = nn.ModuleList([TransformerBlock(dim=embed_dims[4], num_heads=num_heads[4], convFFN_ratio=convFFN_ratios[4],
														qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
														attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
														sr_ratio=sr_ratios[4])
									for i in range(depths[4])])
		self.norm5 = norm_layer(embed_dims[4])

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def reset_drop_path(self, drop_path_rate):
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
		cur = 0
		for i in range(self.depths[0]):
			self.block1[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[0]
		for i in range(self.depths[1]):
			self.block2[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[1]
		for i in range(self.depths[2]):
			self.block3[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[2]
		for i in range(self.depths[3]):
			self.block4[i].drop_path.drop_prob = dpr[cur + i]

		cur += self.depths[3]
		for i in range(self.depths[4]):
			self.block5[i].drop_path.drop_prob = dpr[cur + i]

	def forward_features(self, x):
		B = x.shape[0]
		outs = []

		# stage 1
		x1, H1, W1 = self.patch_embed1(x)
		for i, blk in enumerate(self.block1):
			x1 = blk(x1, H1, W1)
		x1 = self.norm1(x1)
		x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x1)

		# stage 2
		x1, H1, W1 = self.patch_embed2(x1)
		for i, blk in enumerate(self.block2):
			x1 = blk(x1, H1, W1)
		x1 = self.norm2(x1)
		x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x1)

		# stage 3
		x1, H1, W1 = self.patch_embed3(x1)
		for i, blk in enumerate(self.block3):
			x1 = blk(x1, H1, W1)
		x1 = self.norm3(x1)
		x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x1)

		# stage 4
		x1, H1, W1 = self.patch_embed4(x1)
		for i, blk in enumerate(self.block4):
			x1 = blk(x1, H1, W1)
		x1 = self.norm4(x1)
		x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x1)

		# stage 5
		x1, H1, W1 = self.patch_embed5(x1)
		for i, blk in enumerate(self.block5):
			x1 = blk(x1, H1, W1)
		x1 = self.norm5(x1)
		x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
		outs.append(x1)

		return outs

	def forward(self, x):

		x = self.forward_features(x)

		return x


class OverlapPatchEmbed(nn.Module):
	def __init__(self, img_size=160, patch_size=3, stride=1, in_chans=3, embed_dim=512):
		super().__init__()

		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)

		self.img_size = img_size
		self.patch_size = patch_size
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
								padding=(patch_size[0] // 2, patch_size[1] // 2))
		self.norm = nn.LayerNorm(embed_dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x):

		x = self.proj(x)

		_, _, H, W = x.shape

		x = x.flatten(2).transpose(1, 2)

		x = self.norm(x)

		return x, H, W


class ConvFFN(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()

		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.dwconv = DWConv(hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):

		x = self.fc1(x)

		x = self.dwconv(x, H, W)

		x = self.act(x)

		x = self.drop(x)

		x = self.fc2(x)

		x = self.drop(x)

		return x


class ReductionSpatialAttention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
		super().__init__()

		self.dim = dim
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** - 0.5

		self.q = nn.Linear(dim, dim, bias=qkv_bias)
		self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

		self.sr_ratio = sr_ratio
		if sr_ratio > 1:
			self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
			self.norm = nn.LayerNorm(dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):

		B, N, C = x.shape

		q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

		if self.sr_ratio > 1:
			x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
			x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
			x_ = self.norm(x_)
			kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		else:
			kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

		k, v = kv[0], kv[1]

		attn = (q @ k.transpose(-2, -1)) * self.scale

		attn = attn.softmax(dim=-1)

		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)

		x = self.proj(x)

		x = self.proj_drop(x)

		return x


class TransformerBlock(nn.Module):
	def __init__(self, dim, num_heads, convFFN_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
		super().__init__()

		self.norm1 = norm_layer(dim)
		self.attn = ReductionSpatialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
							  attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		convFFN_hidden_dim = int(dim * convFFN_ratio)
		self.convFFN = ConvFFN(in_features=dim, hidden_features=convFFN_hidden_dim, act_layer=act_layer, drop=drop)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x, H, W):

		x = x + self.drop_path(self.attn(self.norm1(x), H, W))

		x = x + self.drop_path(self.convFFN(self.norm2(x), H, W))

		return x


class ResidualGroup(nn.Module):
	def __init__(self, embed_dims=[8, 16, 64, 128, 320, 512], **kwargs):
		super(ResidualGroup, self).__init__()

		self.convd32x = UpsampleConvLayer(embed_dims[-1], embed_dims[-2], kernel_size=4, stride=2)
		self.dense_5 = nn.Sequential(ResidualBlock(embed_dims[-2]))

		self.convd16x = UpsampleConvLayer(embed_dims[-2], embed_dims[-3], kernel_size=4, stride=2)
		self.dense_4 = nn.Sequential(ResidualBlock(embed_dims[-3]))

		self.convd8x = UpsampleConvLayer(embed_dims[-3], embed_dims[-4], kernel_size=4, stride=2)
		self.dense_3 = nn.Sequential(ResidualBlock(embed_dims[-4]))

		self.convd4x = UpsampleConvLayer(embed_dims[-4], embed_dims[-5], kernel_size=4, stride=2)
		self.dense_2 = nn.Sequential(ResidualBlock(embed_dims[-5]))

		self.convd2x = UpsampleConvLayer(embed_dims[-5], embed_dims[-6], kernel_size=4, stride=2)
		self.dense_1 = nn.Sequential( ResidualBlock(embed_dims[-6]))

	def forward(self,x1):

		res16x = self.convd32x(x1[4])

		res16x = self.dense_5(res16x) + x1[3]

		res8x = self.convd16x(res16x)

		res8x = self.dense_4(res8x) + x1[2]

		res4x = self.convd8x(res8x)

		res4x = self.dense_3(res4x) + x1[1]

		res2x = self.convd4x(res4x)

		res2x = self.dense_2(res2x) + x1[0]

		res1x = self.convd2x(res2x)

		x = self.dense_1(res1x)

		return x


class TurbFPNet(nn.Module):

	def __init__(self, path=None, **kwargs):
		super(TurbFPNet, self).__init__()

		self.PyramidEncoder = TransformerGroup(img_size=160, patch_size=16, in_chans=1, num_classes=1,
												embed_dims=[16, 64, 128, 320, 512], num_heads=[1, 1, 2, 5, 8],
												convFFN_ratios=[4, 8, 8, 4, 4], qkv_bias=False, qk_scale=None,
												drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
												norm_layer=nn.LayerNorm, depths=[3, 3, 6, 8, 3], sr_ratios=[4, 8, 4, 2, 1])

		self.ResidualDecoder = ResidualGroup(embed_dims=[8, 16, 64, 128, 320, 512])
		self.upSample = UpsampleConvLayer(8, 2, kernel_size=4, stride=2)
		self.upSampleResidual = nn.Sequential(ResidualBlock(2))
		self.conv_output = ConvLayer(2, 1, kernel_size=3, stride=1, padding=1)
		self.active = nn.Tanh()

	def forward(self, x):

		x1 = self.PyramidEncoder(x)

		x = self.ResidualDecoder(x1)

		lr_sum = torch.sum(torch.sum(x, dim = 1, keepdim =True), dim = 0, keepdim =True)

		x = self.upSampleResidual(self.upSample(x))

		clean = self.active(self.conv_output(x))

		return lr_sum, 0.5*(clean+1)

