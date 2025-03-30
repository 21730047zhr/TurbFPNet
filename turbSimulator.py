import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from utils import load_state_dict

class amp_simulator(nn.Module): 
	def __init__(self, Dr0, img_size, thre, corr = -4, data_path = './PRT_model', device = 'cuda:0', scale = 1.0, use_temp = False):
		super().__init__()

		self.img_size = img_size
		self.initial_grid = 16
		self.Dr0 = torch.tensor(Dr0)
		self.device = torch.device(device)
		self.Dr0 = torch.tensor(Dr0).to(self.device,dtype=torch.float32)
		self.mapping = amp_P2S()
		self.mapping = load_state_dict(self.mapping, './PRT_model/PRT_amp_model.pt')
		self.dict_psf = np.load('./PRT_model/dictionary.npy', allow_pickle = True)

		self.mu = torch.tensor(self.dict_psf.item()['mu']).reshape((1,1,33,33)).to(self.device,dtype=torch.float32)
		self.dict_psf = torch.tensor(self.dict_psf.item()['dictionary'][:100,:]).reshape((100,1,33,33))
		self.dict_psf = self.dict_psf.to(self.device,dtype=torch.float32)

		self.R = np.load(os.path.join(data_path,'R-corr_{}.npy'.format(corr)))
		self.R = torch.tensor(self.R).to(self.device,dtype=torch.float32)
		self.offset = torch.tensor([31,31]).to(self.device,dtype=torch.float32)

		if use_temp: 
			self.S_half = np.load(os.path.join(data_path,'S_half-temp.npy'.format(img_size,Dr0)), allow_pickle=True)
		else:
			self.S_half = np.load(os.path.join(data_path,'S_half-size_{}-D_r0_{:.4f}_thre_{}.npy'.format(img_size,Dr0, thre)), allow_pickle=True)
		self.const = self.S_half.item()['const']
		self.S_half = torch.tensor(self.S_half.item()['s_half']).to(self.device,dtype=torch.float32)

		xx = torch.arange(0, img_size).view(1,-1).repeat(img_size,1)
		yy = torch.arange(0, img_size).view(-1,1).repeat(1,img_size)
		xx = xx.view(1,1,img_size,img_size).repeat(1,1,1,1)
		yy = yy.view(1,1,img_size,img_size).repeat(1,1,1,1)
		self.grid = torch.cat((xx,yy),1).permute(0,2,3,1).to(self.device,dtype=torch.float32)
		self.scale=scale

	def forward(self, img): 

		img_pad = F.pad(img.view((-1,1,self.img_size,self.img_size)), (16,16,16,16), mode = 'reflect')

		img_mean = F.conv2d(img_pad, self.mu).squeeze()

		dict_img = F.conv2d(img_pad, self.dict_psf)

		random_ = torch.sqrt(self.Dr0**(5/3)) * torch.randn((self.initial_grid**2 * 36),1,device=self.device)

		zer = torch.matmul(self.R,random_).view(self.initial_grid,self.initial_grid,36).permute(2,0,1).unsqueeze(0)

		zer = F.interpolate(zer,size=(self.img_size,self.img_size),mode='bilinear', align_corners=False)

		zer = zer * self.scale

		weight = self.mapping(zer.squeeze().permute(1,2,0).view(self.img_size**2,-1))

		weight = weight.view((self.img_size,self.img_size,100)).permute(2,0,1)

		out = weight.unsqueeze(0) * dict_img

		out = torch.sum(out,1) + img_mean

		pos = torch.fft.irfft2((self.S_half.permute(1, 2, 0).unsqueeze(0) * torch.randn(1, self.img_size,
								self.img_size, 2, device=self.device)), s=(self.img_size,self.img_size), dim=(1,2)) * self.const

		flow = 2.0*(self.grid+pos) / (self.img_size-1) - 1.0

		out = F.grid_sample(out.view((1,-1,self.img_size,self.img_size)), flow, 'bilinear', padding_mode='border', align_corners=False).squeeze()

		return out


class amp_P2S(nn.Module): 
	def __init__(self, input_dim = 36, hidden_dim = 100, output_dim = 100): 
		super().__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x): 

		y = F.relu(self.fc1(x))

		y = F.relu(self.fc2(y))

		y = F.relu(self.fc2(y))

		out = self.fc3(y)

		return out


class phase_simulator(nn.Module): 
	def __init__(self, Dr0, img_size, thre, corr = -4, data_path = './PRT_model', device = 'cuda:0', scale = 0.01, use_temp = False):
		super().__init__()

		self.img_size = img_size
		self.Dr0 = torch.tensor(Dr0)
		self.device = torch.device(device)
		self.Dr0 = torch.tensor(Dr0).to(self.device,dtype=torch.float32)
		self.mapping = phase_P2S()
		self.mapping = load_state_dict(self.mapping, './PRT_model/PRT_phase_model.pt')

		self.R = np.load(os.path.join(data_path,'R-corr_{}.npy'.format(corr)))
		self.R = torch.tensor(self.R).to(self.device,dtype=torch.float32).reshape(1,1,9216,9216)

		if use_temp: 
			self.S_half = np.load(os.path.join(data_path,'S_half-temp.npy'.format(img_size,Dr0)), allow_pickle=True)
		else:
			self.S_half = np.load(os.path.join(data_path,'S_half-size_{}-D_r0_{:.4f}_thre_{}.npy'.format(img_size,Dr0, thre)), allow_pickle=True)
		self.const = self.S_half.item()['const']
		self.S_half = torch.tensor(self.S_half.item()['s_half']).to(self.device,dtype=torch.float32)

		xx = torch.arange(0, img_size).view(1,-1).repeat(img_size,1)
		yy = torch.arange(0, img_size).view(-1,1).repeat(1,img_size)
		xx = xx.view(1,1,img_size,img_size).repeat(1,1,1,1)
		yy = yy.view(1,1,img_size,img_size).repeat(1,1,1,1)
		self.grid = torch.cat((xx,yy),1).permute(0,2,3,1).to(self.device,dtype=torch.float32)
		self.scale=scale

	def forward(self, img): 

		zer = torch.sqrt(self.Dr0**(5/3)) * F.interpolate(self.R, size=(self.img_size,self.img_size), mode='bilinear', align_corners=False)

		out = img + self.mapping(zer.permute(0,2,3,1)).permute(0,3,1,2) * self.scale

		pos = torch.fft.irfft2((self.S_half.permute(1, 2, 0).unsqueeze(0) * torch.randn(1, self.img_size,
								self.img_size, 2, device=self.device)), s=(self.img_size,self.img_size), dim=(1,2)) * self.const

		flow = 2.0*(self.grid+pos) / (self.img_size-1) - 1.0

		out = F.grid_sample(out.view((1,-1,self.img_size,self.img_size)), flow, 'bilinear', padding_mode='border', align_corners=False).squeeze()

		return out


class phase_P2S(nn.Module): 
	def __init__(self, input_dim = 1, hidden_dim = 16, output_dim = 1): 
		super().__init__()

		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x): 

		y = F.relu(self.fc1(x))

		for _ in range(7):

			y = F.relu(self.fc2(y))

		out = self.fc3(y)

		return out

