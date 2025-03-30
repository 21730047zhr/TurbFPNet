import argparse
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from model_PRT import *
import cv2
from os.path import join
from utils import load_state_dict

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--model_name', default='./model/TurbFPNet_exp.pth', type=str, help='generator model epoch name')
parser.add_argument('--image_path', default='./test_data/', type=str, help='generator model epoch name')
parser.add_argument('--image_name', default='00b', type=str, help='generator model epoch name')
parser.add_argument('--save_path', default='./test_results/' , type=str, help='turn on img augmentation (default: default)')
opt = parser.parse_args()


MODEL_NAME = opt.model_name
IMAGE_PATH = opt.image_path
SAVE_PATH = opt.save_path
IMAGE_NAME = opt.image_name


image_filename_lr = IMAGE_NAME+'-1-1.png'
img = cv2.imread(join(IMAGE_PATH, image_filename_lr),-1)
maskSize = np.shape(img)
padTop = (maskSize[0]+31)//32*32 - maskSize[0]
padLeft = (maskSize[1]+31)//32*32 - maskSize[1]

x=[]
for idx in range(9):
	image_filename_lr = IMAGE_NAME+'-1-'+str(idx+1)+'.png'
	img = cv2.imread(join(IMAGE_PATH, image_filename_lr),-1)
	img = np.float32(img)
	img = np.pad(img, ((padTop, 0), (padLeft, 0)), 'reflect')
	img = (img - np.min(img)) / (np.max(img)-np.min(img))
	img = torch.FloatTensor(img)
	x.append(img)

lr_in = torch.stack(x, axis=0).unsqueeze(1)
lr_in = lr_in.cuda()
with torch.no_grad():
	model = TurbFPNet()
	model.cuda()
	model = load_state_dict(model, MODEL_NAME)
	start = time.perf_counter()
	_, out = model(lr_in)
	elapsed = (time.perf_counter() - start)
	print('cost' + str(elapsed) + 's')
	sr_img = torch.mean(out[:,:,-maskSize[0]*2:,-maskSize[1]*2:], dim=0, keepdim=True)
	tmp = sr_img.detach().cpu().numpy().squeeze()
	tmp = (tmp - np.min(tmp))/ (np.max(tmp) - np.min(tmp)) * 65535
	cv2.imwrite(SAVE_PATH+'results_'+IMAGE_NAME+'.png',(tmp).astype("uint16"))
	del model
		
		


	

















