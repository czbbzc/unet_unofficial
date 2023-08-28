import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import tqdm
from pathlib import Path
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import unet
from dataset import MyDataset
from metrics import SegmentationMetric


def read_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--input', '-i', type=str, default='predict_input/t003.tif', help='predict pic input')
    parser.add_argument('--size_input', '-s', type=int, default=572, help='size of model input')
    parser.add_argument('--model', '-m', type=str, default='predict_input/499.pth', help='predict model input')
    parser.add_argument('--num_classes', '-c', type=int, default=9, help='number of classes')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = read_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet(1, args.num_classes).to(device=device)

    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((args.size_input, args.size_input))])
    
    img0 = Image.open(args.input)
    img0 = np.array(img0, dtype=np.uint8)
    img = img_transform(img0.copy())
    img = img.unsqueeze(0).to(device=device)
    
    pred = model(img)
    output = transforms.functional.resize(pred, (img0.shape[0], img0.shape[1]))
    output = torch.argmax(output, dim=1)[0]
    # output = output.unsqueeze(0)
    
    pred_pic = np.array(output.cpu(), dtype=np.uint8)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img0, cmap='gray')
    ax[1].imshow(pred_pic, cmap='gray')
    plt.show()
    
    # classes = pred_pic.max() + 1
    # fig, ax = plt.subplots(1, classes + 1)
    # ax[0].set_title('Input image')
    # ax[0].imshow(img0, cmap='gray')
    # for i in range(classes):
    #     ax[i + 1].set_title(f'Mask (class {i + 1})')
    #     ax[i + 1].imshow(pred_pic == i, cmap='gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    
    
    
    # pred_pic = pred_pic*int(255/(args.num_classes-1))
    # print(output.shape)
    # print(pred_pic.shape)
    # pred_pic = Image.fromarray(pred_pic)
    # pred_pic.show()
    
    
