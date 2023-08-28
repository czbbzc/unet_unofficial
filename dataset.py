import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class MyDataset(Dataset):
    
    def __init__(self, img_dir, mask_dir, pic_size):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        self.img_file_list = os.listdir(self.img_dir)
        self.img_file_list = sorted(self.img_file_list, key=lambda x:int(x[1:4]))
        
        self.img_list = []
        for i in range(len(self.img_file_list)):
            path = os.path.join(self.img_dir, self.img_file_list[i])
            self.img_list.append(path)
            
            # ll = os.listdir(path)
            # ll = sorted(ll, key=lambda x:int(x[1:4]))
            # for j in range(len(ll)):
            #     path_img = os.path.join(path, ll[j])
            #     self.img_list.append(path_img)    
                
        self.mask_file_list = os.listdir(self.mask_dir)
        self.mask_file_list = sorted(self.mask_file_list, key=lambda x:int(x[7:10]))
        
        self.mask_list = []
        for i in range(len(self.mask_file_list)):
            path = os.path.join(self.mask_dir, self.mask_file_list[i])
            self.mask_list.append(path)  
            
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5),(0.5)),
                transforms.Resize((pic_size, pic_size))])
        
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor()])
        

            
            # ll = os.listdir(path)
            # ll = sorted(ll, key=lambda x:int(x[7:10]))
            # for j in range(len(ll)):
            #     path_mask = os.path.join(path, ll[j])
            #     self.mask_list.append(path_mask)  
            
        
        
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        mask_path = self.mask_list[index]
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        # mask[mask>0]=1
        
        img = self.img_transform(img.copy())
        mask = torch.as_tensor(mask.copy()).long().contiguous()
        
        # img = img.reshape(1, img.shape[0], img.shape[1])
        # mask = mask.reshape(1, mask.shape[0], mask.shape[1])
        
        # mask = mask.long()
        
        return img, mask
    
    
    def __len__(self):
        return len(self.img_file_list)
        
        
    
# imgdir = './data/img/01'
# maskdir = './data/mask/01'

# mask1 = './data/mask/01/man_seg023.tif'

# m1 = Image.open(mask1)
# m1 = np.array(m1)
# m1 = m1 * (255/9)
# m1 = Image.fromarray(m1)
# m1.show()

# data1 = MyDataset(imgdir, maskdir, 572)

# img, mask = data1[5]

# print(img.shape)
# print(mask.shape)

# # max_num = 0
# num_list = []

# for j in range(len(data1)):
#     img1, mask1 = data1[j]
#     for i in mask1:
#         for z in i:
#             if (z != 0) and (z not in num_list):
#                 print(z, type(z))
#                 num_list.append(z)
                
# print(num_list)


# dataloader1 = DataLoader(data1, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# # print((enumerate(dataloader1)))

# for data in dataloader1:
#     imgs, targets = data

#     print(imgs.shape)
#     print(targets.shape)
    
#     for i in targets[0]:
#         for z in i:
#             if (z != 0) and (z not in num_list):
#                 print(z, type(z))
#                 num_list.append(z)
    
#     break

    # for i in targets:
    #     for z in i:
    #         if (z != 0) and (z not in num_list):
    #             print(z, type(z))
    #             num_list.append(z)

# t1 = transforms.Resize((targets.shape[2], targets.shape[3]))
# imgs1 = transforms.functional.resize(imgs, (targets.shape[2], targets.shape[3]))
# print(imgs1.shape)



# print(len(data1))
