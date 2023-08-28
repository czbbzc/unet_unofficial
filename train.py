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

from model import unet
from dataset import MyDataset
from metrics import SegmentationMetric



def evaluate1(pred, label, num_classes):
    acc_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []
 
    num_img = pred.shape[0]
 
    for i in range(num_img):
        # imgPredict = np.array(pred[i].cpu())
        imgPredict = pred[i].cpu().detach().numpy()
        imgLabel = label[i].cpu().detach().numpy()
        # imgLabel = np.array(label[i].cpu())
 
        metric = SegmentationMetric(num_classes) 
        metric.addBatch(imgPredict, imgLabel)
        
        # print(metric.confusionMatrix)
        
        acc = metric.pixelAccuracy()
        macc = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
 
        acc_list.append(acc)
        macc_list.append(macc)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
 
        # print('{}: acc={}, macc={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))
 
    return np.mean(acc_list), np.mean(macc_list), np.mean(mIoU_list), np.mean(fwIoU_list)


def read_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='batch size')
    parser.add_argument('--validation', '-v', type=float, default=0.1,
                        help='percent of the data that is used as validation (0-1)')
    parser.add_argument('--num_classes', '-c', type=int, default=9, help='number of classes')
    parser.add_argument('--img_data', '-i', type=str, default='./data/img/01', help='path of image data')
    parser.add_argument('--mask_data', '-m', type=str, default='./data/mask/01', help='path of mask data')
    parser.add_argument('--results', '-r', type=str, default='./results', help='results')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = read_args()
    
    print('num_classes: ', args.num_classes)
    
    file_name = str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)
    result_path = os.path.join(args.results, file_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(result_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    model = unet(1, args.num_classes).to(device=device)
    
    # for layer in model.modules():
    #     if isinstance(layer, nn.Conv2d):
    #         N = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
    #         nn.init.normal_(layer.weight,std=N**0.5)
    
    # model = model.to(device=device)
    
    dataset = MyDataset(args.img_data, args.mask_data, 572)
    val = int(len(dataset) * args.validation)
    train = len(dataset) - val
    train_set, val_set = random_split(dataset, [train, val], generator=torch.Generator().manual_seed(0))
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=args.batch_size, num_workers=0)
    
    
    learning_rate = 1e-3
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=1e-8, momentum=0.99, foreach=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    # loss_func = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()
    metrics = SegmentationMetric(args.num_classes)
    
    train_iter_len = int(len(train_set) / args.batch_size)
    eval_iter_len = int(len(val_set) / args.batch_size)
    
    t1 = transforms.Resize((520, 696))
    
    for epoch in range(args.epochs):
        
        train_iter_now = 1
        model.train()
        train_loss = 0
        train_acc = []
        train_macc = []
        train_miou = []
        train_fwiou = []
        
        eval_iter_now = 1
        eval_loss = 0
        eval_acc = []
        eval_macc = []
        eval_miou = []
        eval_fwiou = []
        
        for imgs, targets in tqdm.tqdm(train_loader, desc=f'Train Epoch: {epoch+1}'):
            
            # targets = torch.LongTensor(targets)
            imgs = imgs.to(device=device, dtype=torch.float32)           
            targets = targets.to(device=device, dtype=torch.long)

            pred = model(imgs)           
            pred = transforms.functional.resize(pred, (targets.shape[1], targets.shape[2]))   
            sm_pred = torch.nn.functional.softmax(pred, dim=1)
            pred_mask = torch.argmax(sm_pred, dim=1)
                      
            acc, macc, miou, fwiou = evaluate1(pred_mask, targets, args.num_classes)
            
            train_acc.append(acc)
            train_macc.append(macc)
            train_miou.append(miou)
            train_fwiou.append(fwiou)
            
            loss_iter = loss_func(pred, targets)
            train_loss += loss_iter
                
            optimizer.zero_grad()
            loss_iter.backward()
            optimizer.step()

            train_iter_now += 1
            
            if (epoch+1) % 5 ==0 and (train_iter_now+1) == train_iter_len:

                # print('here: ', train_iter_now)
                
                origin = imgs[0]
                # print(origin)
                origin = transforms.functional.resize(origin, (targets.shape[1], targets.shape[2])) 
                writer.add_image('Train/origin', origin, epoch)
                
                gt = torch.as_tensor(targets[0], dtype=torch.float32)
                # gt[gt>0] = 1
                gt = gt/(args.num_classes - 1)
                gt = gt.unsqueeze(0)
                writer.add_image('Train/gt', gt, epoch)
                
                pred_pic = pred_mask[0]
                # pred_pic[pred_pic>0] = 1
                pred_pic = pred_pic.float()/(args.num_classes - 1)
                pred_pic = pred_pic.unsqueeze(0)
                writer.add_image('Train/pred', pred_pic, epoch)  
                
                

        
        # print(optimizer.param_groups[0]['lr'])
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch) 
        scheduler.step()         
                        
        
        print('train loss: ', train_loss, '\t',
              'acc: ', np.mean(train_acc), '\t',
              'macc: ', np.mean(train_macc), '\t',
              'miou: ', np.mean(train_miou), '\t',
              'fwiou: ', np.mean(train_fwiou), '\t'
              )
        
        writer.add_scalar('Train/train_loss', train_loss, epoch)
        writer.add_scalar('Train/train_acc', np.mean(train_acc), epoch)
        writer.add_scalar('Train/train_macc', np.mean(train_macc), epoch)
        writer.add_scalar('Train/train_miou', np.mean(train_miou), epoch)
        writer.add_scalar('Train/train_fwiou', np.mean(train_fwiou), epoch)
            
        if (epoch+1)%5 == 0:
            model.eval()
            with torch.no_grad():
                for imgs, targets in tqdm.tqdm(val_loader, desc=f'Eval Epoch: {epoch+1}'):
                    
                    imgs = imgs.to(device=device, dtype=torch.float32)           
                    targets = targets.to(device=device, dtype=torch.long)

                    pred = model(imgs)           
                    pred = transforms.functional.resize(pred, (targets.shape[1], targets.shape[2]))   
                    sm_pred = torch.nn.functional.softmax(pred, dim=1)
                    pred_mask = torch.argmax(sm_pred, dim=1)
                            
                    acc, macc, miou, fwiou = evaluate1(pred_mask, targets, args.num_classes)
                
                    eval_acc.append(acc)
                    eval_macc.append(macc)
                    eval_miou.append(miou)
                    eval_fwiou.append(fwiou)
                    
                    
                    loss_iter = loss_func(pred, targets)
                    eval_loss += loss_iter

                    eval_iter_now += 1 
                    
                    if (epoch+1) % 5 == 0 and (eval_iter_now+1) == eval_iter_len:

                        # print('eval here: ', eval_iter_now)
                        
                        origin = imgs[0]
                        # print(origin)
                        origin = transforms.functional.resize(origin, (targets.shape[1], targets.shape[2])) 
                        writer.add_image('Eval/origin', origin, epoch)
                        
                        gt = torch.as_tensor(targets[0])
                        # gt[gt>0] = 1
                        gt = gt.float()/(args.num_classes - 1)
                        gt = gt.unsqueeze(0)
                        writer.add_image('Eval/gt', gt, epoch)
                        
                        pred_pic = pred_mask[0]
                        # pred_pic[pred_pic>0] = 1
                        pred_pic = pred_pic.float()/(args.num_classes - 1)
                        pred_pic = pred_pic.unsqueeze(0)
                        writer.add_image('Eval/pred', pred_pic, epoch)  
                        
                      
                
                print('eval loss: ', eval_loss/eval_iter_len*train_iter_len, '\t',
                    'acc: ', np.mean(eval_acc), '\t',
                    'macc: ', np.mean(eval_macc), '\t',
                    'miou: ', np.mean(eval_miou), '\t',
                    'fwiou: ', np.mean(eval_fwiou), '\t'
                    )
                
                writer.add_scalar('Eval/eval_loss', eval_loss/eval_iter_len*train_iter_len, epoch)
                writer.add_scalar('Eval/eval_acc', np.mean(eval_acc), epoch)
                writer.add_scalar('Eval/eval_macc', np.mean(eval_macc), epoch)
                writer.add_scalar('Eval/eval_miou', np.mean(eval_miou), epoch)
                writer.add_scalar('Eval/eval_fwiou', np.mean(eval_fwiou), epoch)
            
        if (epoch+1)%20 == 0:
            name = str(epoch) + '_' + str(train_loss) + '_' + str(eval_loss) + '.pth'
            save_path = os.path.join(result_path, name)
            torch.save(model.state_dict(), save_path)
