#coding:utf - 8
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from collections import OrderedDict
# from data_loader.data_loader import load_CIFIAR10
#import data_loader

import torchvision
import torchvision.transforms as transforms

#启动ＴＥＮＳＯＲＢＯＡＲＤｘrom
from tensorboardX import SummaryWriter
class Trainer(object):
    def __init__(self,settings,model,save_path):
        '''
        initialize class,define a template trainer for cnn        
        '''
        self.model = model.cuda()
        self.settings = settings
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.cuda()
        self.optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=float(self.settings['train_config']['learning_rate'])
        )
        self.logger = SummaryWriter(save_path)
        self.epoch_info = None
        trainset = torchvision.datasets.CIFAR10(root = './data',train=True,download=True,transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(trainset,batch_size = settings['model_config']['batchsize'][0],
                                shuffle = True,num_workers =settings['run_config']['num_workers'])
        testset = torchvision.datasets.CIFAR10(root = './data',train=False,download=True,transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(testset,batch_size = settings['model_config']['batchsize'][1],
                                shuffle = False,num_workers =settings['run_config']['num_workers'])




    def run(self,epoch, mode = 'Train'):
        if mode == "train":
            self.model.train()
            dataloader = self.train_loader
        elif mode == "val":
            self.model.eval()
            dataloader = self.test_loader

        

        if dataloader is None:
            raise ValueError("[{}] dataloader is None, please check your code".format(mode))
        
        
        data_t_start = time.time()
        n_iter = len(dataloader)
        loss_mean = 0

        runing_loss = 0.0
        runing_corrects = 0
        runing_acc = 0.0
        result_dict = {}
        total = 0.0
        for i,(feature,label) in enumerate(dataloader,0):
            data_t_end = time.time()

            feature = feature.cuda()
            label= label.cuda()
            iter_info = OrderedDict()
            # print("feature shape is :{}".format(feature.size()))
            # print("label is :{}".format(label.shape))
            torch.cuda.synchronize()
            fwd_start = time.time()
            self.optimizer.zero_grad()
            if mode == "train":
                output = self.model(feature)
            else:
                with torch.no_grad():
                    output = self.model(feature)
            torch.cuda.synchronize()
            fwd_time = time.time()-fwd_start
            # print("output:-----------{}".format(output.data))
            # predict = output
            _,predict= torch.max( output.data,1)
            # print("output shape is---------------- :{}".format(output.shape))
            loss = self.criterion(output,label)
    

            runing_loss += loss.item()
            num_correct = (predict==label).sum()
            accuracy  = (predict==label).float().mean()
            runing_acc +=(num_correct.item()/output.shape[0])
            if mode == "train":
                
                loss.backward()
                self.optimizer.step()
            
            data_time = data_t_end - data_t_start

            # if self.local_rank == 0:  # print log only for rank_0 GPU
            log_head = "{:s} Epoch [{:d}|{:d}] Iter [{:d}|{:d}], FWDTime: {:.3f}, DataTime: {:.3f}".format(
                mode, int(self.settings['train_config']['epochs']), epoch+1, n_iter, i+1, fwd_time, data_time)
            

            #print(log_head)
           # data_t_start = time.time()
            # if self.settings['run_config']['debug_mode']:
            #     break    
            # print(loss.item())
          #  runing_loss += loss.item()
            # print("predict:------{}".format(predict))
            # print("label:--------{}".format(label.data))
           # runing_corrects += torch.sum(predict == label.data)
           # total += label.size(0)
            # print("runing_corects:-----{}".format(runing_corrects))
            # print("dataoadser:--------{}".format(len(dataloader)))
            # acc =runing_corrects/len(dataloader)
            # print("acc:-------{}".format(acc))
        # epoch_loss = float(runing_loss) / total
        # epoch_acc = float(runing_corrects)/total
        iter_info['loss'] = runing_loss/len(dataloader)
        iter_info['precision'] = runing_acc/len(dataloader)
        print('Finish {} epoch,LossL:{:.6f},Acc:{:.6f}'.format(epoch+1,runing_loss/len(dataloader),runing_acc/len(dataloader)))
        # print("epoch_acc:--------{}".format(epoch_acc))

        self.logger.add_scalar("{:s}_LOSS".format(mode),runing_loss/len(dataloader),epoch)
        self.logger.add_scalar("{:s}_precision".format(mode),runing_acc/len(dataloader),epoch)

        

        #compute  avg
        # return result_dict


    # def init_dataloader(self):
    #     root = self.settings['dataset']['root']
    #     X_train,Y_train,X_test,Y_test = load_CIFIAR10(root)
    #     train_loader = torch.utils.data.DataLoader(X_train)

