# -*- coding: utf-8 -*-
"""

@author: Ammad
"""
from random import randint
import scipy.io
import os
import torch
import torch.nn as nn
import torch.utils.data
import sys

from TempCNNSelf import TempCNN
import numpy as np

#from PIL import Image
import pandas as pd
from glob import glob

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
#from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.utils.data 
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
#import Unet_Lakes_Nauman_VGG16 as unet_origin
#from resnet_Unet_BigEarthNet import UNetWithResnet50Encoder 

#import cv2
#from torch.utils.data.sampler import SubsetRandomSampler
# Parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# instantiate the network     


def train(net, train_loader, val_loader, test_loader, optimizer, epochs, scheduler=None, weights=None, save_epoch = 1):
    losses=[]; acc=[];mean_losses=[]
    mean_F1score=[]
    F1score=[0]
    iter_ =0 
    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (data, target, labels, ridx) in enumerate(train_loader):
            data, target, labels = Variable(data), Variable(target),Variable(labels)             
            
            optimizer.zero_grad()
            output = net(data)
           
           # output=F.log_softmax(output, dim=1)
            loss = criterion(output, target.flatten())
              
            loss.backward()
            optimizer.step()
       
    
            losses = np.append(losses,loss.item())
            #losses[iter_] = loss.item()
            mean_losses = np.append(mean_losses, np.mean(losses[max(0,iter_-100):iter_]))
            pred=output.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            gt=target.detach().cpu().numpy()
            #gt = np.argmax(gt, axis=1)
            precision, recall, fscore, support = score(gt.ravel(), pred.ravel(),zero_division= 1)
            if len(fscore)>1:
                F1score=np.append(F1score,np.mean(fscore))
            else:
                F1score=np.append(F1score,mean_F1score[-1])
            mean_F1score = np.append(mean_F1score, np.mean(F1score[max(0,iter_-50):iter_]))
            
            #print('Iteration: {}',iter_)
            
            if iter_ % 20 == 0:
         
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t  Image:{}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(),2+ridx[0]))                
                plt.plot(mean_losses) and plt.show()
                                
                print('The F1score of the Training Data')
                
                print(fscore)
                plt.plot(mean_F1score) and plt.show()
            
                
                #print(mylabels[np.where(gt[1,:])[0]])
            iter_ += 1
            
            del(data, target, loss)
        scheduler.step()    
        if e % save_epoch == 0:        
            val_acc = validation(net, val_loader)
            print('Validation Accuracy {}'.format(val_acc))
        
            acc = validation(net, test_loader)
           # acc = test(net, test_ids, IMAGE_FOLDER_Test, LABEL_FOLDER_Test, BATCH_SIZE, WINDOW_SIZE, len(LABELS))
            print('Test Accuracy {}'.format(acc))
            torch.save(net.state_dict(), './Crop_classification_TempCNN_epoch_{}_{:.3f}_{:.3f}'.format(e,acc,val_acc))
            
            
        
 
    return net
            #torch.save(net.state_dict(), './UNET_final_epoch')


def validation(model, val):
    model.eval()
    #tot_acc=[]
    avAccuracy=[]
    filename='Validation_Data.csv'
    Allpred=[]
    Allgt=[]
    
    with torch.no_grad():
        for batch_idx, (data, target, labels, ridx) in enumerate(val):
            data, target = Variable(data), Variable(target)
            output = model(data)
 
            output=F.log_softmax(output, dim=1)
            pred = np.argmax(output.data.cpu().numpy(), axis=1)
            #gt =  np.argmax(target.data.cpu().numpy(), axis=1)
            gt=target.data.cpu().numpy()
            Allpred=np.append(Allpred,pred.ravel())
            Allgt=np.append(Allgt,gt.ravel())
            
            
        precision, recall, fscore, support = score(pred.ravel(), gt.ravel(),labels=[0,1])
        df = pd.DataFrame(fscore)
        #if os.path.exists(filename):
        #    df.to_csv('Validation_Data.csv', mode='a', header=False)
        #else:
        #    df.to_csv('Validation_Data.csv', sep='\t',header=False)
           
           
        if fscore.shape[0]>1:
            avAccuracy=np.append(avAccuracy,np.mean(fscore))

        model.train()
        return np.mean(avAccuracy)  
    
    
class Crop_dataset(torch.utils.data.Dataset):
    def __init__(self,ids, IMAGE_FOLDER,LABEL_FOLDER, SeqLength,  Training=False, cache=False):
        super(Crop_dataset, self).__init__()
        self.Image_folder_dir = glob(IMAGE_FOLDER)
        self.LABEL_folder_dir = glob(LABEL_FOLDER)
        self.SeqLength=SeqLength
        self.Training=Training
        self.files=ids

    def __getitem__(self, Id):
        
        Im_Id=random.randint(0, len(self.files)-1)
        #print(Im_Id)
        image = scipy.io.loadmat(self.Image_folder_dir[self.files[Im_Id]])
        image=(1 /10000) * np.asarray(image['d'], dtype='float32')
        label = scipy.io.loadmat(self.LABEL_folder_dir[self.files[Im_Id]])
        label=np.asarray(label['Label'], dtype='int64')
   
        # Get a random patch
        #nD=random.uniform(0.8,1.2)
        samp,_,_=np.shape(image)
        #print(samp)
        #randB = [random.randint(0, samp) for i in range(self.Batch_s)]
        randB = random.randint(0, samp-1)
        image=image[randB,:,:]
        #image = np.expand_dims(image,axis=0)
        label=label[randB]
        image=image.transpose((1,0))       
        
 
        return (torch.from_numpy(image),torch.from_numpy(label),torch.from_numpy(label), self.files[Im_Id])

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 441972#len(self.files)



if __name__ == '__main__':#https://discuss.pytorch.org/t/brokenpipeerror-errno-32-broken-pipe-when-i-run-cifar10-tutorial-py/6224/4
    
    FOLDER = r'TrainData/'
    MAIN_FOLDER = FOLDER 
    IMAGE_FOLDER = MAIN_FOLDER + 'Images/*.mat'
    LABEL_FOLDER = MAIN_FOLDER + 'GT/*.mat'
    SeqLength =  17
    IN_CHANNELS = 3# Number of input channels (e.g. RGB)
    BATCH_SIZE = 8# Number of samples in a mini-batch


 
    
    fileIds = list(range(0,8))
    random.seed(0)
    random.shuffle(fileIds)

    train_ids = [1]
    val_ids = [2]
    test_ids = [0]
    #train_ids, val_ids, test_ids = fileIds[:int(len(fileIds)*0.02)], fileIds[int(len(fileIds)*0.7):int(len(fileIds)*0.8)], fileIds[int(len(fileIds)*0.8):]
    #fileIds=np.roll(fileIds,int(len(fileIds)*0.2))
  
    train_set= Crop_dataset(train_ids,IMAGE_FOLDER, LABEL_FOLDER, SeqLength,  Training=True)
    train_loader= torch.utils.data.DataLoader(train_set, batch_size= BATCH_SIZE, num_workers=0, shuffle=True)# sampler=sampler)
    
    
    val_set = Crop_dataset(val_ids,IMAGE_FOLDER, LABEL_FOLDER, SeqLength, Training=False)
    val_loader= torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)
    
    test_set = Crop_dataset(test_ids,IMAGE_FOLDER, LABEL_FOLDER, SeqLength, Training=False)
    test_loader= torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)
  
    #weights = [5.52, 0.5497]
    #class_weights = torch.FloatTensor(weights)
    criterion=nn.CrossEntropyLoss()#weight = class_weights)#pos_weight = torch.ones([43]))
    #criterion = nn.KLDivLoss(reduction='none')
    base_lr = 0.001
    
    net = TempCNN(input_dim=12, num_classes=2, sequencelength=17, kernel_size=7)
   

  
    net=net 

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.90, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [70,90], gamma=0.1)
    net = train(net, train_loader, val_loader, test_loader, optimizer,100, scheduler)
        
