import torch
import torch.nn as nn
import numpy
import csv
import numpy as np
preds=[[[1,2,3,4],[3,4,5,6],[5,6,7,8]],[[4,5,6,7],[6,7,8,9],[8,9,10,11]],[[4,5,6,7],[6,7,8,9],[8,9,10,11]]]
trial=np.array([[1,2,3],[4,5,6],[7,8,9]])
sum1=[]
sum2=[]
for i,j,m in preds:
    sum1.append(i)
    sum2.append(j)
sum1=torch.tensor([[[1,2,3,4],[3,4,5,6]],[[1,2,3,4],[3,4,5,6]]])
print(sum1.size())
sum2=torch.tensor([[[6,7],[9,10]],[[6,7],[9,10]]])
sum3=[];
for i,j in sum1:
    sum3.append(i)
    sum3.append(j)
sum4=torch.cat(sum3,dim=0)

with open('./covid.train.csv', 'r') as fp:
    data = list(csv.reader(fp))
    data = np.array(data[1:])[:, 1:].astype(float)

loss=nn.MSELoss(reduction='mean')
pred=torch.tensor([[1.,2,3],[2,3,5]])
targ=torch.tensor([[5.,3,4],[2,3,4]])
total=loss(pred,targ)

config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}
optimizer = getattr(torch.optim, config['optimizer'])

