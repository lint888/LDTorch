import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import Cla


#training part


traningData =
valData=

num_epochs = 10
batch_size =
lr2 =
optimizer = optim.Adam(lr=lr2,betas=(0.9,0.999), eps = 1e-8, weight_decay=0)
log_nth=10
train_loss_history=[]
train_acc_history=[]
val_acc_history=[]
val_loss_history=[]

model = Cla.Net() #initialieze the model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    train_acc_epoch=[]

    for i, data in enumerate(trainingData):

        truthlabels =

        optimizer.zero_grad()  #clear provious gradients


        prediction = model()   #calculate new outputs

        losses = nn.CrossEntropyLoss(prediction, truthlabels)

        losses.backward()

        optimizer.step()
        if i>0 and i%10 == 0:
            print('step{} / {}, loss {}'.format(i, len(traningData), losses.item()))

    print("Starting to evaluate the model...")
    model.eval()


