import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool=nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*13*29,120)
        self.fc2 = nn.Linear(120,2)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(x.shape[0],-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x