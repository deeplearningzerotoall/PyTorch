'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (6_wandb) ***

    This file can run on Mac and linux. 

    Use this code you have to initialize.

    In this code, I explain how to use wandb!

    More detail, you can check in "https://www.wandb.com"


    '''

from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import model

#import wandb library
import wandb


data = 0


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 13 * 29, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    
    #wandb initialize function
    wandb.init()

    if (torch.cuda.is_available() == 1):
        print("cuda is available")
        device = 'cuda'
    else:
        device = 'cpu'

    # device = 'cpu'

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root='./train_data', transform=trans)
    trainloader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=4)
    length = len(trainloader)
    print(length)

    net = NN().to(device)
    #if you use wandb.hook_torch(net) we can check our model's information
    wandb.hook_torch(net)
    optim = torch.optim.Adam(net.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss()

    epochs = 15
    for epoch in range(epochs):
        running_loss = 0.0
        for num, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = net(inputs)

            loss = loss_function(out, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if num % length == (length - 1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, num + 1, running_loss / length))
                #through wandb.log we can check running_loss
                wandb.log({'loss' :running_loss})
                running_loss = 0.0

    torch.save(net.state_dict(),"./model/model.pth")
    new_net = model.NN()
    new_net.load_state_dict(torch.load('./model/model.pth',map_location=device))
    new_net = new_net.to(device)


