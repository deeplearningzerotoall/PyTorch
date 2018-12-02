'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (5)***

    training을 위한 코드를 작성 중 optim과 loss를 결정하고 학습을 진행해 보겠습니다.
    optim과 loss에 대해서 배운 내용은 기억 나시는지요?

    4번에서 NN 이라는 Neural Network를 완성했으니
    5번에서는 optim과 loss function을 추가해서 학습을 진행해 봅시다.

    bonus!

    코드 배포를 했는데
    GPU만 있는사람도 있고
    CPU만 있는 사람도 있어요!

    git clone 하면 바로 실행할수 있도록 해주고 싶은데 어떻게 해야하나요?
    아래처럼 하세요~!
    ex)
    if(torch.cuda_is_available() ==1):
        device = 'cuda'
    else:
        device = 'cpu'

        ~~~~~~~

        model = model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
    '''


from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
data=0

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


if __name__ =="__main__":

    if (torch.cuda.is_available() ==1):
        print("cuda is available")
        device ='cuda'
    else:
        device = 'cpu'

    #device = 'cpu'

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data=torchvision.datasets.ImageFolder(root='./train_data',transform=trans)
    trainloader=DataLoader(dataset=train_data,batch_size=8,shuffle=True,num_workers=4)
    length = len(trainloader)
    print(length)

    net = NN().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=0.000005)
    loss_function = nn.CrossEntropyLoss()

    epochs =30
    for epoch in range(epochs):
        running_loss=0.0
        for num, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = net(inputs)

            loss = loss_function(out, labels)
            loss.backward()
            optim.step()

            running_loss +=loss.item()
            if num % length == (length-1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, num + 1, running_loss / length))
                running_loss = 0.0
