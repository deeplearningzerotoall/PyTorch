'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (6)***

        학습도 다 했고 model이 나왔으니 이제 model을 저장하고 필요할때 써먹을수 있도록 하면 되겠네요?
        이제 모델 저장 및 불러오기를 진행해보고
        모델을 사용하는 방법을 익혀 볼까요?

        모델을 저장하기 전에 거쳐야할 작업이 있어요
        model.py 를 만들어서 내가 만들어 놓은 Neural Network를 분리 시키는작업이지요

        pytorch는 model file에서 state_dict()라는 명령어로 weight 값을 추출해 내는데요
        아쉽게도 깔끔하게 Network Architecture까지 한번에 저장하는 기능은 (있는데 에러가 자주 나서 그냥)
        없어요(라고할래요... 잘 하시는 분은 아래 댓글 달아주세요! 수정할게요!).

        그래서 아래와 같은 명령어를 통해서 model을 저장하고 불러올수 있어요!

        "After Training"

        torch.save(net.state_dict(),"./model/model_weight.pth")

        from model import model

        new_net = model.NN()

        new_net.load_state_dict(torch.load('./model/model_weight.pth'))

    '''

from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import model

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
    optim = torch.optim.Adam(net.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss()

    epochs = 30
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
                running_loss = 0.0


    torch.save(net.state_dict(),"./model/model.pth")
    new_net = model.NN()
    new_net.load_state_dict(torch.load('./model/model.pth',map_location=device))
    new_net = new_net.to(device)

    #아래 출력값이 전부 1이면 같은 weight value를 가지고 있는 거랍니다~

    print(net.conv1.weight == new_net.conv1.weight)

    # 직접 출력해서 확인하셔도 됩니다.
    print(net.conv1.weight[0][0])
    print(new_net.conv1.weight[0][0])




