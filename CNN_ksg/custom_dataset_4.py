'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (4)***

    이번에는 Neural Network를 만들어 보겠습니다.
    이전 장에서 다뤘던 내용들을 한번 다시 되집어 볼까요?

    우리는 학습시킬 Neural Network를 class를 통해서 정의합니다.

    class "Neural Network의 이름"(nn.Module):
        def __init__(self):
            super(Neural Network의 이름",self).__init__()
            ~~~~~~~~~~~~~~~~
        def __forword(self,inputs):
            ~~~~~~~~~~~~~~~~

        위와 같은 형태로 선언 했던것 기억 나시나요?

        우리는 Convolution layer를 사용하기로 했으니까 Convolution 연산에 대해서 알아봅시다.

        자 빠르게 command창을 켜고(linux나 mac이라면 terminal)

        import torch.nn as nn을 하고
        dir(nn)명령어를 입력해 볼까요?
        엄청 나게 많은 것을이 나오는걸 보셨나요?
        dir은 괄호 안의 값에 속한 function이나 value를 보여주는 pythnon의 기본 기능입니다.
        내가 사용해야 되는 function이 무슨 기능이 있는지 아주 좋은 함수죠! (모르셨다면 어마어마한 꿀팁 아닙니까 정말?)
        거기 나와있는거 다 쓰시면 됩니다.
        CNN Architecture중 가장 간단한 LeNet-5를 만들어 볼껀데
        이제 시작해볼까요?
    '''


from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

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

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data=torchvision.datasets.ImageFolder(root='./train_data',transform=trans)
    trainloader=DataLoader(dataset=train_data,batch_size=4,shuffle=True,num_workers=4)

    net = NN()

    for num, data in enumerate(trainloader):
        print(data[0].shape)
        out = net(data[0])
        print(out.shape)
        break
