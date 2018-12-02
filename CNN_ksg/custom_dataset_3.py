'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (3)***

    이제 training을 위한 코드를 작성해 보도록 하겠습니다.

    datasets으로 불러오는 방법을 수행해보죠

    우선 form torch.utils.data import DataLoader 명령으로

    DataLoader를 가져 옵니다.

    DataLoader에는 아까 만든 train_data를 넣어주고 몇가지 인자를 추가하여 값을 넣어줍니다.

    torch.utils.data.DataLoader가 입력을 받는 자세한 값들은 아래 링크에서 확인해보세요
    https://pytorch.org/docs/master/data.html?highlight=dataloader#torch.utils.data.DataLoader

    간단한 것들만 살펴보겠습니다.

    dataset     : torchvision.datasets.ImageFolder로 만들어낸 train_data값을 넣어주면 됩니다.
                  이어서 진행할 강의에서 사용할 torchvision.datasets.이하의 dataset도 불러온 다음  dataset = 하고 넣어주시면 됩니다.
                  사용방법은 아래를 참고하세요.

    batch_size  : batch_size는 말그대로 batch_size 입니다.
    shuffle     : dataset을 섞는 겁니다.
    num_worker  : 데이터 loader를 하기 위해 사용할 core의 수를 결정합니다. core가 많을 수록 빠릅니다.
    '''


from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

data=0

if __name__ =="__main__":

    trans = transforms.Compose([
        #아까 없던 코드니까 추가해주세요!
        transforms.ToTensor()
    ])
    train_data=torchvision.datasets.ImageFolder(root='./train_data',transform=trans)

    trainloader=DataLoader(dataset=train_data,batch_size=4,shuffle=True,num_workers=4)

    for num, data in enumerate(trainloader):
        #num은 몇번쨰 값인지
        #data[0]는 input으로 들어갈 data value
        #data[1]는 label 값 입니다.
        #batch_size 값을 바꿔가면서 아래 print 값을 확인해 보세요!
        print(num, type(data[0]), data[1])