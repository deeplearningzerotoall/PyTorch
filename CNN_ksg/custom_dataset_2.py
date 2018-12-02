'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (2)***

    2. 읽어온 Custom Dataset에 대해서 transform 진행하기

    from torchvision import transoforms 추가

    transforms는 입력된 이미지를 transform 할 수 있는 함수들이 모여 있습니다.

    transforms를 통해 이미지의 크기와 type을 변경해 봅시다.

    transforms.Compose( ) 를 이용해서 변형을 한번에 여러개를 진행할 수 있습니다.

    transforms.Compose( 이 안에 list 형태로 transforms 명령어 들을 넣어주면 됩니다. )

    ex)
    trans = transforms.Compose( [transforms.Resize((256,256)),
                                 transforms.ToTensor()]
     )

     이번 코드에서는 transforms를 이용해서 기존의 origin_data를 작은 크기로 변환하고 저장하는 작업을 수행해 봅시다.

    '''


import torchvision
from torchvision import transforms

data=0

if __name__ =="__main__":

    #실제로 예를 들기 위해서 Resize만 진행했습니다.
    #다음장에서는 ToTensor까지 적용된 코드로 수행하도록 하겠습니다.

    trans = transforms.Compose([
        transforms.Resize((64,128)),
    ])
    train_data=torchvision.datasets.ImageFolder(root='./origin_data',transform=trans)

    for num, value in enumerate(train_data):
        data, label = value
        print(num, data, label)

        # 위에 import 하지는 않았지만 torchvision이 이미지를 PIL library를 이용해서 다루기 때문에
        # 자동으로 읽어진 이미지는 PIL의 data type을 가지게 됩니다.
        # 따라서 PIL.Image.Image의 기능중 save라는 기능을 이용해서 저장을 수행합니다.
        # 지금 코드에는 train_data folder가 있지만 삭제하고 직접 train_folder와 gray, red 폴더를 만들고
        # 아래 코드를 수행해보세요
        if(label==0):
            data.save("./train_data/gray/%d_%d.jpeg" % (num, label))
        else:
            data.save("./train_data/red/%d_%d.jpeg" % (num, label))

data.show()
