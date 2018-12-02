'''
    모두를 위한 딥러닝 시즌2 pytorch
    *** Custom Dataset은 어떻게 쓰나요? (1)***
    맨날 나오는 MNIST가 지겨워서 새로운 데이터셋을 만들어 봤습니다.
    여러분이 가지고 있는 사진을 분류해보고 싶을수 있으니 아래 프로젝트를 따라가면서
    내가 가진 사진을 어떻게 학습시키는지 익혀봅시다!

    상황 설명 =>
    저는 Naver CONNECT 재단에 있는 빨간색의자와 회색의자를 구분하고 싶었어요!
    그런데 사진을 어떻게 가져오고 어떻게 입력해야 되는지 잘 모르겠어요!!

    시작!

    torchvision.datasets.ImageFolder는 내가 가지고 있는 사진을 이용하는 방법입니다.

    가지고 있는 사진을 아래와 같이 구분해 두고 시작합니다.

    지금 폴더에
    origin_data라는 폴더를 만들고!(=> ./origin_data) {./ 에서 . <= 요 점이 현 위치를 의미한다는거 아시죠?}
    origin_data폴더 안에 구분할 의자별 폴더를 만듭니다. (=>./origin_data/red & ./origin_data/\gray)
    이제 각 폴더에 색깔별로 넣으면 됩니다.

    ./origin_data
            |-----red
            |-----gray



    torchvision.datasets.ImageFolder는 다음과 같은 내용을 인자로 받습니다.

    root                          = 내 폴더의 위치를 str 값으로 입력 받음
    transform(optional)           = 입력받을 데이터들을 원하는 형태로 수정하는 방법입니다.
                                    torch는 입력 값이 무조건 tensor여야 하는데 여기서 하면 되겠지요?
                                    모르시겠다면 앞에 xx강을 참조하세요!
    target_transform(optional)    = A function/transform that takes in the target and transforms it.
    loader                        = A function to load an image given its path.
    '''




import torchvision

data=0

if __name__ =="__main__":

    train_data=torchvision.datasets.ImageFolder(root='./origin_data',transform=None)

    for num, value in enumerate(train_data):
        data, label = value
        #data.show()로 사진을 직접 볼수 있습니다. 그런데 이건 한번에 다열리니까 break와 함께 쓰기!
        #data.show()
        #break

        print(num, data, label)